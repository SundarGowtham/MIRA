"""
stratified_difficulty_eval.py
--------------------------------
Answers the question this session ended on: does the current SFT
checkpoint already saturate the (now-patched) validator, or is there real
room for GDPO/GRPO to matter? Runs generation - not ground-truth scoring -
against the RL pool (kononova_triage_results.json's targets), stratified
by a real difficulty ladder, and reports per-tier score distributions.

Reuses evaluate_batched.py's generation harness (load_eval_model,
generate_batch, score_one) rather than reimplementing it. Reconstructs
the exact training-time prompt (SYSTEM_MSG + CLOSED_BOOK_USER from
generate_traces_openrouter.py, copied verbatim - not paraphrased, since
prompt-template drift would confound "is the task hard" with "is the
prompt different from training") and the exact stability_data logic
(get_stability_data), as a SYNCHRONOUS reimplementation using the
validator's own PD access rather than importing the async trace-generation
module directly (avoids pulling in aiohttp/aiofiles/API-key dependencies
that module needs for live generation but this script doesn't).

FIXES A BUG rather than reproducing it: evaluate_batched.py's own
load_validator call points pd_cache_path at "phase_diagrams.pkl" - the
nonexistent monolithic file from the ORIGINAL HANDOFF_2 bug this entire
session started by diagnosing. This script uses pd_index.json (sharded
cache) throughout, like everything else built this session.

Difficulty tiers (computed from real signals already validated this
session, not invented heuristics):
  1_easy      - discrete gradeability, integer stoichiometry, <=2 precursors
  2_medium    - discrete gradeability, integer stoichiometry, 3+ precursors
  3_hard      - interpolated gradeability (fractional/doped - genuinely
                underdetermined composition, no discrete MP entry)
  4_ambiguous - target has a DANGEROUS METASTABLE SIDE-PHASE within 50
                meV/atom of the hull (get_stability_data's own competing-
                phase detection) - real thermodynamic ambiguity, not a
                proxy for it
  ungradeable - can't be thermo-scored at all even now; tracked
                separately, not folded into the difficulty ladder, since a
                low score here could mean "hard" or could mean "PD cache
                still doesn't cover this chemsys," and those are
                different findings

NOT included (deliberately, rather than faked): "naive heuristic produces
a balanceable-but-wrong route" and "genuine frontier (not in MP)" tiers
from the original proposal. The first needs simulating a heuristic
baseline, the second needs a real MP-queried formula set (mp_formula_set
is self-referential, built from this corpus, not from MP - established
earlier this session). Both are real follow-ups, not silently assumed.

Usage:
  uv run python stratified_difficulty_eval.py \
      --checkpoint outputs/sft_v2/final \
      --model Qwen/Qwen3-8B \
      --triage-results kononova_triage_results.json \
      --formula-set data/cache/mp_formula_set.pkl \
      --pd-index data/cache/pd_index.json \
      --project-root . \
      --samples-per-tier 40 \
      --batch-size 8 \
      --out stratified_eval_results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- verbatim from generate_traces_openrouter.py (not paraphrased - prompt
# drift from training would confound difficulty with distribution shift) ---

SYSTEM_MSG = """You are a working materials chemist designing solid-state synthesis routes.

For every target compound, your internal reasoning MUST address:
1. STOICHIOMETRY: Oxidation states and balancing.
2. PRECURSOR CHOICE: Justify reagents.
3. BALANCED EQUATION: Explicit molar coefficients.
4. CONDITIONS: Justify temps/times/atmosphere based on thermodynamics.

You must output your final answer as a pure JSON object matching this schema:

{
  "precursors": [{"formula": "str", "amount": float}],
  "operations": [
    {"type": "str", "temperature_c": float, "time_h": float, "atmosphere": "str", "media": "str"}
  ],
  "thermodynamic_checks": [
    // Each entry must be one of the four structured claim types below.
    // Numeric values will be verified against the Materials Project convex hull.

    {"type": "oxidation_state", "element": "Fe", "avg_valence": 3.5, "requires_atmosphere": "oxidizing"},
    // Use when the target requires a specific cation valence.
    // requires_atmosphere is one of: "oxidizing", "reducing", "inert", "any".

    {"type": "competing_phase", "formula": "LaFeO3", "form_energy_per_atom": -2.85},
    // Use to name a phase that competes with the target during synthesis.
    // form_energy_per_atom is optional but graded against MP if provided.

    {"type": "stoichiometric_constraint", "species": "O2", "moles_per_formula_unit": 0.125, "role": "consumed"},
    // Use for non-precursor species exchanged with the atmosphere.
    // role is one of: "consumed", "released".

    {"type": "hull_stability", "formula": "SrLa(FeO3)2", "e_above_hull": 0.005}
    // Use to flag a metastable competitor by its energy above the convex hull (eV/atom).
  ]
}

The "type" field of each operation MUST be one of the following nine values
(any synonym will be normalized but using these exactly avoids ambiguity):

  - "mix"     — grinding, ball-milling, mortar-and-pestle prep (room temperature)
  - "dry"     — moisture or solvent removal
  - "press"   — pelletization, shaping, pressing into pellets
  - "calcine" — initial heating step; precursor decomposition or reaction
                (use for sealed-tube heating and hydrothermal — describe the
                vessel in "media", not in "type")
  - "sinter"  — final densification heating
  - "anneal"  — intermediate or final heat treatment (post-sinter)
  - "quench"  — rapid cooling (water, oil, gas blast)
  - "cool"    — controlled slow cooling at a defined rate
  - "wash"    — post-synthesis solvent/acid wash



The temperature you specify in heating operations (calcine, sinter, anneal) determines the
thermodynamic regime at which your route is evaluated. Routes whose temperatures don't enable
the proposed chemistry to proceed will be marked as thermodynamically unfavorable.


Output 3-6 thermodynamic claims total. Reference only formulas that appear
in the SYSTEM STABLE PHASES or DANGEROUS METASTABLE SIDE-PHASES sections of
the prompt; do not invent phase names. When a numeric value (formation
energy, e_above_hull) is given in the prompt for a phase you reference,
include that exact number in your claim. Do not include markdown formatting,
backticks, or comments in the final output. Just the JSON object."""

CLOSED_BOOK_USER = """Target: {target}{context}

Thermodynamic Context (Phase Stability Data):
{stability_data}

Provide your synthesis route as a JSON object."""

# Cap on stable/competing phase lines included in stability_data. Without
# this, dense phase diagrams (heavily-studied chemsystems, or ones with
# defect/composition-series entries like LiMn437O874, LiMn438O876...) blow
# the prompt up to 20k+ characters, which was directly causing the model
# to get stuck in repetition loops rather than producing a normal answer -
# discovered by inspecting actual ParseFailure completions, not assumed.
STABILITY_TEXT_MAX_ENTRIES = 15


def _safe_form_e(pd, entry):
    try:
        return float(pd.get_form_energy_per_atom(entry))
    except Exception:
        return None


def get_chemsys(formula: str) -> str | None:
    from pymatgen.core import Composition
    try:
        els = sorted(str(el) for el in Composition(formula).elements)
        return "-".join(els)
    except Exception:
        return None


def classify_atmosphere_sync(pd, target_comp) -> str:
    from pymatgen.core import Element
    MU_O_OXIDIZING_REQUIRED = -1.0
    MU_O_REDUCING_REQUIRED = -3.0
    try:
        all_chempots = pd.get_all_chempots(target_comp)
    except Exception:
        return "ATMOSPHERE: chempot analysis unavailable for this composition."
    if not all_chempots:
        return "ATMOSPHERE: no stability facets returned."
    o_el = Element("O")
    if o_el not in pd.el_refs:
        return "ATMOSPHERE: target contains no oxygen (analysis skipped)."
    o_ref = pd.el_refs[o_el].energy_per_atom
    o_mus_delta = [float(f[o_el]) - float(o_ref) for f in all_chempots.values() if o_el in f]
    if not o_mus_delta:
        return "ATMOSPHERE: oxygen chempot not present on any facet (unusual)."
    mu_lo, mu_hi = min(o_mus_delta), max(o_mus_delta)
    if mu_lo > MU_O_OXIDIZING_REQUIRED:
        return (f"ATMOSPHERE REQUIRED: oxidizing (air or O2). Target requires "
                f"Δμ_O > {mu_lo:.2f} eV relative to O2 reference across all "
                f"stability facets; reducing conditions will decompose it.")
    if mu_hi < MU_O_REDUCING_REQUIRED:
        return (f"ATMOSPHERE REQUIRED: reducing or inert (Ar, N2, H2, vacuum). "
                f"Target lies in low-μ_O regime, Δμ_O ∈ [{mu_lo:.2f}, {mu_hi:.2f}] eV; "
                f"oxidizing conditions will oxidize it away.")
    return (f"ATMOSPHERE: flexible. Target stable across Δμ_O ∈ [{mu_lo:.2f}, {mu_hi:.2f}] eV; "
            f"air, inert, or mildly reducing all acceptable.")


def target_status_line_sync(pd, target: str, target_comp) -> str:
    target_red = target_comp.reduced_formula
    matches = [e for e in pd.all_entries if e.composition.reduced_formula == target_red]
    if matches:
        try:
            best = min(matches, key=lambda e: e.energy_per_atom)
            e_hull = float(pd.get_e_above_hull(best, on_error="ignore"))
            if e_hull <= 0.001:
                return f"TARGET STATUS: {target} is THERMODYNAMICALLY STABLE (on the convex hull)."
            return (f"TARGET STATUS: {target} is METASTABLE (+{e_hull:.3f} eV/atom above hull). "
                    f"Will tend to decompose into more stable phases listed below.")
        except Exception:
            pass
    try:
        decomp, hull_e = pd.get_decomp_and_hull_energy_per_atom(target_comp)
        decomp_str = " + ".join(f"{amt:.3f} {e.composition.reduced_formula}" for e, amt in decomp.items())
        return (f"TARGET STATUS: {target} is a non-discrete composition (no MP entry — "
                f"likely a solid solution or doped phase). At this composition the convex "
                f"hull lies at {hull_e:.3f} eV/atom and decomposes into: {decomp_str}. "
                f"Your synthesis must stabilize the target against this decomposition "
                f"(typically via configurational entropy at high T plus controlled cooling).")
    except Exception:
        return f"TARGET STATUS: stability analysis unavailable for {target}."


def get_stability_data_sync(target: str, validator) -> tuple[str, bool]:
    """Returns (stability_data_text, has_competing_phase_within_50meV)."""
    from pymatgen.core import Composition

    tc = validator.thermo_checker
    chemsys = get_chemsys(target)
    pd, _ = tc._resolve_pd([target]) if chemsys else (None, None)
    if pd is None:
        return "No phase diagram data computed for this system.", False

    try:
        target_comp = Composition(target)
        target_status = target_status_line_sync(pd, target, target_comp)
        atmosphere_hint = classify_atmosphere_sync(pd, target_comp)

        entry_form_e = {id(e): _safe_form_e(pd, e) for e in pd.stable_entries}
        stable_entries_sorted = sorted(
            pd.stable_entries,
            key=lambda e: entry_form_e[id(e)] if entry_form_e[id(e)] is not None else 0.0
        )
        stable_lines = []
        seen_formulas = set()
        for entry in stable_entries_sorted:
            formula = entry.composition.reduced_formula
            if formula in seen_formulas:
                continue  # near-duplicate composition-series entries (e.g. LiMn437O874,
                          # LiMn438O876...) blew prompts up to 20k+ chars before this cap -
                          # that's what was causing the model to get stuck in repetition
                          # loops, not a genuine chemistry-difficulty signal
            seen_formulas.add(formula)
            form_e = entry_form_e[id(entry)]
            if form_e is None:
                continue
            stable_lines.append(f"{formula} (ΔEf={form_e:.2f})")
            if len(stable_lines) >= STABILITY_TEXT_MAX_ENTRIES:
                break

        competing_dict: dict[str, float] = {}
        for entry in pd.unstable_entries:
            try:
                e_above = float(pd.get_e_above_hull(entry, on_error="ignore"))
            except Exception:
                continue
            if e_above is None or e_above >= 0.05:
                continue
            formula = entry.composition.reduced_formula
            if formula not in competing_dict or e_above < competing_dict[formula]:
                competing_dict[formula] = e_above
        # cap AFTER dedup, sorted closest-to-hull first (most chemically relevant warnings)
        competing_sorted = sorted(competing_dict.items(), key=lambda kv: kv[1])[:STABILITY_TEXT_MAX_ENTRIES]
        competing_lines = [f"{form} (+{e:.3f} above hull)" for form, e in competing_sorted]
        has_competing = len(competing_dict) > 0

        text = (
            "--- THERMODYNAMIC PHASE COMPETITION ---\n"
            f"{target_status}\n\n"
            f"{atmosphere_hint}\n\n"
            "SYSTEM STABLE PHASES (Formation Energy in eV/atom):\n"
            f"  {', '.join(stable_lines) if stable_lines else 'None resolved.'}\n\n"
            "DANGEROUS METASTABLE SIDE-PHASES (Energy above hull in eV/atom):\n"
            f"  {', '.join(competing_lines) if competing_lines else 'None within 50 meV/atom threshold.'}"
        )
        return text, has_competing
    except Exception as e:
        return f"[Error reading phase diagram: {e}]", False


def assign_tier(record: dict, has_competing: bool) -> str:
    tier = record.get("thermo_tier")
    if tier == "ungradeable":
        return "ungradeable"
    if tier == "interpolated":
        return "3_hard"
    if tier == "discrete":
        if has_competing:
            return "4_ambiguous"
        n_prec = record.get("n_precursors", 0)
        return "1_easy" if n_prec <= 2 else "2_medium"
    return "unknown"


# ---------------------------------------------------------------------------
# Parallel tiering. MUST run before the model is loaded onto the GPU - forking
# ProcessPoolExecutor workers after CUDA is initialized in the parent process
# is a known hazard (CUDA contexts aren't fork-safe; can hang or corrupt
# silently rather than crash cleanly). Same memory-bounded pattern as
# kononova_triage.py: each worker builds its own validator once, clears its
# own PD cache periodically - a single-threaded, unbounded-cache sweep over
# ~17.6k targets is exactly what OOM'd the original triage run at 93GB/123GB.
# ---------------------------------------------------------------------------
_tier_worker_validator = None
_tier_worker_n_since_clear = 0
_tier_worker_cache_clear_every = 200


def _tier_worker_init(formula_set_path, pd_index_path, project_root, cache_clear_every):
    global _tier_worker_validator, _tier_worker_n_since_clear, _tier_worker_cache_clear_every
    try:
        from core.reward import load_validator
    except ImportError:
        from reward import load_validator
    _tier_worker_validator = load_validator(Path(formula_set_path), Path(pd_index_path), Path(project_root))
    _tier_worker_n_since_clear = 0
    _tier_worker_cache_clear_every = cache_clear_every


def _tier_batch(batch: list[tuple[int, dict]]) -> list[tuple[int, str]]:
    global _tier_worker_n_since_clear
    results = []
    for i, rec in batch:
        _, has_competing = get_stability_data_sync(rec["target"], _tier_worker_validator)
        results.append((i, assign_tier(rec, has_competing)))
        _tier_worker_n_since_clear += 1
        if _tier_worker_n_since_clear >= _tier_worker_cache_clear_every:
            _tier_worker_validator.thermo_checker.phase_diagrams.clear()
            _tier_worker_n_since_clear = 0
    return results


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--triage-results", type=Path, required=True)
    ap.add_argument("--formula-set", type=Path, required=True)
    ap.add_argument("--pd-index", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--samples-per-tier", type=int, default=40)
    ap.add_argument("--pretier-sample-size", type=int, default=3000,
                     help="randomly subsample the candidate pool to this size BEFORE tiering, "
                          "since only ~5x samples-per-tier records are actually needed. "
                          "Set to 0 to tier the full pool (slow, only worth it if a tier is "
                          "rare enough that 3000 candidates might not contain enough of it).")
    ap.add_argument("--tier-workers", type=int, default=8,
                     help="parallel workers for the tiering pass - this is a CPU/PD-cache-bound "
                          "step, same memory-bounded pattern as kononova_triage.py")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=11000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("stratified_eval_results.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.project_root))
    try:
        from core.evaluate_batched import load_eval_model, generate_batch, score_one
    except ImportError:
        from evaluate_batched import load_eval_model, generate_batch, score_one
    try:
        from core.reward import load_validator, parse_completion
    except ImportError:
        from reward import load_validator, parse_completion

    random.seed(args.seed)

    triage = json.loads(args.triage_results.read_text())
    all_targets = [r for r in triage["records"] if r.get("status") == "graded"]
    print(f"{len(all_targets)} candidate targets from triage results.", file=sys.stderr)

    pretier_pool = all_targets
    if args.pretier_sample_size and len(all_targets) > args.pretier_sample_size:
        pretier_pool = random.sample(all_targets, args.pretier_sample_size)
        print(f"Pre-sampled {len(pretier_pool)} of {len(all_targets)} candidates before "
              f"tiering (--pretier-sample-size={args.pretier_sample_size}, only ~"
              f"{5 * args.samples_per_tier} records are actually needed).", file=sys.stderr)

    # --- Parallel tiering, BEFORE the model touches the GPU. Forking workers
    # after CUDA is initialized in the parent is a known hazard - reordered
    # from the original version specifically to avoid that. ---
    print(f"Tiering with {args.tier_workers} workers (memory-bounded, "
          f"each worker clears its own PD cache periodically)...", file=sys.stderr)
    indexed = list(enumerate(pretier_pool))
    tier_batches = list(_chunked(indexed, 20))
    by_tier: dict[str, list[dict]] = defaultdict(list)
    t0 = time.time()
    n_tiered = 0
    with ProcessPoolExecutor(
        max_workers=args.tier_workers, initializer=_tier_worker_init,
        initargs=(str(args.formula_set), str(args.pd_index), str(args.project_root), 200),
    ) as executor:
        futures = [executor.submit(_tier_batch, b) for b in tier_batches]
        for fut in as_completed(futures):
            for i, tier in fut.result():
                by_tier[tier].append(pretier_pool[i])
            n_tiered += len(fut.result())
            if n_tiered % 200 < 20:
                elapsed = time.time() - t0
                rate = n_tiered / elapsed if elapsed > 0 else 0
                print(f"  tiered {n_tiered}/{len(pretier_pool)}  ({rate:.1f} rec/s)", file=sys.stderr)

    print("\ntier sizes (pre-sampled pool):", file=sys.stderr)
    for tier, recs in sorted(by_tier.items()):
        print(f"  {tier:<16} {len(recs)}", file=sys.stderr)
        if len(recs) < args.samples_per_tier:
            print(f"    WARNING: only {len(recs)} available, fewer than "
                  f"--samples-per-tier={args.samples_per_tier}. Consider raising "
                  f"--pretier-sample-size if this tier matters to you.", file=sys.stderr)

    # --- NOW load the model - tiering (and its forked workers) is done. ---
    print("\nLoading model...", file=sys.stderr)
    model, tok = load_eval_model(args.checkpoint, args.model)

    print("Loading validator for generation-time scoring (pd_index.json, not the stale "
          "phase_diagrams.pkl path evaluate_batched.py's own main() uses)...", file=sys.stderr)
    validator = load_validator(args.formula_set, args.pd_index, args.project_root)
    if validator.thermo_checker is None:
        print("FATAL: thermo_checker is None - check --pd-index path.", file=sys.stderr)
        sys.exit(1)

    sampled: dict[str, list[dict]] = {}
    for tier, recs in by_tier.items():
        if tier == "unknown":
            continue
        k = min(args.samples_per_tier, len(recs))
        sampled[tier] = random.sample(recs, k)

    all_results: dict[str, list[dict]] = defaultdict(list)
    t0 = time.time()
    n_done = 0
    n_total = sum(len(v) for v in sampled.values())

    for tier, recs in sampled.items():
        for batch_start in range(0, len(recs), args.batch_size):
            batch_recs = recs[batch_start:batch_start + args.batch_size]
            prompts_full = []
            for rec in batch_recs:
                stability_text, _ = get_stability_data_sync(rec["target"], validator)
                user_msg = CLOSED_BOOK_USER.format(target=rec["target"], context="", stability_data=stability_text)
                prompts_full.append(SYSTEM_MSG + "\n\n" + user_msg)

            completions = generate_batch(model, tok, prompts_full, args)

            for rec, completion in zip(batch_recs, completions):
                # NOT using evaluate_batched.score_one here - its except-clause
                # discards the actual exception (returns bare {"error": 1.0}),
                # which is exactly what made the first stratified run's 52.5%
                # mechanical-failure rate on 3_hard undiagnosable without a
                # re-run. Capture stage + exception type + message instead.
                stage = "parse_completion"
                try:
                    route = parse_completion(completion, rec["target"])
                    stage = "validate"
                    reward, breakdown = validator.validate(route, rec["target"])
                except Exception as e:
                    reward, breakdown = 0.0, {
                        "error": 1.0,
                        "error_stage": stage,
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:300],
                    }
                all_results[tier].append({
                    "target": rec["target"], "reward": reward, "breakdown": breakdown,
                    "completion": completion,
                })
            n_done += len(batch_recs)
            elapsed = time.time() - t0
            print(f"  [{tier}] {n_done}/{n_total}  ({n_done/elapsed:.2f} rec/s)", file=sys.stderr)

    summary = {}
    for tier, results in all_results.items():
        import statistics
        from collections import Counter
        rewards = [r["reward"] for r in results]
        n_mechanical_failure = sum(1 for r in results if r["breakdown"].get("error") == 1.0)
        error_types = Counter(
            r["breakdown"].get("error_type", "unknown")
            for r in results if r["breakdown"].get("error") == 1.0
        )
        summary[tier] = {
            "n": len(rewards),
            "mean_reward": round(statistics.mean(rewards), 4) if rewards else None,
            "median_reward": round(statistics.median(rewards), 4) if rewards else None,
            "frac_pass_0.65": round(sum(1 for r in rewards if r >= 0.65) / len(rewards), 3) if rewards else None,
            "frac_ge_0.9": round(sum(1 for r in rewards if r >= 0.9) / len(rewards), 3) if rewards else None,
            "frac_mechanical_failure": round(n_mechanical_failure / len(rewards), 3) if rewards else None,
            "error_type_counts": dict(error_types),
        }

    print("\n" + "=" * 70)
    print("STRATIFIED DIFFICULTY EVAL — model generation vs. patched validator")
    print("=" * 70)
    for tier in sorted(summary.keys()):
        s = summary[tier]
        print(f"  {tier:<16} n={s['n']:<5} mean={s['mean_reward']}  median={s['median_reward']}  "
              f"pass@0.65={s['frac_pass_0.65']}  ge0.9={s['frac_ge_0.9']}  "
              f"mech_fail={s['frac_mechanical_failure']}")
        if s["error_type_counts"]:
            print(f"    error types: {s['error_type_counts']}")

    args.out.write_text(json.dumps({"summary": summary, "results": dict(all_results)}, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()