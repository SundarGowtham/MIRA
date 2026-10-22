#!/usr/bin/env python
"""
research/phase17_ra_symmetry_axioms.py — Phase 17 Block R, R-a
(symmetry): a fixed metamorphic-test suite for the 7 axioms a
chemistry reward must satisfy (PHASE17_PREREG.md's R-a table), run
against the CURRENT validator and against each historical commit
where a named bug was present, via read-only `git worktree`s (never a
checkout of the working tree).

GIT ARCHAEOLOGY (done once, here, not re-derived at runtime): the 6
named historical bugs in the pre-reg's R-a table map to 4 validator.py
commits, in this chronological order (confirmed via `git show` on each
diff, not guessed from commit messages alone):
  1. 6b2f63c (2026-07-15) fixes SCALE invariance.
  2. c3d6547 (2026-07-20) fixes NOTATION invariance (hydrate middle dot).
  3. 32c9e45 (2026-07-21) fixes BOTH token identity (atmosphere
     substring trap, "h2" inside "h2o") AND monotonicity (unknown
     operation types scored as a free 1.0) in the same commit -- also
     covers notation invariance's "atmosphere synonyms" sub-case
     (same tokenizer rewrite).
  4. 923f311 (2026-07-22) fixes ABSTENTION (None-propagation: sentinel-
     tagged checks no longer pay a 0.5 into the weighted reward).
Each worktree below is built at the PARENT of its fix commit (bug
still present). Because these 4 fixes landed sequentially, an earlier
worktree is buggy on its own axis AND every later axis too (none of
those fixes exist yet at that point) -- a staircase, not independent
cells. EXPECTED_VIOLATIONS below encodes this explicitly rather than
re-deriving it from dates at runtime.

BOUNDARY axiom ("0.0 treated as missing"): git archaeology found NO
historical commit changing this. `_check_target_stability` has used
`if e_hull is None` (never a falsy/truthy check that would conflate a
genuine e_above_hull=0.0 with "missing") in every commit checked here,
including the earliest (6b2f63c~1, 2026-07-15). Reported as a
forward-looking correctness check only -- no historical regression to
catch, unlike the other 5 named bugs. Flagged, not fabricated.

ORDER invariance: the pre-reg's own table lists no historical bug for
this axiom ("--"). Tested here as a forward-looking check only
(precursor-order permutation; operation-order permutation is
deliberately OUT of scope -- see "order" case construction below).

ABSTENTION's verdict is computed, not just observed: from each
version's OWN live output (the `scores` dict + its public `weights`
attribute), this script evaluates BOTH the pre-fix aggregation formula
(every computed channel counted, sentinel tag or not) and the post-fix
one (sentinel-tagged channels excluded, weights renormalized) and
checks which one the version's actual returned `reward` matches. This
avoids any cross-version raw-reward-number comparison (contaminated by
unrelated weight-table changes over the same period) while still being
an exact, not fuzzy, diagnostic.

NO EDIT to validator.py (every worktree is read-only; the main repo's
validator.py is never touched). NO NEW GENERATION -- the 200 base
routes are drawn from data/novelty/rssft_reference_{astral,heldout}.jsonl,
already-existing samples, per the pre-reg's own instruction.

Usage (tmux, CPU only -- git worktree + pymatgen/PD-cache calls):
  PYTHONPATH=. uv run python research/phase17_ra_symmetry_axioms.py
"""
from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.reward import ParseFailure, parse_completion  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
SEED = 20261016
N_BASE_ROUTES = 200
OUT_PATH = REPO_ROOT / "results" / "phase17" / "ra_symmetry_axioms.json"
SCRATCH = Path("/tmp/claude-872600018/-users-gsundar-projects-MIRA/2ee586f5-2384-4590-b7a1-09974934c2f3/scratchpad")
WORKTREE_ROOT = SCRATCH / "ra_worktrees"
RUNNER = SCRATCH / "_ra_runner.py"
FORMULA_SET_PATH = REPO_ROOT / "data" / "cache" / "mp_formula_set.pkl"
PD_INDEX_PATH = REPO_ROOT / "data" / "cache" / "pd_index.json"
SENTINEL_TAGS_FALLBACK = {"ungradeable", "sentinel_no_entry", "no_balance_found", "no_thermo_checker"}

SOURCES = [
    REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl",
    REPO_ROOT / "data" / "novelty" / "rssft_reference_heldout.jsonl",
]

# version label -> git ref (parent of the fix commit = bug still present); "current" is run
# directly against the main repo's checked-out HEAD, no worktree needed.
WORKTREE_REFS = {
    "v0_pre_scale_fix_2026-07-15": "6b2f63c~1",
    "v1_pre_hydrate_fix_2026-07-20": "c3d6547~1",
    "v2_pre_atmtoken_monotonicity_fix_2026-07-21": "32c9e45~1",
    "v3_pre_noneprop_fix_2026-07-22": "923f311~1",
}
VERSION_ORDER = list(WORKTREE_REFS.keys()) + ["current_HEAD"]

# Staircase ground truth from the archaeology above: which (axiom, version) cells are
# EXPECTED to still show the bug. "order" and "boundary" have no historical bug (empty sets).
EXPECTED_BUGGY_VERSIONS = {
    "scale": {"v0_pre_scale_fix_2026-07-15"},
    "notation_hydrate": {"v0_pre_scale_fix_2026-07-15", "v1_pre_hydrate_fix_2026-07-20"},
    "notation_atmosphere_synonym": {"v0_pre_scale_fix_2026-07-15", "v1_pre_hydrate_fix_2026-07-20",
                                     "v2_pre_atmtoken_monotonicity_fix_2026-07-21"},
    "token_identity": {"v0_pre_scale_fix_2026-07-15", "v1_pre_hydrate_fix_2026-07-20",
                        "v2_pre_atmtoken_monotonicity_fix_2026-07-21"},
    "monotonicity": {"v0_pre_scale_fix_2026-07-15", "v1_pre_hydrate_fix_2026-07-20",
                      "v2_pre_atmtoken_monotonicity_fix_2026-07-21"},
    "abstention": {"v0_pre_scale_fix_2026-07-15", "v1_pre_hydrate_fix_2026-07-20",
                   "v2_pre_atmtoken_monotonicity_fix_2026-07-21", "v3_pre_noneprop_fix_2026-07-22"},
    "order": set(),
    "boundary": set(),
}
NAMED_HISTORICAL_BUGS = ["scale", "notation_hydrate", "notation_atmosphere_synonym",
                          "token_identity", "monotonicity", "abstention"]  # the 6 in the pre-reg table


# ---------------------------------------------------------------------------
# Load base routes (existing samples, no new generation)
# ---------------------------------------------------------------------------

def load_base_routes(n: int) -> list[dict]:
    routes = []
    for path in SOURCES:
        if len(routes) >= n:
            break
        with path.open() as f:
            for line in f:
                if len(routes) >= n:
                    break
                if not line.strip():
                    continue
                rec = json.loads(line)
                target = rec.get("target")
                completion = rec.get("completion")
                if not target or not completion:
                    continue
                try:
                    route = parse_completion(completion, target)
                except Exception:
                    continue
                precursors = [{"formula": p.formula, "amount": getattr(p, "amount", 1.0)}
                              for p in (route.precursors or [])]
                if not precursors:
                    continue
                operations = []
                for op in (route.operations or []):
                    c = op.conditions
                    operations.append({
                        "type": op.type,
                        "conditions": {
                            "heating_temperature": list(getattr(c, "heating_temperature", []) or []),
                            "heating_time": list(getattr(c, "heating_time", []) or []),
                            "heating_atmosphere": list(getattr(c, "heating_atmosphere", []) or []),
                            "mixing_media": getattr(c, "mixing_media", None),
                            "atmosphere": getattr(c, "atmosphere", None),
                        },
                    })
                routes.append({"target": target, "precursors": precursors, "operations": operations})
    return routes


# ---------------------------------------------------------------------------
# Case construction (one dict of named variants per case; compared pairwise below)
# ---------------------------------------------------------------------------

def build_cases(base_routes: list[dict]) -> tuple[dict, dict]:
    """Returns (cases, case_meta). `cases[case_id]` = {variant_name: route_dict}.
    `case_meta[case_id]` = {"axiom": ..., "compare": spec}."""
    rng = random.Random(SEED)
    cases: dict[str, dict] = {}
    meta: dict[str, dict] = {}

    # routes with at least one operation at all (used for atmosphere-string tests,
    # which inject/overwrite the FIRST operation's heating_atmosphere regardless of
    # whether the model itself declared a heating step -- a controlled synthetic
    # perturbation, not a claim about what the model actually wrote).
    with_ops = [r for r in base_routes if r["operations"]]

    for i, r in enumerate(base_routes):
        cid = f"order_{i}"
        variant = copy.deepcopy(r)
        rng.shuffle(variant["precursors"])
        cases[cid] = {"orig": r, "shuffled": variant}
        meta[cid] = {"axiom": "order", "compare": "equal_channels",
                     "channels": ["stoichiometry", "amount_accuracy", "charge_neutrality", "precursors_exist"]}

    for i, r in enumerate(base_routes):
        cid = f"scale_{i}"
        variant = copy.deepcopy(r)
        for p in variant["precursors"]:
            p["amount"] = round(p["amount"] * 0.37, 6)
        cases[cid] = {"orig": r, "scaled": variant}
        meta[cid] = {"axiom": "scale", "compare": "equal_channels",
                     "channels": ["stoichiometry", "amount_accuracy", "charge_neutrality",
                                  "precursors_exist", "operation_order", "temperature_plausible"]}

    for i, r in enumerate(base_routes):
        if not r["precursors"]:
            continue
        cid = f"notation_hydrate_{i}"
        base_formula = r["precursors"][0]["formula"]
        dot_variant = copy.deepcopy(r)
        dot_variant["precursors"][0]["formula"] = f"{base_formula}·1H2O"
        paren_variant = copy.deepcopy(r)
        paren_variant["precursors"][0]["formula"] = f"{base_formula}(H2O)1"
        cases[cid] = {"dot": dot_variant, "parenthetical": paren_variant}
        meta[cid] = {"axiom": "notation_hydrate", "compare": "equal_channels",
                     "channels": ["stoichiometry", "amount_accuracy", "charge_neutrality",
                                  "precursors_exist", "target_stability", "thermodynamic_favorable"]}

    atmosphere_pool = with_ops if with_ops else base_routes
    for i, r in enumerate(atmosphere_pool[:min(60, len(atmosphere_pool))]):
        cid = f"notation_atm_synonym_{i}"
        ar_variant = copy.deepcopy(r)
        ar_variant["operations"][0]["conditions"]["heating_atmosphere"] = ["Ar"]
        argon_variant = copy.deepcopy(r)
        argon_variant["operations"][0]["conditions"]["heating_atmosphere"] = ["argon"]
        cases[cid] = {"ar": ar_variant, "argon": argon_variant}
        meta[cid] = {"axiom": "notation_atmosphere_synonym", "compare": "equal_channels",
                     "channels": ["chempot_atmosphere"]}

        cid2 = f"token_identity_{i}"
        h2_variant = copy.deepcopy(r)
        h2_variant["operations"][0]["conditions"]["heating_atmosphere"] = ["H2"]
        h2o_variant = copy.deepcopy(r)
        h2o_variant["operations"][0]["conditions"]["heating_atmosphere"] = ["H2O"]
        o2_variant = copy.deepcopy(r)
        o2_variant["operations"][0]["conditions"]["heating_atmosphere"] = ["O2"]
        cases[cid2] = {"h2_reducer": h2_variant, "h2o_trap": h2o_variant, "o2_oxidizer": o2_variant}
        meta[cid2] = {"axiom": "token_identity", "compare": "trap_vs_reducer_oxidizer",
                      "channel": "chempot_atmosphere"}

    # CORRECTED construction (see module note "MONOTONICITY TEST FIX" below):
    # the real historical bug/fix operates at the assessed==0 level (a route
    # of ENTIRELY unrecognized op types), not at "append one garbage op to an
    # otherwise-good sequence" -- _check_operation_order EXCLUDES any pair
    # touching an unrecognized type from both `assessed` and `violations`,
    # so appending a single unknown op to a clean route changes nothing (the
    # new pair is simply skipped, confirmed by direct inspection of
    # validator.py's current source). Only when NO pair is assessable at all
    # does the function fall back to a fixed return (1.0 pre-fix bug, 0.5
    # post-fix, per its own docstring/comment).
    for i, r in enumerate(base_routes):
        cid = f"monotonicity_{i}"
        variant = copy.deepcopy(r)
        variant["operations"] = [
            {"type": "frobnicate_unknown_op_phase17ra_a", "conditions": {
                "heating_temperature": [], "heating_time": [], "heating_atmosphere": [],
                "mixing_media": None, "atmosphere": None}},
            {"type": "frobnicate_unknown_op_phase17ra_b", "conditions": {
                "heating_temperature": [], "heating_time": [], "heating_atmosphere": [],
                "mixing_media": None, "atmosphere": None}},
        ]
        cases[cid] = {"all_unknown_ops": variant}
        meta[cid] = {"axiom": "monotonicity", "compare": "exact_value_check",
                     "channel": "operation_order", "buggy_value": 1.0, "fixed_value": 0.5}

    for i, r in enumerate(base_routes):
        cid = f"abstention_{i}"
        variant = copy.deepcopy(r)
        variant["precursors"] = [{"formula": "Og2Rn3", "amount": 1.0}]  # shares no elements with any real target
        cases[cid] = {"no_balance": variant}
        meta[cid] = {"axiom": "abstention", "compare": "aggregation_formula_match"}

    cases["boundary_cu_metal"] = {"cu_on_hull": {"target": "Cu", "precursors": [{"formula": "Cu", "amount": 1.0}],
                                                   "operations": [{"type": "mix", "conditions": {
                                                       "heating_temperature": [], "heating_time": [],
                                                       "heating_atmosphere": [], "mixing_media": None,
                                                       "atmosphere": None}}]}}
    meta["boundary_cu_metal"] = {"axiom": "boundary", "compare": "exact_zero_not_missing",
                                  "channel": "target_stability"}

    return cases, meta


# ---------------------------------------------------------------------------
# Worktree management + subprocess scoring
# ---------------------------------------------------------------------------

def setup_worktree(label: str, ref: str) -> Path:
    wt_dir = WORKTREE_ROOT / label
    if wt_dir.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(wt_dir)],
                        cwd=str(REPO_ROOT), capture_output=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(wt_dir), ref],
                    cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
    return wt_dir


def teardown_worktree(label: str):
    wt_dir = WORKTREE_ROOT / label
    subprocess.run(["git", "worktree", "remove", "--force", str(wt_dir)],
                    cwd=str(REPO_ROOT), capture_output=True)


def score_version(worktree_dir: Path, cases_path: Path, out_path: Path) -> dict:
    result = subprocess.run(
        ["uv", "run", "python", str(RUNNER), str(worktree_dir), str(cases_path), str(out_path),
         str(FORMULA_SET_PATH), str(PD_INDEX_PATH), str(REPO_ROOT)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"runner failed (exit {result.returncode}): "
                            f"stdout={result.stdout[-2000:]} stderr={result.stderr[-2000:]}")
    print(f"  {result.stdout.strip()}", flush=True)
    return json.loads(out_path.read_text())


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def evaluate_case(case_id: str, m: dict, variants_out: dict) -> dict:
    compare = m["compare"]
    if any("error" in v for v in variants_out.values()):
        return {"skipped_error": {k: v.get("error") for k, v in variants_out.items() if "error" in v}}

    if compare == "equal_channels":
        names = list(variants_out.keys())
        a, b = variants_out[names[0]], variants_out[names[1]]
        diffs = {}
        for c in m["channels"]:
            va, vb = a["breakdown"].get(c), b["breakdown"].get(c)
            if numeric(va) and numeric(vb):
                diffs[c] = round(va - vb, 6)
        violation = any(abs(d) > 1e-6 for d in diffs.values()) if diffs else False
        return {"diffs": diffs, "n_comparable_channels": len(diffs), "violation": violation}

    if compare == "trap_vs_reducer_oxidizer":
        ch = m["channel"]
        v_red = variants_out["h2_reducer"]["breakdown"].get(ch)
        v_trap = variants_out["h2o_trap"]["breakdown"].get(ch)
        v_ox = variants_out["o2_oxidizer"]["breakdown"].get(ch)
        if not (numeric(v_red) and numeric(v_trap) and numeric(v_ox)):
            return {"skipped": "channel not gradeable (likely no PD coverage for this target)",
                    "values": {"h2": v_red, "h2o": v_trap, "o2": v_ox}}
        matches_reducer = abs(v_trap - v_red) < 1e-6
        matches_oxidizer = abs(v_trap - v_ox) < 1e-6
        violation = matches_reducer and not matches_oxidizer
        return {"values": {"h2": v_red, "h2o": v_trap, "o2": v_ox},
                "matches_reducer": matches_reducer, "matches_oxidizer": matches_oxidizer,
                "violation": violation}

    if compare == "exact_value_check":
        ch = m["channel"]
        v = variants_out["all_unknown_ops"]["breakdown"].get(ch)
        if not numeric(v):
            return {"skipped": "channel not gradeable", "value": v}
        matches_buggy = abs(v - m["buggy_value"]) < 1e-6
        matches_fixed = abs(v - m["fixed_value"]) < 1e-6
        return {"value": v, "matches_buggy_value": matches_buggy, "matches_fixed_value": matches_fixed,
                "violation": matches_buggy and not matches_fixed}

    if compare == "aggregation_formula_match":
        v = variants_out["no_balance"]
        scores = v["breakdown"]
        reward_actual = v["reward"]
        weights = variants_out.get("_weights", {})
        sentinel_tags = set(variants_out.get("_sentinel_tags", SENTINEL_TAGS_FALLBACK))
        buggy_active = {k: w for k, w in weights.items() if k in scores and numeric(scores.get(k))}
        buggy_sum = sum(buggy_active.values())
        buggy_reward = (round(sum((w / buggy_sum) * scores[k] for k, w in buggy_active.items()), 4)
                        if buggy_sum else 0.0)
        fixed_active = {k: w for k, w in weights.items()
                        if k in scores and numeric(scores.get(k))
                        and scores.get(f"{k}_gradeability") not in sentinel_tags}
        fixed_sum = sum(fixed_active.values())
        fixed_reward = (round(sum((w / fixed_sum) * scores[k] for k, w in fixed_active.items()), 4)
                        if fixed_sum else 0.0)
        matches_buggy = abs(reward_actual - buggy_reward) < 1e-4
        matches_fixed = abs(reward_actual - fixed_reward) < 1e-4
        stoich_tag = scores.get("stoichiometry_gradeability")
        return {"reward_actual": reward_actual, "buggy_formula_reward": buggy_reward,
                "fixed_formula_reward": fixed_reward, "matches_buggy_formula": matches_buggy,
                "matches_fixed_formula": matches_fixed, "stoichiometry_gradeability_tag": stoich_tag,
                "violation": matches_buggy and not matches_fixed}

    if compare == "exact_zero_not_missing":
        ch = m["channel"]
        v = variants_out["cu_on_hull"]
        score = v["breakdown"].get(ch)
        tag = v["breakdown"].get(f"{ch}_gradeability")
        violation = tag != "discrete" or not numeric(score) or abs(score - 1.0) > 1e-6
        return {"score": score, "gradeability_tag": tag, "violation": violation}

    return {"error": f"unknown compare spec {compare}"}


def main():
    print("loading base routes (existing samples, no new generation)...", flush=True)
    base_routes = load_base_routes(N_BASE_ROUTES)
    print(f"  loaded {len(base_routes)} base routes", flush=True)

    cases, meta = build_cases(base_routes)
    print(f"  built {len(cases)} metamorphic cases across {len(set(m['axiom'] for m in meta.values()))} axioms",
          flush=True)

    cases_path = SCRATCH / "ra_cases.json"
    cases_path.write_text(json.dumps(cases))

    all_results = {}  # version -> case_id -> variant -> {reward, breakdown}
    axiom_version_summary = {}  # axiom -> version -> {n_cases, n_violations, examples}

    for version in VERSION_ORDER:
        print(f"\n=== scoring version: {version} ===", flush=True)
        out_path = SCRATCH / f"ra_scored_{version}.json"
        if version == "current_HEAD":
            worktree_dir = REPO_ROOT
            scored = score_version(worktree_dir, cases_path, out_path)
        else:
            ref = WORKTREE_REFS[version]
            worktree_dir = setup_worktree(version, ref)
            try:
                scored = score_version(worktree_dir, cases_path, out_path)
            finally:
                teardown_worktree(version)
        weights = scored.get("weights", {})
        sentinel_tags = scored.get("sentinel_tags", list(SENTINEL_TAGS_FALLBACK))
        results = scored.get("results", {})
        for cid in results:
            results[cid]["_weights"] = weights
            results[cid]["_sentinel_tags"] = sentinel_tags
        all_results[version] = results

    print("\n" + "=" * 78, flush=True)
    print("Evaluating metamorphic comparisons per case, per version...", flush=True)
    case_evaluations = {}  # version -> case_id -> eval dict
    for version in VERSION_ORDER:
        case_evaluations[version] = {}
        for cid, m in meta.items():
            variants_out = all_results[version].get(cid, {})
            case_evaluations[version][cid] = evaluate_case(cid, m, variants_out)

    for axiom in set(m["axiom"] for m in meta.values()):
        axiom_version_summary[axiom] = {}
        case_ids_for_axiom = [cid for cid, m in meta.items() if m["axiom"] == axiom]
        for version in VERSION_ORDER:
            evals = [case_evaluations[version][cid] for cid in case_ids_for_axiom]
            n_total = len(evals)
            n_skipped = sum(1 for e in evals if "skipped" in e or "skipped_error" in e)
            n_scored = n_total - n_skipped
            n_violations = sum(1 for e in evals if e.get("violation") is True)
            examples = [cid for cid, e in zip(case_ids_for_axiom, evals) if e.get("violation") is True][:3]
            axiom_version_summary[axiom][version] = {
                "n_cases": n_total, "n_scored": n_scored, "n_skipped": n_skipped,
                "n_violations": n_violations,
                "violation_rate_of_scored": round(n_violations / n_scored, 3) if n_scored else None,
                "example_violating_cases": examples,
                "expected_buggy": version in EXPECTED_BUGGY_VERSIONS.get(axiom, set()),
            }
            print(f"  {axiom:<32}{version:<42}n_scored={n_scored:<4}violations={n_violations:<4}"
                  f"expected_buggy={axiom_version_summary[axiom][version]['expected_buggy']}", flush=True)

    # Pre-registered predictions.
    bugs_flagged_at_expected_version = 0
    for axiom in NAMED_HISTORICAL_BUGS:
        expected = EXPECTED_BUGGY_VERSIONS[axiom]
        if not expected:
            continue
        earliest_expected = sorted(expected, key=lambda v: VERSION_ORDER.index(v))[-1]  # version right before the fix
        if axiom_version_summary[axiom][earliest_expected]["n_violations"] > 0:
            bugs_flagged_at_expected_version += 1
    prediction_1 = {
        "statement": "The suite flags at least 4 of the 6 named historical bugs at the versions where they existed.",
        "n_flagged": bugs_flagged_at_expected_version, "n_total_named_bugs": len(NAMED_HISTORICAL_BUGS),
        "passes": bugs_flagged_at_expected_version >= 4,
    }
    n_current_violations = sum(axiom_version_summary[ax]["current_HEAD"]["n_violations"]
                               for ax in axiom_version_summary)
    prediction_2 = {
        "statement": "It flags none on the current version.",
        "n_violations_on_current": n_current_violations,
        "passes": n_current_violations == 0,
    }
    print(f"\nPrediction 1 (>=4/6 historical bugs flagged): {prediction_1}", flush=True)
    print(f"Prediction 2 (zero violations on current): {prediction_2}", flush=True)

    report = {
        "prereg_commit": PREREG_COMMIT,
        "block": "R",
        "subproperty": "R-a (symmetry axioms)",
        "seed": SEED,
        "n_base_routes": len(base_routes),
        "n_cases": len(cases),
        "method": "git worktree per historical commit (parent of each fix), read-only, never a checkout "
                  "of the working tree; subprocess-per-version scoring to avoid cross-version import caching",
        "worktree_refs": WORKTREE_REFS,
        "boundary_caveat": "No historical commit found changing _check_target_stability's None-check "
                           "(is-None, never falsy) across any version tested here, including the earliest "
                           "(6b2f63c~1, 2026-07-15) -- reported as a forward-looking correctness check only.",
        "order_caveat": "Operation-order permutation deliberately excluded (would conflate with "
                        "operation_order's own correctness check, which legitimately penalizes a bad "
                        "order); only precursor-order permutation tested. Pre-reg table lists no "
                        "historical bug for this axiom.",
        "expected_buggy_versions": {k: sorted(v) for k, v in EXPECTED_BUGGY_VERSIONS.items()},
        "axiom_version_summary": axiom_version_summary,
        "prediction_1_at_least_4_of_6": prediction_1,
        "prediction_2_zero_on_current": prediction_2,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
