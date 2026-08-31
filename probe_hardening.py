#!/usr/bin/env python
"""
probe_hardening.py — does constraining the task revive the dead reward channels?

PRE-REGISTERED HYPOTHESIS
-------------------------
Reward capacity is low (14.3%: total within-group z-variance 1.43 of 10) not
because the checks are badly designed, but because the policy's variation lies
along directions the validator is insensitive to. 54.6% of groups already contain
>=2 distinct precursor sets, yet charge_neutrality is 100% constant within group
and stoichiometry 83.5% constant.

PREDICTION: constraints that make the model's existing variation score-relevant
will raise reward capacity.
  - distinct routes/group:  2.01  ->  3+
  - reward capacity:        14%   ->  40%+

PRIMARY METRIC IS CAPACITY, NOT p-hat. A condition that lowers p-hat without
raising capacity has made the task harder without making it more learnable --
that is a failure, not a success.

DECISION RULE (set before running):
  capacity > 40% in any condition  -> that condition is run-4's task
  all conditions < 25%             -> the diagnosis is wrong; the task does not
                                      support factored-reward RL, which is
                                      itself the strongest form of the result

CONDITIONS
  0 baseline            replicate, no constraint
  1 temp_ceiling        "max temperature X C" (per-target, from literature route)
  2 inventory           "use only these reagents" (lit precursors + decoys)
  3 atmosphere          "air only" / restricted atmosphere
  4 combined            all three
  5 low_temp            soft "prefer lower T" preference + temperature_economy
                        reward channel, gated on thermodynamic_favorable. Round-1
                        result: capacity 23.8%, just under the 25% floor --
                        round 2 (default T_ref_margin=150/T_span=400, tightened
                        from round 1's 300/600) is Fix A in
                        misc/some_claude_files/post_low_temp_probe_steps.md.
                        Opt-in, not in the default --conditions list.
  6 low_temp_ceiling    Fix B: low_temp's soft preference AND a hard per-target
                        ceiling (temp_ceiling's construction) together; economy
                        is banded to [lit_T, ceiling] via
                        compute_temperature_economy_banded, not a fixed margin --
                        round 1 found the ceiling gives the behavioral pull
                        (100% compliance) the soft preference alone didn't.
  7 low_temp_inventory  low_temp + inventory (round 1's low_temp_combined,
                        renamed for round 2's naming)

Usage:
    python probe_hardening.py --n-targets 40 --samples 8 --out misc/hardening.json
    python probe_hardening.py --conditions baseline temp_ceiling --n-targets 20
    python probe_hardening.py --analyze-only --out misc/hardening.json
    python probe_hardening.py \
        --conditions baseline low_temp low_temp_ceiling low_temp_inventory \
        --reuse-pool-from misc/hardening.json --out misc/hardening_low_temp2.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_batched import load_eval_model, generate_batch  # noqa: E402
from core.reward import parse_completion, load_validator  # noqa: E402
from stratified_difficulty_eval import SYSTEM_MSG  # noqa: E402
from validator import ThermoChecker  # noqa: E402

CONDITIONS = ["baseline", "temp_ceiling", "inventory", "atmosphere", "combined"]
# misc/some_claude_files/post_low_temp_probe_steps.md Task 1: low_temp round 2.
# low_temp_ceiling combines Fix A (rescaled economy) with Fix B (hard ceiling
# provides the behavioral pull the soft preference alone didn't); its economy
# term is banded to [lit_T, ceiling], not [lit_T, lit_T+margin], via
# compute_temperature_economy_banded. low_temp_inventory (was
# low_temp_combined in round 1) is low_temp + inventory.
LOW_TEMP_CONDITIONS = ["low_temp", "low_temp_ceiling", "low_temp_inventory"]
ALL_CONDITIONS = CONDITIONS + LOW_TEMP_CONDITIONS

# misc/some_claude_files/low_temperature_objective_SPEC.md: a soft preference,
# not a numeric ceiling -- temp_ceiling already showed a stated ceiling gets
# 100% compliance and therefore zero variance. The point here is a gradient.
LOW_TEMP_HINT = (
    "Constraint: prefer the lowest processing temperature that still allows "
    "the reaction to proceed. Lower maximum temperature is better, provided "
    "the route remains thermodynamically favorable."
)


# --------------------------------------------------------------------------
# Constraint construction
# --------------------------------------------------------------------------

def load_literature(triage_path: Path, synthesis_path: Path) -> dict:
    """
    target -> {"precursors": [...], "max_T": float|None}

    The triage file (misc/kononova_triage_results3.json) only carries scores
    (reward, stoichiometry, ...) and an "idx" -- it has no precursor or
    temperature fields. Those live in the raw source file named in its
    meta.synthesis_file (data/raw/synthesis_clean.json), a flat list where
    "idx" is a direct positional index (see data_curation/kononova_triage.py,
    which builds the triage file by enumerate()-ing that same list). We only
    keep targets whose triage status is "graded", i.e. the literature route
    actually built into a validator-scorable Route.
    """
    lit = {}
    lit_reward = {}
    raw = json.loads(triage_path.read_text())
    records = raw if isinstance(raw, list) else raw.get("records", raw.get("results", []))
    synthesis = json.loads(synthesis_path.read_text())

    for r in records:
        if not isinstance(r, dict) or r.get("status") != "graded":
            continue
        idx = r.get("idx")
        if idx is None or not (0 <= idx < len(synthesis)):
            continue
        synth = synthesis[idx]
        target = synth.get("target_formula") or r.get("target")
        if not target:
            continue
        # Multiple papers per target: keep the route the TRIAGE scored
        # highest, not the last one seen — last-write-wins was surfacing
        # doped-variant routes (e.g. Pr/Mg nitrates for SrAl12O19), which
        # breaks the inventory condition's feasibility guarantee.
        rw = r.get("reward")
        rw = float(rw) if isinstance(rw, (int, float)) else -1.0
        if target in lit and rw <= lit_reward.get(target, -1.0):
            continue

        names = []
        for p in synth.get("precursors") or []:
            f = p.get("formula") if isinstance(p, dict) else p
            if f:
                names.append(str(f).strip())

        vals = []
        for op in synth.get("operations") or []:
            for t in op.get("heating_temperature") or []:
                ts = t if isinstance(t, list) else [t]
                for x in ts:
                    try:
                        vals.append(float(x))
                    except (TypeError, ValueError):
                        pass
        max_T = max(vals) if vals else None
        n_ops = len(synth.get("operations") or [])

        lit[target] = {"precursors": names, "max_T": max_T, "n_ops": n_ops}
        lit_reward[target] = rw
    return lit


def build_decoy_pool(lit: dict, top_n: int = 200) -> list[str]:
    """Frequency-weighted precursor vocabulary, for plausible inventory decoys."""
    counts = defaultdict(int)
    for v in lit.values():
        for p in v["precursors"]:
            counts[p] += 1
    return [p for p, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]]


def make_constraints(target: str, lit_rec: dict, pool: list[str],
                     rng: random.Random, inv_size: int = 16) -> dict:
    """
    Per-target constraint text.

    Design notes:
      - Temperature ceiling is PER TARGET (lit_T + margin). A global ceiling is
        trivially satisfiable for low-T targets and impossible for high-T ones,
        which would confound difficulty with target chemistry.
      - Inventory always contains the literature precursors, so a valid route is
        guaranteed to exist; decoys force discrimination rather than recall.
    """
    out = {}

    max_T = lit_rec.get("max_T")
    if max_T:
        ceiling = int(round((max_T + 100) / 50.0) * 50)
        out["temp_ceiling"] = (
            f"Constraint: no processing step may exceed {ceiling} C."
        )
        out["_ceiling_value"] = float(ceiling)

    lit_prec = [p for p in lit_rec.get("precursors", []) if p]
    if lit_prec:
        decoys = [p for p in pool if p not in lit_prec]
        rng.shuffle(decoys)
        inv = lit_prec + decoys[: max(0, inv_size - len(lit_prec))]
        rng.shuffle(inv)
        out["inventory"] = (
            "Constraint: use only reagents from this inventory: "
            + ", ".join(inv) + "."
        )
        out["_inventory_list"] = inv

    out["atmosphere"] = (
        "Constraint: only ambient air is available. No inert gas, no flowing "
        "O2, no reducing atmosphere."
    )
    return out


def build_prompt(target: str, condition: str, constraints: dict,
                 base_prompt_fn) -> str:
    """
    Closed-book base prompt + constraint lines.

    TODO: point base_prompt_fn at whatever probe_passk.py uses to render a
    closed-book prompt so this probe is prompt-identical to the baselines.
    """
    prompt = base_prompt_fn(target)
    lines = []
    if condition in ("temp_ceiling", "combined", "low_temp_ceiling") \
            and "temp_ceiling" in constraints:
        lines.append(constraints["temp_ceiling"])
    if condition in ("inventory", "combined") and "inventory" in constraints:
        lines.append(constraints["inventory"])
    if condition in ("atmosphere", "combined"):
        lines.append(constraints["atmosphere"])
    if condition in ("low_temp", "low_temp_ceiling", "low_temp_inventory"):
        lines.append(LOW_TEMP_HINT)
    if condition == "low_temp_inventory" and "inventory" in constraints:
        lines.append(constraints["inventory"])
    if lines:
        prompt = prompt.rstrip() + "\n\n" + "\n".join(lines)
    return prompt


# --------------------------------------------------------------------------
# Post-hoc constraint adherence (NOT a validator edit -- user owns validator)
# --------------------------------------------------------------------------

def route_max_T(route) -> float | None:
    """Max processing temperature reported anywhere in the route, or None."""
    temps = []
    for op in getattr(route, "operations", []) or []:
        # PredictedOperation: temps live in conditions.heating_temperature
        conds = getattr(op, "conditions", None)
        vals = getattr(conds, "heating_temperature", None) or []
        for t in (vals if isinstance(vals, list) else [vals]):
            try:
                if t is not None:
                    temps.append(float(t))
            except (TypeError, ValueError):
                pass
    return max(temps) if temps else None


def check_adherence(route, constraints: dict, condition: str) -> dict:
    """Did the model actually respect the constraint? Scored outside the reward.

    max_T_reported is recorded for EVERY condition, including baseline
    (post_low_temp_probe_steps.md Task 1: "Fix the baseline instrumentation
    first" -- round 1 only tracked it for temp-constrained conditions, so
    there was no baseline number to compare low_temp's reported T against).
    """
    res = {}
    if route is None:
        return res

    res["max_T_reported"] = route_max_T(route)

    if condition in ("temp_ceiling", "combined", "low_temp_ceiling") \
            and "_ceiling_value" in constraints:
        ceiling = constraints["_ceiling_value"]
        res["temp_ok"] = (res["max_T_reported"] <= ceiling) \
            if res["max_T_reported"] is not None else None

    if condition in ("inventory", "combined", "low_temp_inventory") \
            and "_inventory_list" in constraints:
        inv = {s.strip().lower() for s in constraints["_inventory_list"]}
        used = []
        for p in getattr(route, "precursors", []) or []:
            f = getattr(p, "formula", None)
            if f is None and isinstance(p, dict):
                f = p.get("formula") or p.get("material")
            if f:
                used.append(str(f).strip().lower())
        res["inventory_ok"] = all(u in inv for u in used) if used else None
        res["n_offlist"] = sum(1 for u in used if u not in inv) if used else None

    return res


def compute_temperature_economy(route, breakdown: dict, lit_max_T: float | None,
                                sentinel_tags: frozenset,
                                t_ref_margin: float = 150.0,
                                t_span: float = 400.0) -> float | None:
    """
    Continuous reward for lower T_max, gated on feasibility -- converts the
    confirmed temperature hack (validator grades ΔG at the model's OWN
    reported T, so reporting higher T is free favorability) into a scored
    objective instead of a validity gate. None (excluded from capacity, same
    None-propagation convention as every other check) when the route parsed
    to nothing, literature T is unavailable, or thermodynamic_favorable
    itself couldn't be computed for this completion.

    Per-target normalization is essential: T_ref = lit_max_T + margin, so
    this measures "how much colder than the literature route, capped at a
    plausible ceiling" rather than an absolute T -- a global scale would
    make low-T targets trivially easy and high-T targets impossible.

    Gated (multiplied, not summed) on thermodynamic_favorable >= 0.5
    (the validator's own borderline-or-better cutoff -- see
    RXN_ENERGY_BORDERLINE in validator.py) so the model can't trivially
    maximize by reporting room temperature; ungradeable thermo -> None,
    not a silent pass or fail.
    """
    if lit_max_T is None:
        return None
    return _temperature_economy_core(route, breakdown, sentinel_tags,
                                     lit_max_T + t_ref_margin, t_span)


def _temperature_economy_core(route, breakdown: dict, sentinel_tags: frozenset,
                              t_ref: float, t_span: float) -> float | None:
    if route is None or t_span <= 0:
        return None
    thermo = breakdown.get("thermodynamic_favorable")
    tag = breakdown.get("thermodynamic_favorable_gradeability")
    if not isinstance(thermo, (int, float)) or isinstance(thermo, bool):
        return None
    if tag in sentinel_tags:
        return None
    max_t = route_max_T(route)
    if max_t is None:
        return None
    economy = max(0.0, min(1.0, (t_ref - max_t) / t_span))
    return economy if thermo >= 0.5 else 0.0


def compute_temperature_economy_banded(route, breakdown: dict, band_lo: float | None,
                                       band_hi: float | None,
                                       sentinel_tags: frozenset) -> float | None:
    """
    Fix B (post_low_temp_probe_steps.md): for low_temp_ceiling, the economy
    gradient is rescaled to the band the model is actually confined to
    ([lit_max_T, ceiling]) rather than [lit_max_T, lit_max_T + margin] --
    round 1's fixed margin/span put a third of the mass off-scale (piled at
    0.0/1.0) because it didn't track the model's actual, ceiling-constrained
    output range. band_hi=ceiling gives 0.0 at the cap the hard constraint
    already enforces; band_lo=lit_max_T gives 1.0 at the literature route's
    own temperature -- the gradient does its work entirely within the band
    the ceiling produces the behavioral pull for.
    """
    if band_lo is None or band_hi is None or band_hi <= band_lo:
        return None
    return _temperature_economy_core(route, breakdown, sentinel_tags,
                                     t_ref=band_hi, t_span=band_hi - band_lo)


# --------------------------------------------------------------------------
# Capacity metrics (mirrors reward_geometry.py D1/D4/D5)
# --------------------------------------------------------------------------

def within_group_z(R: np.ndarray, gids: np.ndarray, eps: float = 1e-4):
    Z = np.full_like(R, np.nan)
    C = R.shape[1]
    zero_std, seen = np.zeros(C), np.zeros(C)
    for g in np.unique(gids):
        m = gids == g
        block = R[m]
        for c in range(C):
            col = block[:, c]
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                continue
            seen[c] += 1
            vals = col[valid]
            sd = vals.std(ddof=1)
            if sd < 1e-12:
                zero_std[c] += 1
                z = np.zeros_like(vals)
            else:
                z = (vals - vals.mean()) / (sd + eps)
            Z[np.where(m)[0][valid], c] = z
    frac_zero = {c: (zero_std[c] / seen[c] if seen[c] else float("nan"))
                 for c in range(C)}
    return Z, frac_zero


def capacity_metrics(records: list[dict], check_names: list[str]) -> dict:
    """Reward capacity + per-channel liveness for one condition."""
    rows, gids = [], []
    gindex = {}
    for r in records:
        bd = r.get("breakdown") or {}
        if bd.get("error"):
            continue
        key = r["target"]
        gindex.setdefault(key, len(gindex))
        rows.append([bd.get(n) if isinstance(bd.get(n), (int, float))
                     and not isinstance(bd.get(n), bool) else np.nan
                     for n in check_names])
        gids.append(gindex[key])
    if not rows:
        return {}
    R = np.asarray(rows, dtype=float)
    Z, frac_zero = within_group_z(R, np.asarray(gids))

    C = Z.shape[1]
    cov = np.zeros((C, C))
    for i in range(C):
        for j in range(i, C):
            both = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
            if both.sum() > 2:
                cov[i, j] = cov[j, i] = np.cov(Z[both, i], Z[both, j])[0, 1]
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    total = float(eig[eig > 0].sum())

    return {
        "total_variance": round(total, 4),
        "capacity_pct": round(100 * total / C, 2),
        "n_groups": len(gindex),
        "per_channel_zero_std_pct": {
            check_names[c]: round(100 * frac_zero[c], 1) for c in range(C)
        },
        "per_channel_z_var": {
            check_names[c]: round(float(np.nanvar(Z[:, c][~np.isnan(Z[:, c])]))
                                  if (~np.isnan(Z[:, c])).any() else float("nan"), 4)
            for c in range(C)
        },
    }


def canon(s) -> str:
    return str(s).strip().replace("(", "").replace(")", "").replace(" ", "").lower()


def route_set(route):
    if route is None:
        return None
    names = []
    for p in getattr(route, "precursors", []) or []:
        f = getattr(p, "formula", None)
        if f is None and isinstance(p, dict):
            f = p.get("formula") or p.get("material")
        if f:
            names.append(canon(f))
    return tuple(sorted(set(names))) if names else None


def diversity_metrics(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[r["target"]].append(r.get("precursor_set"))
    counts, identical = [], 0
    for sets in groups.values():
        parsed = [tuple(s) for s in sets if s]
        if len(parsed) < 2:
            continue
        d = len(set(parsed))
        counts.append(d)
        if d == 1:
            identical += 1
    if not counts:
        return {}
    return {
        "mean_distinct_routes": round(float(np.mean(counts)), 3),
        "pct_groups_all_identical": round(100 * identical / len(counts), 1),
        "n_groups": len(counts),
    }


def phat_metrics(records: list[dict], bars=(0.65, 0.9)) -> dict:
    groups = defaultdict(list)
    for r in records:
        if r.get("reward") is not None:
            groups[r["target"]].append(r["reward"])
    out = {}
    for bar in bars:
        ps = [sum(1 for x in v if x >= bar) / len(v) for v in groups.values() if v]
        if not ps:
            continue
        out[f"bar_{bar}"] = {
            "mean_phat": round(float(np.mean(ps)), 3),
            "n_zero": sum(1 for p in ps if p == 0),
            "n_mid": sum(1 for p in ps if 0 < p < 0.95),
            "n_saturated": sum(1 for p in ps if p >= 0.95),
        }
    return out


def adherence_metrics(records: list[dict]) -> dict:
    out = {}
    for key in ("temp_ok", "inventory_ok"):
        vals = [r["adherence"].get(key) for r in records
                if r.get("adherence") and r["adherence"].get(key) is not None]
        if vals:
            out[key] = round(100 * sum(vals) / len(vals), 1)
    offs = [r["adherence"].get("n_offlist") for r in records
            if r.get("adherence") and r["adherence"].get("n_offlist") is not None]
    if offs:
        out["mean_offlist_precursors"] = round(float(np.mean(offs)), 2)
    maxT = [r["adherence"].get("max_T_reported") for r in records
            if r.get("adherence") and r["adherence"].get("max_T_reported")]
    if maxT:
        out["mean_max_T_reported"] = round(float(np.mean(maxT)), 1)
    return out


# --------------------------------------------------------------------------

def analyze(path: Path, check_names: list[str]) -> None:
    blob = json.loads(path.read_text())
    by_cond = defaultdict(list)
    for r in blob["records"]:
        by_cond[r["condition"]].append(r)
    # iterate whatever conditions were actually run, not the original
    # 5-condition default -- otherwise low_temp/low_temp_combined (or any
    # future condition) silently vanish from the analysis
    run_conditions = blob.get("conditions") or list(by_cond)
    ordered_conditions = [c for c in ALL_CONDITIONS if c in run_conditions] + \
        [c for c in run_conditions if c not in ALL_CONDITIONS]

    print("\n" + "=" * 78)
    print("HARDENING PROBE — PRIMARY METRIC: REWARD CAPACITY")
    print("=" * 78)
    print("\nPre-registered prediction: capacity 14% -> 40%+, routes/group 2.01 -> 3+\n")

    print(f"{'condition':<18}{'capacity%':>11}{'routes/grp':>12}"
          f"{'%identical':>12}{'p̂@0.9':>9}{'n':>7}")
    summary = {}
    for cond in ordered_conditions:
        recs = by_cond.get(cond)
        if not recs:
            continue
        cap = capacity_metrics(recs, check_names)
        div = diversity_metrics(recs)
        ph = phat_metrics(recs)
        summary[cond] = (cap, div, ph, adherence_metrics(recs))
        print(f"{cond:<18}{cap.get('capacity_pct', float('nan')):>10.1f}%"
              f"{div.get('mean_distinct_routes', float('nan')):>12.2f}"
              f"{div.get('pct_groups_all_identical', float('nan')):>11.1f}%"
              f"{ph.get('bar_0.9', {}).get('mean_phat', float('nan')):>9.3f}"
              f"{len(recs):>7}")

    base_cap = summary.get("baseline", ({}, {}, {}, {}))[0].get("capacity_pct")
    print("\n--- per-channel zero-std %% (lower = more live) ---")
    conds = [c for c in ordered_conditions if c in summary]
    print(f"{'channel':<28}" + "".join(f"{c[:11]:>13}" for c in conds))
    for name in check_names:
        row = f"{name:<28}"
        for c in conds:
            v = summary[c][0].get("per_channel_zero_std_pct", {}).get(name)
            row += f"{v:>12.1f}%" if v is not None else f"{'-':>13}"
        print(row)

    print("\n--- constraint adherence (did the model obey?) ---")
    for c in conds:
        adh = summary[c][3]
        if adh:
            print(f"  {c:<16}{adh}")

    print("\n" + "=" * 78)
    best = max((summary[c][0].get("capacity_pct", 0), c) for c in conds)
    print(f"BEST CAPACITY: {best[1]} at {best[0]:.1f}%"
          + (f"   (baseline {base_cap:.1f}%)" if base_cap else ""))
    if best[0] > 40:
        print(">> PREDICTION CONFIRMED. This condition is run-4's task; GDPO finally")
        print("   gets a reward with enough within-group variance to exploit.")
    elif best[0] < 25:
        print(">> PREDICTION FAILED across all conditions. The diagnosis that dead")
        print("   channels are caused by score-irrelevant variation is wrong, or the")
        print("   constraints did not bind. Check adherence above: if the model")
        print("   ignored the constraints, that is a prompt problem, not a null.")
        print("   If adherence is high and capacity still flat, the honest")
        print("   conclusion is that this task does not support factored-reward RL.")
    else:
        print(">> PARTIAL. Capacity moved but did not clear the bar. Consider")
        print("   tightening the binding constraint (smaller inventory, lower")
        print("   ceiling) and re-probing before committing GPU to a full run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="base model (tokenizer fallback + LoRA base). NOT a "
                         "checkpoint path — --checkpoint carries the adapter.")
    ap.add_argument("--triage", type=Path,
                    default=Path("misc/kononova_triage_results3.json"))
    ap.add_argument("--synthesis", type=Path,
                    default=Path("data/raw/synthesis_clean.json"))
    ap.add_argument("--val", type=Path, default=Path("data/rl/val.jsonl"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("runs/sft-qlora-sft-v3-2nd-rank16/final"))
    ap.add_argument("--out", type=Path, default=Path("misc/hardening_probe.json"))
    ap.add_argument("--reuse-pool-from", type=Path, default=None,
                    help="load the exact (target, stratum) pool from a prior "
                         "hardening run's baseline records instead of "
                         "re-sampling val.jsonl, for direct comparability "
                         "(spec: 'same target pool as the previous hardening "
                         "probe'). E.g. misc/hardening.json.")
    ap.add_argument("--n-targets", type=int, default=40)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--inventory-size", type=int, default=16)
    ap.add_argument("--t-ref-margin", type=float, default=150.0,
                    help="low_temp/low_temp_inventory: T_ref = lit_max_T + this "
                         "(round-2 default 150, tightened from round 1's 300 -- "
                         "Fix A in post_low_temp_probe_steps.md). Ignored by "
                         "low_temp_ceiling, which bands to [lit_T, ceiling] instead.")
    ap.add_argument("--t-span", type=float, default=400.0,
                    help="low_temp/low_temp_inventory: economy scale width "
                         "(round-2 default 400, tightened from round 1's 600).")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=ALL_CONDITIONS)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scorer", choices=["validator", "ranker"], default="validator",
                    help="validator = Arm A (validity checks, RANKER_SPEC.md's "
                         "control). ranker = Arm B (core/ranker.py's gates x "
                         "objectives quality scorer) -- the verifier is the "
                         "independent variable, everything else about this "
                         "probe stays identical between the two.")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--dump-lit", action="store_true")
    args = ap.parse_args()

    lit = load_literature(args.triage, args.synthesis)
    if args.dump_lit:
        for k, v in list(lit.items())[:3]:
            print(k, json.dumps(v, indent=2))
        return

    ranker = None
    if args.scorer == "ranker":
        from core.ranker import Ranker, OBJECTIVE_NAMES, build_precursor_frequency
        import pickle
        with open("data/cache/mp_formula_set.pkl", "rb") as f:
            formula_set = pickle.load(f)
        thermo = ThermoChecker.from_sharded_cache(
            Path("data/cache/pd_index.json"), Path("."))
        freq = build_precursor_frequency(args.synthesis)
        ranker = Ranker(formula_set, thermo, freq)
        check_names = list(OBJECTIVE_NAMES)
        validator = None
    else:
        validator = load_validator(Path("data/cache/mp_formula_set.pkl"),
                                   Path("data/cache/pd_index.json"), Path("."))
        # The full 10-channel vector — the pre-registered 14.3% capacity
        # baseline (reward_geometry D1) was computed over all ten; a subset
        # would make the comparison invalid. validator.weights carries 9;
        # target_match is added by validate() itself. temperature_economy is
        # an 11th, LOCAL-ONLY channel (never touches validator.py) computed
        # post-hoc for low_temp* conditions; it's simply absent/NaN for the
        # original five, so always including it here doesn't change their
        # capacity numbers.
        check_names = sorted(set(validator.weights) | {"target_match", "temperature_economy"})

    if args.analyze_only:
        analyze(args.out, check_names)
        return

    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.reuse_pool_from and args.reuse_pool_from.exists():
        prior = json.loads(args.reuse_pool_from.read_text())
        seen = {}
        for r in prior.get("records", []):
            seen.setdefault(r["target"], r.get("stratum", "unknown"))
        targets = [(t, s) for t, s in seen.items() if t in lit and lit[t]["precursors"]]
        n_missing_lit = len(seen) - len(targets)
        print(f"reusing {len(targets)}/{len(seen)} targets from {args.reuse_pool_from} "
              f"({n_missing_lit} dropped: no longer have usable literature data)")
    else:
        # Target pool: prefer targets with literature data (needed for constraints).
        pool = []
        for line in open(args.val):
            r = json.loads(line)
            t = r["target"]
            if t in lit and lit[t]["precursors"]:
                pool.append((t, r.get("stratum", "unknown")))
        rng.shuffle(pool)
        targets = pool[: args.n_targets]
    n_missing_maxT = sum(1 for t, _ in targets if lit[t].get("max_T") is None)
    if n_missing_maxT:
        print(f"WARNING: {n_missing_maxT}/{len(targets)} targets have no literature "
              f"max_T -- temperature_economy will be None (excluded from capacity) "
              f"for those under low_temp*", file=sys.stderr)
    if not targets:
        raise RuntimeError(
            f"No targets with usable literature constraints found: {len(pool)} "
            f"candidates in pool, {len(lit)} targets loaded from literature "
            f"({args.triage} x {args.synthesis}), val set {args.val}. "
            "Check that the target formulas in --val actually appear in "
            "--triage/--synthesis, and that load_literature() is parsing "
            "precursors/temperatures correctly (--dump-lit)."
        )
    print(f"{len(targets)} targets with literature constraints available")

    decoys = build_decoy_pool(lit)
    model, tok = load_eval_model(checkpoint=args.checkpoint, model_name=args.model)
    

    # Matches probe_passk.py's closed_prompt() user content, so this probe is
    # prompt-identical to the pass@k baselines (SYSTEM_MSG is prepended below,
    # same as generate_batch's other callers -- see stratified_difficulty_eval.py).
    def base_prompt_fn(target: str) -> str:
        return f"Target: {target}\n\nProvide your synthesis route as a JSON object."

    records = []
    done_pairs = set()
    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text())
            records = prev.get("records", [])
            done_pairs = {(r["condition"], r["target"]) for r in records}
            print(f"resuming: {len(done_pairs)} (condition, target) cells done")
        except Exception as e:
            print(f"could not resume from {args.out}: {e}; starting fresh")

    t0 = time.time()
    for cond in args.conditions:
        for i, (target, stratum) in enumerate(targets):
            if (cond, target) in done_pairs:
                continue
            # per-cell seed: decoy sampling is identical whether or not a
            # resume skipped earlier cells
            cell_rng = random.Random(f"{args.seed}:{cond}:{target}")
            cons = make_constraints(target, lit[target], decoys, cell_rng,
                                    args.inventory_size)
            prompt = build_prompt(target, cond, cons, base_prompt_fn)
            prompt_full = SYSTEM_MSG + "\n\n" + prompt

            completions = []
            while len(completions) < args.samples:
                n = min(args.batch_size, args.samples - len(completions))
                completions += generate_batch(model, tok, [prompt_full] * n, args)

            for comp in completions:
                # parse_completion RAISES ParseFailure (it never returns
                # None) and needs the target for the route schema.
                try:
                    route = parse_completion(comp, target)
                except Exception:
                    route = None

                if ranker is not None:
                    # Arm B: the ranker computes its own temperature_economy
                    # (gates x objectives) uniformly for every condition --
                    # no low_temp*-specific post-hoc injection needed here.
                    reward, breakdown = ranker.score(
                        route, target, lit_T=lit[target].get("max_T"),
                        lit_n_ops=lit[target].get("n_ops"))
                else:
                    try:
                        if route is None:
                            raise ValueError("unparseable")
                        reward, breakdown = validator.validate(route, target)
                    except Exception:
                        reward, breakdown = (None, {"error": "parse_failure"})
                    if cond in LOW_TEMP_CONDITIONS and isinstance(breakdown, dict):
                        breakdown = dict(breakdown)
                        if cond == "low_temp_ceiling":
                            breakdown["temperature_economy"] = compute_temperature_economy_banded(
                                route, breakdown, lit[target].get("max_T"),
                                cons.get("_ceiling_value"), validator.SENTINEL_TAGS)
                        else:
                            breakdown["temperature_economy"] = compute_temperature_economy(
                                route, breakdown, lit[target].get("max_T"),
                                validator.SENTINEL_TAGS, args.t_ref_margin, args.t_span)
                records.append({
                    "condition": cond, "target": target, "stratum": stratum,
                    "reward": reward, "breakdown": breakdown,
                    "precursor_set": list(route_set(route) or []),
                    "adherence": check_adherence(route, cons, cond),
                })

            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(
                {"records": records, "conditions": args.conditions,
                 "n_targets": len(targets), "samples": args.samples}))
            print(f"  [{cond}] {i+1}/{len(targets)} {target:<28} "
                  f"({(time.time()-t0)/60:.0f} min)", file=sys.stderr, flush=True)

    del model
    torch.cuda.empty_cache()
    analyze(args.out, check_names)


if __name__ == "__main__":
    main()