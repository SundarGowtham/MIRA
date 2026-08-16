#!/usr/bin/env python
"""
reward_geometry.py — offline diagnostic battery for MIRA GDPO runs.

Runs against runs/*/generations.jsonl. CPU only, no model loading.

  D1  reward-vector geometry      -- how much signal, and what shape
  D2  answer-level diversity      -- distinct routes per group, over time
  D3  target fixed-effects trend  -- real trends, or an inestimable design
  D4  dead-channel / NaN audit    -- which channels are dead, and where
  D5  within-group route identity -- direct test for diversity collapse

Usage:
    python reward_geometry.py --gens runs/<run>/generations.jsonl --strata data/rl --all
    python reward_geometry.py --gens ... --schema           # inspect record shape
    python reward_geometry.py --gens ... --d1 --eps 1e-8    # epsilon sensitivity
    python reward_geometry.py --gens ... --d2 --debug-parse
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_generations(path: Path) -> list[dict]:
    recs = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  warn: skipping malformed line {i}")
    print(f"loaded {len(recs)} records from {path}")
    return recs


def discover_check_names(recs: list[dict]) -> list[str]:
    names = set()
    for r in recs[:5000]:
        bd = r.get("breakdown") or {}
        for k, v in bd.items():
            if k.endswith("_gradeability"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                names.add(k)
    return sorted(names)


# --------------------------------------------------------------------------
# Route parsing
# --------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_PRECURSOR_KEYS = ("precursors", "precursor", "reactants", "reagents",
                   "inputs", "starting_materials")
_FORMULA_KEYS = ("formula", "material", "name", "compound", "precursor",
                 "chemical_formula", "reagent", "species")


def _balanced_json_objects(text: str) -> list[str]:
    """Top-level {...} substrings via brace matching, ignoring braces inside strings."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(text[start:i + 1])
                    start = None
    return out


def extract_route_json(completion: str) -> dict | None:
    """
    Pull the route object out of a completion. Handles <think>...</think>
    preamble, ``` fences, trailing prose, and multiple JSON objects (prefers
    the largest one containing a precursor list, so an example embedded in
    the reasoning does not win over the real answer).
    """
    if not completion:
        return None

    body = _THINK_RE.sub("", completion)

    candidates: list[str] = [m.group(1) for m in _FENCE_RE.finditer(body)]
    candidates += _balanced_json_objects(body)
    candidates.append(body)
    candidates += _balanced_json_objects(completion)

    best = None
    for c in candidates:
        c = c.strip()
        if not c.startswith("{"):
            objs = _balanced_json_objects(c)
            if not objs:
                continue
            c = max(objs, key=len)
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in _PRECURSOR_KEYS):
            if best is None or len(c) > best[0]:
                best = (len(c), obj)
    return best[1] if best else None


def canonical_formula(s: str) -> str:
    """
    Light normalization so trivially different spellings collapse together.
    Deliberately does NOT parse chemistry -- it only removes formatting noise,
    so distinct-route counts are a mild upper bound on true diversity.
    """
    s = str(s).strip()
    s = s.replace("\u00b7", ".").replace("·", ".").replace("*", ".")
    s = re.sub(r"\s+", "", s)
    s = s.replace("(", "").replace(")", "")
    return s.lower()


def _precursor_list(obj: dict):
    for k in _PRECURSOR_KEYS:
        if isinstance(obj.get(k), list):
            return obj[k]
    return None


def _entry_formula(p):
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for k in _FORMULA_KEYS:
            if p.get(k):
                return p[k]
        vals = [v for v in p.values() if isinstance(v, str)]
        return vals[0] if vals else None
    return None


def extract_precursor_set(completion: str) -> tuple | None:
    """Canonical, order-independent precursor set, or None if unparseable."""
    obj = extract_route_json(completion)
    if obj is None:
        return None
    plist = _precursor_list(obj)
    if plist is None:
        return None
    names = [canonical_formula(f) for f in map(_entry_formula, plist) if f]
    return tuple(sorted(set(names))) if names else None


def extract_amounts(completion: str) -> tuple | None:
    """Rounded (formula, amount) pairs -- separates 'same precursors, different
    stoichiometry' from genuinely distinct routes."""
    obj = extract_route_json(completion)
    if obj is None:
        return None
    plist = _precursor_list(obj)
    if plist is None:
        return None

    pairs = []
    for p in plist:
        if not isinstance(p, dict):
            continue
        f = _entry_formula(p)
        amt = p.get("amount", p.get("moles", p.get("quantity", 1.0)))
        try:
            amt = round(float(amt), 3)
        except (TypeError, ValueError):
            amt = None
        if f:
            pairs.append((canonical_formula(f), amt))
    return tuple(sorted(pairs)) if pairs else None


def probe_schema(recs: list[dict], n: int = 2) -> None:
    print("\n=== SCHEMA ===")
    print("top-level keys:", sorted(recs[0].keys()))
    print("breakdown keys:", sorted((recs[0].get("breakdown") or {}).keys()))

    for r in recs[:n]:
        comp = r.get("completion") or ""
        print(f"\n--- completion tail ({len(comp)} chars) ---")
        print(comp[-1500:])

    for r in recs:
        parsed = extract_route_json(r.get("completion") or "")
        if parsed:
            print("\n--- first successfully parsed route object ---")
            print("top-level keys:", sorted(parsed.keys()))
            print(json.dumps(parsed, indent=2)[:1500])
            return
    print("\n!! no route object parsed from any record -- inspect the tails above "
          "and extend _PRECURSOR_KEYS / _FORMULA_KEYS accordingly")


# --------------------------------------------------------------------------
# Matrix construction
# --------------------------------------------------------------------------

def build_reward_matrix(recs: list[dict], check_names: list[str]):
    rows, gids, steps, targets = [], [], [], []
    group_index: dict[tuple, int] = {}

    for r in recs:
        bd = r.get("breakdown") or {}
        if bd.get("error"):
            continue
        step, target = r.get("step"), r.get("target")
        if step is None or target is None:
            continue

        key = (step, target)
        group_index.setdefault(key, len(group_index))

        row = []
        for name in check_names:
            v = bd.get(name)
            row.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                       else np.nan)
        rows.append(row)
        gids.append(group_index[key])
        steps.append(step)
        targets.append(target)

    R = np.asarray(rows, dtype=float)
    print(f"built reward matrix: {R.shape[0]} completions x {R.shape[1]} checks, "
          f"{len(group_index)} groups")
    return R, np.asarray(gids), np.asarray(steps), np.asarray(targets, dtype=object)


def within_group_z(R: np.ndarray, gids: np.ndarray, eps: float = 1e-4):
    """
    Z-normalize each channel within each group, as GDPO's normalize_then_sum does.
    NaN in == NaN out. Zero-variance channels give z=0 rather than a blown-up value;
    --eps lets you measure how much that choice matters.
    """
    Z = np.full_like(R, np.nan)
    C = R.shape[1]
    zero_std, seen = np.zeros(C), np.zeros(C)

    for g in np.unique(gids):
        mask = gids == g
        block = R[mask]
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
            Z[np.where(mask)[0][valid], c] = z

    frac_zero = {c: (zero_std[c] / seen[c] if seen[c] else float("nan")) for c in range(C)}
    return Z, frac_zero


# --------------------------------------------------------------------------
# D1
# --------------------------------------------------------------------------

def effective_rank_entropy(eigvals: np.ndarray) -> float:
    lam = eigvals[eigvals > 1e-12]
    if lam.size == 0:
        return 0.0
    p = lam / lam.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio(eigvals: np.ndarray) -> float:
    lam = eigvals[eigvals > 1e-12]
    if lam.size == 0:
        return 0.0
    return float(lam.sum() ** 2 / np.square(lam).sum())


def d1_reward_geometry(Z: np.ndarray, check_names: list[str]) -> dict:
    """
    Covariance eigenspectrum of the within-group z-normalized reward.

    TWO different questions, and reporting only the second is misleading:
      TOTAL VARIANCE -- how much signal exists at all. A fully informative
                        C-channel reward totals ~C (unit variance per channel).
      EFFECTIVE RANK -- the shape of whatever signal is present. A reward with
                        almost no signal can still spread it over several
                        directions and score a healthy-looking rank.
    """
    print("\n" + "=" * 70)
    print("D1 — REWARD VECTOR GEOMETRY")
    print("=" * 70)

    C = Z.shape[1]
    cov = np.full((C, C), np.nan)
    for i in range(C):
        for j in range(i, C):
            both = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
            if both.sum() > 2:
                cov[i, j] = cov[j, i] = np.cov(Z[both, i], Z[both, j])[0, 1]
    if np.isnan(cov).any():
        print("  warn: some channel pairs never co-occur; filling NaN with 0")
        cov = np.nan_to_num(cov)

    cov = (cov + cov.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    total = float(eigvals[eigvals > 0].sum())
    capacity = 100 * total / C

    print(f"\n{'idx':<5}{'eigenvalue':>14}{'% var':>10}{'cumulative':>13}")
    cum = 0.0
    for i, lam in enumerate(eigvals):
        if lam <= 1e-12:
            continue
        pct = 100 * lam / total
        cum += pct
        print(f"{i:<5}{lam:>14.5f}{pct:>9.1f}%{cum:>12.1f}%")

    er = effective_rank_entropy(eigvals)
    pr = participation_ratio(eigvals)

    print(f"\nTOTAL VARIANCE:  {total:.3f}   (max possible {C}.0)")
    print(f"CAPACITY USED:   {capacity:.1f}%   <-- how much of the reward is live")
    print(f"effective rank:  {er:.3f}   (shape of the surviving signal only)")
    print(f"participation:   {pr:.3f}")

    print("\nper-channel variance of z  (0 == never varies within a group):")
    for c, name in enumerate(check_names):
        col = Z[:, c]
        valid = col[~np.isnan(col)]
        v = float(np.var(valid)) if valid.size else float("nan")
        print(f"  {name:<30}{v:>7.4f}  {'#' * int(min(v, 1.0) * 40)}")

    print("\ntop eigenvector loadings:")
    for name, w in sorted(zip(check_names, eigvecs[:, 0]), key=lambda x: -abs(x[1])):
        print(f"  {name:<30}{w:+.3f}")

    print()
    if capacity < 25:
        print(">> MOST OF THE REWARD IS DEAD. The effective rank above describes the")
        print("   shape of a small residual, not a healthy multi-channel signal. With")
        print("   this much collapse you are effectively running single-channel GRPO,")
        print("   and a GRPO-vs-GDPO comparison cannot show what it is meant to show.")
        print("   Fix the dead channels (see D4/D5) before running the grid.")
    elif capacity < 50:
        print(">> About half the reward capacity is live. Worth fixing dead channels")
        print("   first, but the factorization is not vacuous.")
    else:
        print(">> Reward is broadly live. Factored-vs-scalar comparison is well posed.")

    return {"eigvals": eigvals, "eigvecs": eigvecs, "eff_rank": er,
            "total_variance": total, "capacity_pct": capacity, "cov": cov}


# --------------------------------------------------------------------------
# D2
# --------------------------------------------------------------------------

def d2_answer_diversity(recs: list[dict], n_bins: int = 12, debug: bool = False) -> None:
    """
    Distinct precursor sets per group, binned by step.

    Token entropy and answer-level diversity can move in OPPOSITE directions.
    If entropy rose while this falls, the entropy curve was falsely reassuring.
    """
    print("\n" + "=" * 70)
    print("D2 — ANSWER-LEVEL DIVERSITY")
    print("=" * 70)

    groups = defaultdict(list)
    n_fail = n_tot = 0
    for r in recs:
        step, target, comp = r.get("step"), r.get("target"), r.get("completion")
        if step is None or target is None or not comp:
            continue
        n_tot += 1
        s = extract_precursor_set(comp)
        if s is None:
            n_fail += 1
        groups[(step, target)].append(s)

    print(f"parsed {n_tot - n_fail}/{n_tot} completions "
          f"({100 * (n_tot - n_fail) / max(n_tot, 1):.1f}%)")

    if debug and n_fail:
        for r in recs:
            comp = r.get("completion") or ""
            if comp and extract_precursor_set(comp) is None:
                print("\n--- sample parse failure (tail) ---")
                print(comp[-1200:])
                break

    rows = []
    for (step, _t), sets in groups.items():
        parsed = [s for s in sets if s is not None]
        if len(parsed) < 2:
            continue
        rows.append((step, len(set(parsed)), len(parsed)))

    if not rows:
        print("  no parseable groups -- run with --schema to inspect the route format")
        return

    rows.sort()
    steps = np.array([r[0] for r in rows], dtype=float)
    distinct = np.array([r[1] for r in rows], dtype=float)
    sizes = np.array([r[2] for r in rows], dtype=float)
    ratio = distinct / sizes

    edges = np.linspace(steps.min(), steps.max() + 1, n_bins + 1)
    print(f"\n{'step range':<18}{'mean distinct':>15}{'mean G':>9}{'ratio':>9}{'n':>7}")
    for i in range(n_bins):
        m = (steps >= edges[i]) & (steps < edges[i + 1])
        if m.sum() == 0:
            continue
        print(f"{int(edges[i]):>6}-{int(edges[i+1]):<11}{distinct[m].mean():>15.2f}"
              f"{sizes[m].mean():>9.2f}{ratio[m].mean():>9.3f}{m.sum():>7}")

    slope = np.polyfit(steps, ratio, 1)[0]
    print(f"\nmean distinct-per-group: {distinct.mean():.2f} of {sizes.mean():.2f}")
    print(f"linear slope of distinct/G vs step: {slope:+.3e}")
    if distinct.mean() < 1.5:
        print(">> SEVERE diversity collapse: groups are near-identical, so group-relative")
        print("   advantage has almost nothing to compare.")
    elif slope < 0:
        print(">> Answer-level diversity FALLING. With rising token entropy this is the")
        print("   Invisible Leash regime: local wandering, global collapse.")
    else:
        print(">> Answer-level diversity stable or rising.")


# --------------------------------------------------------------------------
# D3
# --------------------------------------------------------------------------

def d3_fixed_effects_trend(R: np.ndarray, steps: np.ndarray, targets: np.ndarray,
                           check_names: list[str]) -> None:
    """
    Residualize against each target's own mean, then regress on step.

    If each target appears at only one step, target fixed effects absorb step
    entirely and the within-target trend is INESTIMABLE -- not zero. Detected
    and reported rather than silently emitting ~1e-20 slopes.
    """
    print("\n" + "=" * 70)
    print("D3 — TARGET FIXED-EFFECTS TREND")
    print("=" * 70)

    steps_per_target = defaultdict(set)
    for s, t in zip(steps, targets):
        steps_per_target[t].add(int(s))
    multi = sum(1 for v in steps_per_target.values() if len(v) > 1)
    frac_multi = multi / max(len(steps_per_target), 1)
    print(f"\ntargets seen at >1 step: {multi}/{len(steps_per_target)} "
          f"({100 * frac_multi:.1f}%)")

    if frac_multi < 0.05:
        print("\n>> DESIGN IS RANK-DEFICIENT: essentially every target appears at exactly")
        print("   one step, so target fixed effects absorb the step variable completely.")
        print("   The within-target trend is INESTIMABLE from this data; any FE slope")
        print("   would be floating-point residue, not an effect size.")
        print("\n   Fix for the next run: evaluate a FIXED PROBE SET of ~30 held-out")
        print("   targets every N steps. Repeated measurements on the same targets are")
        print("   what make a trend identifiable.")
        print("\n   Raw slopes below still confound trend with target composition:")
        print(f"\n{'check':<30}{'raw slope':>14}{'raw SE':>12}{'t':>8}")
        for c, name in enumerate(check_names):
            col = R[:, c]
            valid = ~np.isnan(col)
            if valid.sum() < 50:
                continue
            y, s = col[valid], steps[valid].astype(float)
            s_c = s - s.mean()
            denom = float(np.sum(s_c ** 2))
            if denom < 1e-12:
                continue
            slope = float(np.sum(s_c * (y - y.mean())) / denom)
            resid = (y - y.mean()) - slope * s_c
            sigma2 = float(np.sum(resid ** 2) / max(len(y) - 2, 1))
            se = math.sqrt(sigma2 / denom)
            t_stat = slope / se if se > 0 else 0.0
            print(f"{name:<30}{slope:>14.2e}{se:>12.2e}{t_stat:>8.2f}")
        return

    # Proper within-transformation: demean BOTH y and step by target.
    print(f"\n{'check':<30}{'raw slope':>13}{'FE slope':>13}{'FE SE':>11}{'t':>8}")
    for c, name in enumerate(check_names):
        col = R[:, c]
        valid = ~np.isnan(col)
        if valid.sum() < 50:
            print(f"{name:<30}{'insufficient data':>45}")
            continue
        y, s, tg = col[valid], steps[valid].astype(float), targets[valid]
        raw_slope = float(np.polyfit(s, y, 1)[0])

        y_sum, s_sum, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
        for yy, ss, tt in zip(y, s, tg):
            y_sum[tt] += yy
            s_sum[tt] += ss
            cnt[tt] += 1
        y_res = np.array([yy - y_sum[tt] / cnt[tt] for yy, tt in zip(y, tg)])
        s_res = np.array([ss - s_sum[tt] / cnt[tt] for ss, tt in zip(s, tg)])

        denom = float(np.sum(s_res ** 2))
        if denom < 1e-9:
            print(f"{name:<30}{raw_slope:>13.2e}{'inestimable':>13}")
            continue
        fe_slope = float(np.sum(s_res * y_res) / denom)
        resid = y_res - fe_slope * s_res
        dof = max(len(y_res) - len(cnt) - 1, 1)
        sigma2 = float(np.sum(resid ** 2) / dof)
        se = math.sqrt(sigma2 / denom)
        t_stat = fe_slope / se if se > 0 else 0.0
        flag = "  <-- significant" if abs(t_stat) > 2 else ""
        print(f"{name:<30}{raw_slope:>13.2e}{fe_slope:>13.2e}{se:>11.2e}{t_stat:>8.2f}{flag}")

    print("\n|t| > 2 is the bar for a real trend.")


# --------------------------------------------------------------------------
# D4
# --------------------------------------------------------------------------

def d4_dead_channel_audit(R: np.ndarray, gids: np.ndarray, check_names: list[str],
                          frac_zero: dict, strata: dict | None = None,
                          targets: np.ndarray | None = None) -> None:
    print("\n" + "=" * 70)
    print("D4 — DEAD CHANNEL / NaN AUDIT")
    print("=" * 70)
    print(f"\n{'check':<30}{'NaN %':>9}{'mean':>9}{'std':>9}{'zero-std grp %':>17}")

    live = []
    for c, name in enumerate(check_names):
        col = R[:, c]
        nan_pct = 100 * float(np.isnan(col).mean())
        valid = col[~np.isnan(col)]
        mean = float(valid.mean()) if valid.size else float("nan")
        std = float(valid.std()) if valid.size else float("nan")
        fz = 100 * frac_zero.get(c, float("nan"))
        note = ""
        if fz > 95:
            note = "  DEAD"
        elif fz > 70:
            note = "  mostly dead"
        elif std < 0.02:
            note = "  saturated"
        else:
            live.append(name)
        print(f"{name:<30}{nan_pct:>8.1f}%{mean:>9.3f}{std:>9.3f}{fz:>16.1f}%{note}")

    print(f"\nlive channels: {live if live else 'NONE'}")
    print("A channel with high pooled std but ~100% zero-std groups varies ACROSS")
    print("targets but never WITHIN a group -- every completion in the group produced")
    print("the same answer for it. For route-derived checks that is answer-level")
    print("diversity collapse (see D5), not convergence. For prompt-derived checks")
    print("(e.g. target_stability = property of the target, not the route) it is")
    print("zero-gradient BY CONSTRUCTION and the channel should be removed.")

    if strata is None or targets is None:
        print("\n(per-stratum breakdown skipped: pass --strata data/rl to enable)")
        return

    labels = np.array([strata.get(t, "unknown") for t in targets], dtype=object)
    uniq = sorted(set(labels), key=str)

    print("\n--- NaN rate by stratum (difficulty-correlated masking check) ---")
    print(f"{'stratum':<26}" + "".join(f"{n[:11]:>13}" for n in check_names))
    for s in uniq:
        m = labels == s
        if m.sum() == 0:
            continue
        row = f"{str(s)[:25]:<26}"
        for c in range(len(check_names)):
            row += f"{100 * float(np.isnan(R[m, c]).mean()):>12.1f}%"
        print(row)

    print("\n--- active channel count by stratum ---")
    for s in uniq:
        m = labels == s
        if m.sum() == 0:
            continue
        active = float((~np.isnan(R[m])).sum(axis=1).mean())
        print(f"  {str(s)[:30]:<32}{active:>6.2f} of {len(check_names)} channels, n={m.sum()}")
    print("\nIf the hardest stratum has the fewest active channels, the reward is LEAST")
    print("informative exactly where the task is hardest, and cross-stratum reward")
    print("comparisons are not measuring the same quantity.")


# --------------------------------------------------------------------------
# D5
# --------------------------------------------------------------------------

def d5_route_identity(recs: list[dict]) -> None:
    """
    Direct test for diversity collapse: within each group, how often is every
    completion the same route?

    Separates two very different failure modes:
      - same precursors, differing amounts -> only amount_accuracy can generate
        advantage; every other route-derived check is constant within group.
      - same precursors AND amounts -> the group carries no signal at all.
    """
    print("\n" + "=" * 70)
    print("D5 — WITHIN-GROUP ROUTE IDENTITY")
    print("=" * 70)

    groups = defaultdict(list)
    for r in recs:
        step, target, comp = r.get("step"), r.get("target"), r.get("completion")
        if step is None or target is None or not comp:
            continue
        groups[(step, target)].append((extract_precursor_set(comp),
                                       extract_amounts(comp)))

    n_groups = same_prec = same_full = 0
    prec_counts, amt_counts = [], []
    for items in groups.values():
        pset = [p for p, _ in items if p is not None]
        aset = [a for _, a in items if a is not None]
        if len(pset) < 2:
            continue
        n_groups += 1
        npd, nad = len(set(pset)), (len(set(aset)) if aset else 0)
        prec_counts.append(npd)
        amt_counts.append(nad)
        if npd == 1:
            same_prec += 1
            if nad <= 1:
                same_full += 1

    if n_groups == 0:
        print("  no parseable groups -- run --schema first")
        return

    print(f"\ngroups analysed: {n_groups}")
    print(f"mean distinct precursor sets per group:      {np.mean(prec_counts):.2f}")
    print(f"mean distinct (precursor, amount) per group: {np.mean(amt_counts):.2f}")
    print(f"all completions share precursors:            {same_prec} "
          f"({100 * same_prec / n_groups:.1f}%)")
    print(f"identical in precursors AND amounts:         {same_full} "
          f"({100 * same_full / n_groups:.1f}%)")

    print("\ndistinct precursor sets per group:")
    for k, v in sorted(Counter(prec_counts).items()):
        print(f"  {k:>2} distinct: {v:>5} groups  {'#' * int(40 * v / n_groups)}")

    if same_prec / n_groups > 0.7:
        print("\n>> The model emits the SAME precursor set across the group most of the")
        print("   time. Every route-derived check except amount_accuracy is therefore")
        print("   constant within group and contributes zero advantage. This is the")
        print("   MECHANISM behind the dead channels in D4: a sampling-diversity")
        print("   problem, not a reward-design problem. Raising G will not fix it.")
        print("   Levers: higher sampling temperature, DAPO clip-higher, entropy bonus,")
        print("   diversity-aware or stratified sampling within the group.")


def load_strata(rl_dir: Path) -> dict:
    strata = {}
    for split in ("train", "val"):
        p = rl_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    strata[r["target"]] = r.get("stratum")
                except Exception:
                    continue
    print(f"loaded strata for {len(strata)} targets")
    return strata


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=Path, required=True)
    ap.add_argument("--gens2", type=Path, default=None)
    ap.add_argument("--strata", type=Path, default=None)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--debug-parse", action="store_true")
    ap.add_argument("--all", action="store_true")
    for d in ("d1", "d2", "d3", "d4", "d5"):
        ap.add_argument(f"--{d}", action="store_true")
    args = ap.parse_args()

    recs = load_generations(args.gens)
    if args.gens2:
        recs += load_generations(args.gens2)

    if args.schema:
        probe_schema(recs)
        return

    check_names = discover_check_names(recs)
    print(f"discovered {len(check_names)} checks: {check_names}")

    R, gids, steps, targets = build_reward_matrix(recs, check_names)
    Z, frac_zero = within_group_z(R, gids, eps=args.eps)

    run_all = args.all or not any([args.d1, args.d2, args.d3, args.d4, args.d5])
    if run_all or args.d1:
        d1_reward_geometry(Z, check_names)
    if run_all or args.d2:
        d2_answer_diversity(recs, debug=args.debug_parse)
    if run_all or args.d3:
        d3_fixed_effects_trend(R, steps, targets, check_names)
    if run_all or args.d4:
        d4_dead_channel_audit(R, gids, check_names, frac_zero,
                              load_strata(args.strata) if args.strata else None,
                              targets)
    if run_all or args.d5:
        d5_route_identity(recs)


if __name__ == "__main__":
    main()