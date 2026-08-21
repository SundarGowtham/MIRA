#!/usr/bin/env python
"""
build_validity_read.py — the validator-validity check the project has never run.

Claude's item 1 (CLAUDE_RESPONSE_TO_RUN3_JUSTIFICATION.md): take 20 completions
the validator scored >= --bar, put them side by side with the Kononova
literature route for the same target, and let a human judge whether
validator-0.9+ means "a route a chemist would run" or "formally-valid
nonsense that satisfies the checks". Every other diagnostic measures the
apparatus; this is the only one that measures the proxy against reality.

Source of completions: run-2 generation dump (on-policy, most recent RL
policy with full breakdowns). Scalar rewards are reconstructed from the
dumped breakdowns with validate()'s own rule (sentinel exclusion +
WEIGHTS_THERMO renormalization) via the run-3 dataset builder.

For each sampled target the digest shows: model route (precursors + amounts,
operations with T/time/atmosphere), the validator's per-check scores, the
literature route(s) (reaction string, precursors, operations, DOI, paragraph
snippet), and mechanical first-pass flags (precursor-set overlap, amount
ratios vs the reaction string, max-temperature delta, atmosphere match) so
the human reader's attention goes to the disagreements.

Usage:
  PYTHONPATH=. uv run python run_debug_and_analysis/build_validity_read.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_curation"))

from pymatgen.core import Composition  # noqa: E402

from build_rl_run3_dataset import scalar_reward  # noqa: E402  (validate()'s rule)
from reward_geometry import extract_route_json  # noqa: E402
from validator import SynthesisValidator  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gens", type=Path,
                   default=Path("runs/gdpo-qlora-beta-ablation-probe/generations.jsonl"))
    p.add_argument("--literature", type=Path, default=Path("data/raw/synthesis_clean.json"))
    p.add_argument("--rl-dir", type=Path, default=Path("data/rl"))
    p.add_argument("--out", type=Path, default=Path("misc/validity_read_20.md"))
    p.add_argument("--bar", type=float, default=0.9)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def canon(formula: str) -> str:
    """Reduced formula for matching; None if unparsable."""
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def lit_index(recs: list[dict]) -> dict[str, list[dict]]:
    idx = defaultdict(list)
    for r in recs:
        idx[r.get("target_formula")].append(r)
    return idx


def find_lit(idx, target: str) -> list[dict]:
    hits = idx.get(target)
    if hits:
        return hits
    c = canon(target)
    if c:
        for t, rs in idx.items():
            if canon(t) == c:
                return rs
    return []


def parse_reaction_amounts(reaction_string: str) -> dict[str, float]:
    """Left-hand side of '6 Al2O3 + 1 SrCO3 == ...' -> {reduced_formula: coeff}."""
    lhs = (reaction_string or "").split("==")[0]
    out = {}
    for term in lhs.split("+"):
        m = re.match(r"\s*([\d.]+)?\s*(\S+)\s*$", term)
        if not m:
            continue
        coeff = float(m.group(1)) if m.group(1) else 1.0
        c = canon(m.group(2))
        if c:
            out[c] = out.get(c, 0.0) + coeff
    return out


def fmt_ops(ops: list[dict]) -> str:
    parts = []
    for op in ops:
        t = op.get("heating_temperature") or []
        tm = op.get("heating_time") or []
        at = op.get("heating_atmosphere") or []
        flat_t = [x for sub in t for x in (sub if isinstance(sub, list) else [sub])]
        flat_tm = [x for sub in tm for x in (sub if isinstance(sub, list) else [sub])]
        s = op.get("token") or op.get("type", "?")
        if flat_t:
            s += f" @ {'/'.join(f'{x:g}C' for x in flat_t)}"
        if flat_tm:
            s += f" for {'/'.join(f'{x:g}h' for x in flat_tm)}"
        if at:
            s += f" [{','.join(at)}]"
        parts.append(s)
    return " -> ".join(parts) if parts else "(no operations)"


def model_route_summary(route: dict) -> tuple[str, str, list[str], dict[str, float], float | None, list[str]]:
    """Returns (precursor_md, ops_md, model_prec_canon, model_amounts, tmax, atmospheres)."""
    precs, amounts, atms, tmax = [], {}, [], None
    for p in route.get("precursors", []) or []:
        f = p.get("formula", "?")
        amt = p.get("amount", "?")
        precs.append(f"{f} x{amt}")
        c = canon(f)
        if c and isinstance(amt, (int, float)):
            amounts[c] = amounts.get(c, 0.0) + amt
    op_parts = []
    for op in route.get("operations", []) or []:
        o = {"type": op.get("type", "?")}
        temp = op.get("temperature_c", op.get("temperature"))
        time = op.get("time_h", op.get("time"))
        s = o["type"]
        if isinstance(temp, (int, float)):
            s += f" @ {temp:g}C"
            tmax = max(tmax or 0, temp)
        if isinstance(time, (int, float)):
            s += f" for {time:g}h"
        if op.get("atmosphere"):
            s += f" [{op['atmosphere']}]"
            atms.append(str(op["atmosphere"]).lower())
        if op.get("media"):
            s += f" ({op['media']})"
        op_parts.append(s)
    return (" + ".join(precs) or "(none)", " -> ".join(op_parts) or "(none)",
            list(amounts), amounts, tmax, atms)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    strata = {}
    with (args.rl_dir / "rl_train.jsonl").open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                strata[r["target"]] = r.get("stratum")

    # per-completion scalar, keep the best-scoring completion per target;
    # also keep the full sample vector for p-hat context (is this score
    # typical for the target or a lucky draw?)
    best: dict[str, tuple[float, dict]] = {}
    n_ge_bar: dict[str, int] = defaultdict(int)
    n_samples: dict[str, int] = defaultdict(int)
    with args.gens.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            bd = r.get("breakdown")
            t = r["target"]
            n_samples[t] += 1
            s = scalar_reward(bd) if bd else None
            if s is not None and s >= args.bar:
                n_ge_bar[t] += 1
                if t not in best or s > best[t][0]:
                    best[t] = (s, r)
    print(f"targets with a completion >= {args.bar}: {len(best)}")

    by_stratum = defaultdict(list)
    for t in best:
        by_stratum[strata.get(t, "unknown")].append(t)
    per = max(1, args.n // max(len(by_stratum), 1))
    chosen = []
    for s in sorted(by_stratum):
        chosen.extend(rng.sample(sorted(by_stratum[s]), min(per, len(by_stratum[s]))))
    rng.shuffle(chosen)
    chosen = chosen[: args.n]
    print("sampled per stratum:", {s: min(per, len(v)) for s, v in sorted(by_stratum.items())})

    lit = lit_index(json.load(args.literature.open()))

    lines = []
    lines.append("# Validator-validity read: 20 completions scored >= "
                 f"{args.bar}, vs the literature\n")
    lines.append(f"- completions: `{args.gens}` (best-scoring per target)")
    lines.append(f"- literature: `{args.literature}` (Kononova solid-state corpus)")
    lines.append("- flags are mechanical first-pass checks to aim your eye, not verdicts\n")

    n_lit_found = 0
    overlap_fracs = []
    for i, t in enumerate(chosen, 1):
        score, rec = best[t]
        route = extract_route_json(rec["completion"]) or {}
        prec_md, ops_md, model_canon, model_amts, model_tmax, model_atms = \
            model_route_summary(route)
        bd = rec["breakdown"]
        sentinel = SynthesisValidator.SENTINEL_TAGS
        excluded = {k[:-12] for k, v in bd.items()
                    if k.endswith("_gradeability") and v in sentinel}
        check_parts = []
        for k, v in sorted(bd.items()):
            if isinstance(v, (int, float)) and not k.endswith("_gradeability"):
                # asterisk = sentinel-tagged, EXCLUDED from the scalar by
                # validate() — a 1.000 with exclusions is a weaker claim
                check_parts.append(f"{k}={v:.2f}" + ("*" if k in excluded else ""))
        tags = {k[:-12]: v for k, v in bd.items() if k.endswith("_gradeability")}

        lits = find_lit(lit, t)
        lines.append(f"\n## {i}. {t}  —  validator {score:.3f}  "
                     f"({strata.get(t, '?')}, step {rec['step']}, "
                     f"{n_ge_bar[t]}/{n_samples[t]} samples >= {args.bar})\n")
        lines.append(f"**Model route** — precursors: {prec_md}")
        lines.append(f"              operations: {ops_md}")
        lines.append(f"**Checks**: " + ", ".join(check_parts) +
                     ("   (* = sentinel/ungradeable, excluded from scalar)" if excluded else ""))
        if tags:
            lines.append(f"**Gradeability**: " + ", ".join(f"{k}:{v}" for k, v in sorted(tags.items())))

        if not lits:
            lines.append("\n**Literature**: NO ROUTE FOUND in corpus for this target")
            continue
        n_lit_found += 1
        for j, L in enumerate(lits[:2], 1):
            lit_precs = [p["formula"] for p in L.get("precursors", [])]
            lit_amts = parse_reaction_amounts(L.get("reaction_string", ""))
            lines.append(f"\n**Literature {j}** ({L.get('doi', 'no doi')}): "
                         f"`{L.get('reaction_string', '?')}`")
            lines.append(f"  precursors: {' + '.join(lit_precs)}")
            lines.append(f"  operations: {fmt_ops(L.get('operations', []))}")
            para = (L.get("paragraph_string") or "")[:280].replace("\n", " ")
            lines.append(f"  paragraph: …{para}…")

            # ---- mechanical flags ----
            lit_canon = {c for c in (canon(f) for f in lit_precs) if c}
            model_set = set(model_canon)
            inter = len(model_set & lit_canon)
            union = len(model_set | lit_canon) or 1
            overlap_fracs.append(inter / union)
            lines.append(f"  **flags**: precursor overlap {inter}/{union} "
                         f"(model∩lit / model∪lit)")
            if lit_amts and model_amts:
                shared = set(model_amts) & set(lit_amts)
                if shared:
                    mr = [model_amts[k] for k in shared]
                    lr = [lit_amts[k] for k in shared]
                    scale = sum(lr) / max(sum(mr), 1e-9)
                    dev = max(abs(m * scale - l) / max(l, 1e-9) for m, l in zip(mr, lr))
                    lines.append(f"  **flags**: amount ratios on shared precursors, "
                                 f"max rel dev {dev:.2f} (0 = perfect)")
            lit_temps = [x for op in L.get("operations", [])
                         for sub in op.get("heating_temperature", [])
                         for x in (sub if isinstance(sub, list) else [sub])]
            if lit_temps and model_tmax is not None:
                lines.append(f"  **flags**: max T model {model_tmax:g}C vs "
                             f"literature {max(lit_temps):g}C")

    lines.insert(4, f"**Coverage**: literature route found for {n_lit_found}/{args.n} targets\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} lines); "
          f"lit coverage {n_lit_found}/{args.n}; "
          f"mean precursor jaccard {sum(overlap_fracs)/max(len(overlap_fracs),1):.2f}")


if __name__ == "__main__":
    main()
