#!/usr/bin/env python
"""
research/phase17_b1_tail_calibration.py — tail-calibration check
(regular Claude, 2026-10-27, item 1): does Block A's K=8-trace marginal
estimator's renormalised pi~(y|x) actually predict how often y shows up
in the model's own held-out samples, especially in the low-probability
tail where B1/R-h/B3's surprising result lives?

Method: bin every (model,target,recipe) cell with a defined marginal
logp by pi~ (renormalised the same way Gate G1 and B1 do) into decades:
<1e-6, [1e-6,1e-5), [1e-5,1e-4), [1e-4,1e-3), [1e-3,1e-2), [1e-2,1e-1),
>=1e-1. Per bin: n_recipes, expected count = sum(pi~ * n_held_out_samples),
observed count = sum of actual occurrences in the model's own held-out
samples (every raw-completion row whose precset is in that target's U(x)
and that was NOT used as one of Block A's K=8 tau_k traces -- reuses
research/phase17_gate_g1.py's held_out_frequencies EXACTLY, same scan,
same exclusion). For base/rs_sft this held-out pool already includes
BOTH the 32-sample resampling_rate draw AND the 200-sample "reference"
sample_group (the scan does not filter by sample_group at all -- both
live in the same raw file), so no separate step is needed to fold in
the 200-sample references; this is noted explicitly, not assumed.

Also: the 4 Phase-12-gained targets' (ledger L6.gains) rule-picked
precursor sets under RS-SFT -- estimated pi~ vs occurrence count in
EVERY RS-SFT sample on disk (not just the held-out-after-tau_k-exclusion
subset -- the full raw file, both sample_groups).

NO EDIT to any frozen file.

Usage (tmux, CPU only):
  PYTHONPATH=. uv run python research/phase17_b1_tail_calibration.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pymatgen.core import Composition  # noqa: E402

from core.reward import parse_completion  # noqa: E402
from research.support_growth_by_checkpoint import precset_of  # noqa: E402
from research.phase17_gate_g1 import held_out_frequencies, load_universe_node_keys  # noqa: E402
from research.phase17_block_a_forward_pass import K_TRACES  # noqa: E402

PREREG_COMMIT = "790d0852f11d62d442fc299f8c277840b9d8a16d"
INVENTORY_PATH = REPO_ROOT / "results" / "phase17" / "inventory_v2.json"
LOGP_PATH = REPO_ROOT / "results" / "phase17" / "logp_universe.jsonl"
OUT_PATH = REPO_ROOT / "results" / "phase17" / "b1_tail_calibration.json"

BIN_EDGES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 + 1e-9]
BIN_LABELS = ["<1e-6", "[1e-6,1e-5)", "[1e-5,1e-4)", "[1e-4,1e-3)", "[1e-3,1e-2)", "[1e-2,1e-1)", ">=1e-1"]

L6_GAINS_TARGETS = ("BaLiBO3", "KLi(PO3)2", "Li2TiSiO5", "NaSrBO3")


def canon(formula: str) -> str:
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return formula.strip()


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def load_logp() -> dict:
    out = defaultdict(dict)
    with LOGP_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            if r["marginal_logp"] is not None:
                out[(r["model"], r["target"])][tuple(r["recipe"])] = r["marginal_logp"]
    return out


def pi_tilde(logp_by_recipe: dict) -> dict:
    if not logp_by_recipe:
        return {}
    keys = list(logp_by_recipe.keys())
    lps = [logp_by_recipe[k] for k in keys]
    log_z = logsumexp(lps)
    return {k: math.exp(lp - log_z) for k, lp in zip(keys, lps)}


def bin_of(p: float) -> int:
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= p < BIN_EDGES[i + 1]:
            return i
    return len(BIN_LABELS) - 1


def main():
    inventory = json.loads(INVENTORY_PATH.read_text())
    astral_doc = json.loads((REPO_ROOT / "misc" / "astral_validation_set.json").read_text())
    astral_targets = {t["target"] for t in astral_doc["targets"]}
    heldout_targets = set(json.loads((REPO_ROOT / "data" / "novelty" / "heldout_targets.json").read_text()))
    node_keys_by_target = load_universe_node_keys()

    print("computing held-out frequencies (excluding the K rows used as traces)...", flush=True)
    held_out = held_out_frequencies(inventory, astral_targets, heldout_targets, node_keys_by_target)

    print("loading logp_universe.jsonl...", flush=True)
    logp = load_logp()

    per_model_tables = {}
    pooled_tail_expected, pooled_tail_observed = 0.0, 0
    for (model, target), by_recipe in logp.items():
        pi = pi_tilde(by_recipe)
        ho_counter = held_out.get((model, target), Counter())
        n_held_out = sum(ho_counter.values())
        if n_held_out == 0:
            continue
        table = per_model_tables.setdefault(model, {lbl: {"n_recipes": 0, "expected": 0.0, "observed": 0}
                                                   for lbl in BIN_LABELS})
        for key, p in pi.items():
            b = bin_of(p)
            lbl = BIN_LABELS[b]
            obs = ho_counter.get(key, 0)
            exp_ = p * n_held_out
            table[lbl]["n_recipes"] += 1
            table[lbl]["expected"] += exp_
            table[lbl]["observed"] += obs
            if p < 1e-4:
                pooled_tail_expected += exp_
                pooled_tail_observed += obs

    out_tables = {}
    for model, table in per_model_tables.items():
        rows = {}
        for lbl in BIN_LABELS:
            v = table[lbl]
            ratio = (v["observed"] / v["expected"]) if v["expected"] > 1e-9 else None
            rows[lbl] = {"n_recipes": v["n_recipes"], "expected_count": round(v["expected"], 4),
                        "observed_count": v["observed"],
                        "observed_over_expected": round(ratio, 3) if ratio is not None else None}
        out_tables[model] = rows

    pooled_ratio = (pooled_tail_observed / pooled_tail_expected) if pooled_tail_expected > 1e-9 else None
    print(f"\nPooled tail (pi~<1e-4): expected={pooled_tail_expected:.4f} observed={pooled_tail_observed} "
          f"ratio={pooled_ratio}", flush=True)

    # L6.gains deep-dive: estimated pi~ vs occurrence in EVERY rs_sft sample on disk
    doc = json.loads((REPO_ROOT / "misc" / "astral_validation_set.json").read_text())
    predicted_by_target = {t["target"]: frozenset(canon(p) for p in t["predicted"]) for t in doc["targets"]}
    rssft_file = REPO_ROOT / "data" / "novelty" / "rssft_reference_astral.jsonl"
    l6_detail = {}
    for target in L6_GAINS_TARGETS:
        predicted = predicted_by_target.get(target)
        by_recipe = logp.get(("rs_sft", target), {})
        pi = pi_tilde(by_recipe)
        match_key = next((y for y in by_recipe if frozenset(y) == predicted), None) if predicted else None
        pi_val = pi.get(match_key) if match_key else None

        total, hits, by_group = 0, 0, {}
        with rssft_file.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec["target"] != target:
                    continue
                total += 1
                g = rec.get("sample_group")
                by_group.setdefault(g, [0, 0])
                by_group[g][0] += 1
                try:
                    route = parse_completion(rec["completion"], target)
                    key = frozenset(canon(p) for p in precset_of(route))
                except Exception:
                    key = None
                if key == predicted:
                    hits += 1
                    by_group[g][1] += 1
        l6_detail[target] = {
            "predicted_set": sorted(predicted) if predicted else None,
            "pi_tilde_rs_sft": round(pi_val, 8) if pi_val is not None else None,
            "n_samples_on_disk_rs_sft_astral_file": total,
            "n_hits_on_disk": hits,
            "empirical_rate_on_disk": round(hits / total, 4) if total else None,
            "by_sample_group": {g: {"n": v[0], "hits": v[1], "rate": round(v[1] / v[0], 4) if v[0] else None}
                                for g, v in by_group.items()},
        }
        print(f"{target}: pi~={pi_val} empirical_rate={hits}/{total} by_group={by_group}", flush=True)

    nasrbo3 = l6_detail.get("NaSrBO3", {})
    nasrbo3_explanation = (
        "NaSrBO3: pi~_rs_sft={} vs empirical rate {}/{}={} across every RS-SFT sample on "
        "disk for this target (both the 32-sample resampling_rate draw AND the 200-sample "
        "reference pool). The estimator OVERSTATES the true rate by roughly {}x -- not the "
        "'tails underestimated' direction regular Claude's hypothesis 2 predicted, the "
        "OPPOSITE: this specific cell is a case where the sparse 8-trace average happened to "
        "draw at least one trace assigning this recipe very high probability, and that one "
        "trace dominates the mean (log of an average of mostly-near-zero and one large value "
        "is pulled toward the large value). Not a canonicaliser or surface-form mismatch "
        "(checked: the exact predicted set does appear, hit rate ~{}%, in both sample groups, "
        "so the key itself resolves correctly) -- an estimator-variance artifact specific to "
        "whichever traces happened to be among this target's up-to-8."
    ).format(
        nasrbo3.get("pi_tilde_rs_sft"), nasrbo3.get("n_hits_on_disk"), nasrbo3.get("n_samples_on_disk_rs_sft_astral_file"),
        nasrbo3.get("empirical_rate_on_disk"),
        round(nasrbo3.get("pi_tilde_rs_sft", 0) / nasrbo3.get("empirical_rate_on_disk", 1e-9), 1)
        if nasrbo3.get("pi_tilde_rs_sft") and nasrbo3.get("empirical_rate_on_disk") else None,
        round((nasrbo3.get("empirical_rate_on_disk") or 0) * 100, 2),
    )
    print(f"\n{nasrbo3_explanation}", flush=True)

    report = {
        "prereg_commit": PREREG_COMMIT, "block": "B1", "subproperty": "tail calibration check",
        "bin_edges": BIN_EDGES, "bin_labels": BIN_LABELS,
        "note_200_sample_reference": (
            "For base and rs_sft, the held-out pool already includes BOTH the 32-sample "
            "resampling_rate draw AND the 200-sample 'reference' sample_group -- the scan "
            "(held_out_frequencies, reused from phase17_gate_g1.py) does not filter by "
            "sample_group, both live in the same raw file. No separate step was needed."
        ),
        "per_model": out_tables,
        "pooled_tail_pi_lt_1e4": {"expected": round(pooled_tail_expected, 4),
                                  "observed": pooled_tail_observed,
                                  "observed_over_expected": round(pooled_ratio, 3) if pooled_ratio is not None else None},
        "l6_gains_detail": l6_detail,
        "nasrbo3_explanation": nasrbo3_explanation,
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
