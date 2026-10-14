#!/usr/bin/env python
"""
scripts/build_claims_ledger.py — one deterministic record of every number the
reasoning tree and the claim sheet rely on.

Every entry says what the number is, how it is defined, where it comes from,
and whether this script recomputed it:

  RECOMPUTED       computed here from a tracked file under results/
  RECOMPUTED*      computed here, and the value or its wording differs from
                   what an earlier doc or chat stated (see `note`)
  FROM_RESULTS     copied from a tracked results file (already a derived
                   statistic, e.g. a bootstrap CI computed by its own script)
  DOC_ONLY         stated in a tracked doc; the raw data is not tracked
                   (runs/ generations, misc/), so it cannot be recomputed here
  CHAT_ONLY        stated only in a planning chat; recorded so it is not lost

Run (no GPU, no network, CPU only, a few seconds):
    uv run python scripts/build_claims_ledger.py

Writes:
    results/ledger/claims_ledger.json
    results/ledger/claims_ledger.csv
    results/ledger/CLAIMS_LEDGER.md

Everything stochastic (bootstrap) uses a fixed seed, so reruns are identical.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "ledger"
SEED = 20261012

LEDGER: list[dict] = []


def add(id, layer, claim, value, definition, source, status, note="", stated=None):
    LEDGER.append(dict(id=id, layer=layer, claim=claim, value=value,
                       definition=definition, source=source, status=status,
                       stated_earlier=stated, note=note))


def load(p):
    return json.load(open(R / p))


def mcnemar_exact(b: int, c: int) -> dict:
    """Exact McNemar on discordant counts b, c: binomial(b+c, 1/2)."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_two_sided": 1.0, "p_one_sided": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"b": b, "c": c, "p_two_sided": min(1.0, 2 * tail), "p_one_sided": tail}


def r4(x):
    return None if x is None else round(float(x), 4)


# ---------------------------------------------------------------------------
# Layer 2 — pass@k on 200 held-out targets (results/passk_n200.json)
# ---------------------------------------------------------------------------
pk = load("passk_n200.json")["per_target"]
MODELS_PK = ["base", "sft", "gdpo300"]
ks = [1, 2, 4, 8, 16]
means = {m: {k: st.mean(t["models"][m][f"pass@{k}"] for t in pk) for k in ks} for m in MODELS_PK}
n_samples = Counter(t["models"]["sft"]["n"] for t in pk)
add("L2.passk.n", 2, "pass@k test size", f"{len(pk)} targets x {dict(n_samples)} samples per model",
    "held-out targets, closed-book, success = checker score >= 0.9", "results/passk_n200.json", "RECOMPUTED")
for m in MODELS_PK:
    add(f"L2.passk.{m}", 2, f"pass@k for {m}", {f"k={k}": r4(means[m][k]) for k in ks},
        "mean over 200 targets of the per-target unbiased pass@k estimate",
        "results/passk_n200.json", "RECOMPUTED")
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(pk), size=(10000, len(pk)))
gap = {}
for k in ks:
    d = np.array([t["models"]["gdpo300"][f"pass@{k}"] - t["models"]["sft"][f"pass@{k}"] for t in pk])
    boots = d[idx].mean(axis=1)
    gap[k] = {"gap_pct_points": round(100 * d.mean(), 2),
              "ci95_pct_points": [round(100 * np.percentile(boots, 2.5), 2), round(100 * np.percentile(boots, 97.5), 2)]}
add("L2.passk.gap", 2, "GDPO-300 lead over Corpus SFT, by k",
    gap, "difference in pass@k, in PERCENTAGE POINTS of targets solved (not reward); paired bootstrap over targets, 10,000 resamples, seed fixed",
    "results/passk_n200.json", "RECOMPUTED", stated="+2.4 (k=1), +2.3 (k=8), +1.0 (k=16)",
    note="'Points' here are percentage points of the fraction of targets solved, never reward. "
         "k=16 equals the number of samples (16), so pass@16 here is 'solved at least once in these 16 samples', "
         "not an unbiased estimate; the repo's own rule says usable k <= n/2 = 8. The shrinking-gap reading "
         "should lean on k=1..8 and the paired count below.")
s16 = Counter()
for t in pk:
    a = t["models"]["sft"]["n_success"] > 0
    b = t["models"]["gdpo300"]["n_success"] > 0
    s16[(a, b)] += 1
mc = mcnemar_exact(s16[(True, False)], s16[(False, True)])
add("L2.passk.paired16", 2, "Targets solved at least once in 16 samples, paired",
    {"both": s16[(True, True)], "gdpo_only": s16[(False, True)], "sft_only": s16[(True, False)],
     "neither": s16[(False, False)], "mcnemar_p_two_sided": r4(mc["p_two_sided"])},
    "exact McNemar on discordant targets", "results/passk_n200.json", "RECOMPUTED",
    stated="169 both, 7 GDPO-only, 5 SFT-only, 19 neither, p=0.77")
base16 = means["base"][16]
add("L2.passk.headroom", 2, "Room Base left at 16 samples",
    {"base_solved_frac": r4(base16), "headroom_pct_points": round(100 * (1 - base16), 1),
     "sft_captured_frac": r4((means["sft"][16] - base16) / (1 - base16)),
     "gdpo_captured_frac": r4((means["gdpo300"][16] - base16) / (1 - base16))},
    "headroom = 1 - Base solved fraction; captured = (model - Base) / headroom",
    "results/passk_n200.json", "RECOMPUTED", stated="Base 83.5%, headroom 16.5 points, SFT 21%, GDPO 27%")
strata = defaultdict(list)
for t in pk:
    strata[t["stratum"]].append(t["models"]["sft"]["pass@1"] - t["models"]["base"]["pass@1"])
add("L2.stratum", 2, "Corpus SFT minus Base, pass@1, by target stratum",
    {s: {"delta": r4(st.mean(v)), "n": len(v)} for s, v in sorted(strata.items())},
    "mean per-target difference in pass@1", "results/passk_n200.json", "RECOMPUTED",
    stated="frac x interp +0.149 (74), frac x ungradeable +0.106 (20), int x discrete -0.008 (56), int x interp -0.028 (31), int x ungradeable -0.089 (19)")
add("L2.passk15", 2, "First pass@k look, 15 targets x 48 samples",
    {"base": [0.643, 0.720, 0.733], "sft": [0.756, 0.867, 0.867], "gdpo300": [0.778, 0.878, 0.933],
     "columns": "pass@1, pass@8, pass@48"},
    "same estimator, 15 held-out targets", "docs/TIMELINE.md (Phase 8 entry); chat 'Experimental framework review before GDPO run' (Sep 2026)",
    "DOC_ONLY", note="0.867 = 13/15 and 0.933 = 14/15 targets, so the 6.6-point lead is one target. "
                     "At pass@48 with 48 samples the value is 'solved at least once in 48', not an estimate.")

# ---------------------------------------------------------------------------
# Layer 3 — reward capacity and the constraint probe (results/hardening.json)
# ---------------------------------------------------------------------------
CHECKS10 = ["stoichiometry", "amount_accuracy", "charge_neutrality", "precursors_exist",
            "operation_order", "temperature_plausible", "thermodynamic_favorable",
            "target_stability", "chempot_atmosphere", "target_match"]


def capacity(records, checks, eps=1e-4):
    """Verbatim re-implementation of probe_hardening.capacity_metrics (ddof=1 z, ddof=0 variance)."""
    rows, gids, gindex = [], [], {}
    for r in records:
        bd = r.get("breakdown") or {}
        if bd.get("error"):
            continue
        gindex.setdefault(r["target"], len(gindex))
        rows.append([bd.get(n) if isinstance(bd.get(n), (int, float)) and not isinstance(bd.get(n), bool) else np.nan
                     for n in checks])
        gids.append(gindex[r["target"]])
    Rm, g = np.asarray(rows, float), np.asarray(gids)
    Z = np.full_like(Rm, np.nan)
    C = Rm.shape[1]
    zero, seen = np.zeros(C), np.zeros(C)
    for gg in np.unique(g):
        m = g == gg
        for c in range(C):
            col = Rm[m, c]
            v = ~np.isnan(col)
            if v.sum() < 2:
                continue
            seen[c] += 1
            vals = col[v]
            sd = vals.std(ddof=1)
            if sd < 1e-12:
                zero[c] += 1
                z = np.zeros_like(vals)
            else:
                z = (vals - vals.mean()) / (sd + eps)
            Z[np.where(m)[0][v], c] = z
    cov = np.zeros((C, C))
    for i in range(C):
        for j in range(i, C):
            both = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
            if both.sum() > 2:
                cov[i, j] = cov[j, i] = np.cov(Z[both, i], Z[both, j])[0, 1]
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    total = float(eig[eig > 0].sum())
    zvar = {checks[c]: r4(np.nanvar(Z[:, c][~np.isnan(Z[:, c])])) for c in range(C)}
    return {"total": round(total, 4), "capacity_pct": round(100 * total / C, 2),
            "zero_std_pct": {checks[c]: round(100 * zero[c] / seen[c], 1) if seen[c] else None for c in range(C)},
            "z_var": zvar}


h = load("hardening.json")
cap_rows = {}
for cond in h["conditions"]:
    recs = [r for r in h["records"] if r["condition"] == cond]
    cap = capacity(recs, CHECKS10)
    by_t = defaultdict(list)
    for r in recs:
        if r.get("precursor_set"):
            by_t[r["target"]].append(tuple(sorted(r["precursor_set"])))
    routes = st.mean(len(set(v)) for v in by_t.values())
    ident = sum(1 for v in by_t.values() if len(set(v)) == 1) / len(by_t)
    rw = defaultdict(list)
    for r in recs:
        if r.get("reward") is not None:
            rw[r["target"]].append(r["reward"])
    phat = st.mean(sum(x >= 0.9 for x in v) / len(v) for v in rw.values() if v)  # probe_hardening.phat_metrics
    adh = {}
    if cond in ("temp_ceiling", "combined"):
        tk = [r["adherence"].get("temp_ok") for r in recs if "temp_ok" in (r.get("adherence") or {})]
        if tk:
            adh["temp_ok_frac"] = r4(sum(bool(x) for x in tk) / len(tk))
        mt = [r["adherence"]["max_T_reported"] for r in recs if (r.get("adherence") or {}).get("max_T_reported") is not None]
        if mt:
            adh["mean_max_T_reported_C"] = round(st.mean(mt), 1)
    if cond in ("inventory", "combined"):
        ik = [r["adherence"].get("inventory_ok") for r in recs if "inventory_ok" in (r.get("adherence") or {})]
        if ik:
            adh["inventory_ok_frac"] = r4(sum(bool(x) for x in ik) / len(ik))
    cap_rows[cond] = {"capacity_pct": cap["capacity_pct"],
                      "capacity_pct_vs_reachable_max": round(cap["capacity_pct"] / 0.875, 2),
                      "powder_sets_per_8": round(routes, 2), "groups_all_identical_pct": round(100 * ident, 1),
                      "p_hat_at_0.9": r4(phat), **adh}
add("L3.hardening", 3, "Constraint probe: capacity and variety by condition (Corpus SFT, 40 targets x 8)",
    cap_rows, "capacity = sum of within-group z-variance over 10 checks / 10 (probe_hardening.capacity_metrics)",
    "results/hardening.json", "RECOMPUTED",
    stated="16.4 / 17.7 / 16.4 / 16.3 / 16.1 %; 2.17 / 2.15 / 1.65 / 2.10 / 1.75 sets; 37.5 / 37.5 / 62.5 / 35.0 / 55.0 % identical",
    note="Definition detail: z-scores use the sample std (ddof=1) and the variance of z uses ddof=0. "
         "With 8 answers per group, a check that varies in every group therefore reaches 7/8 = 0.875, not 1. "
         "The reachable maximum is 87.5%, so every capacity number reads about 1/0.875 = 1.14x higher on a 0-100% "
         "reachable scale (column capacity_pct_vs_reachable_max). The 40% bar was set on the original scale, "
         "so the pass/fail verdicts do not change.")
add("L3.capacity.run1", 3, "Reward capacity during GDPO-300 training (run 1)",
    {"total_z_variance": 1.43, "capacity_pct": 14.3,
     "per_check_z_var": {"amount_accuracy": 0.601, "thermodynamic_favorable": 0.363, "operation_order": 0.186,
                         "stoichiometry": 0.142, "chempot_atmosphere": 0.105, "precursors_exist": 0.028,
                         "temperature_plausible": 0.006, "charge_neutrality": 0.0, "target_match": 0.0,
                         "target_stability": 0.0},
     "zero_std_groups_pct": {"amount_accuracy": 31.0, "thermodynamic_favorable": 57.8, "operation_order": 78.8,
                             "stoichiometry": 83.5, "chempot_atmosphere": 87.4, "precursors_exist": 96.7,
                             "temperature_plausible": 99.3, "charge_neutrality": 100, "target_match": 100,
                             "target_stability": 100},
     "n": "4,896 completions in 612 groups of 8", "top_eigenvector_loading_amount_accuracy": -0.97},
    "reward_geometry.py D1 on runs/gdpo-qlora-gdpo-v3/generations.jsonl", "CLAUDE.md finding 1; chat 'Experimental framework review before GDPO run' (Aug-Sep 2026)",
    "DOC_ONLY", note="The generations file is not tracked, so this cannot be recomputed here. Each z-variance is "
                     "(fraction of groups where the check varies) x 0.875, e.g. operation_order: 21.2% x 0.875 = 0.186. "
                     "The 'other eight' sum to 0.467.")
add("L3.leash", 3, "Drift penalty loosened (beta 0.04 -> 0.001, lr 5e-6 -> 1e-5), resumed from step 300",
    {"KL_by_step": {"25": 0.0007, "50": 0.0008, "75": 0.00101, "100": 0.00157, "125": 0.0022, "150": 0.00261},
     "clip_ratio": 0, "entropy": "0.629 -> 0.696"},
    "KL from the reference (Base) as logged by TRL", "chat 'Experimental framework review before GDPO run' (Aug 2026); CLAUDE.md finding 7",
    "CHAT_ONLY")
add("L3.temperature_sweep", 3, "Sampling temperature vs variety (Corpus SFT, 20 targets x 8)",
    {"T=1.0": 1.90, "T=1.2": 2.30, "T=1.5": 2.05, "targets_never_above_2_sets": "8 of 20"},
    "distinct starting-powder sets per group of 8", "docs/TIMELINE.md", "DOC_ONLY")
add("L3.low_temp", 3, "Low-temperature objective probe (40 targets x 8)",
    {"capacity_pct_11_checks": 23.8, "capacity_pct_10_checks": 19.0, "temperature_economy_z_var": 0.721,
     "mean_max_T_C": 1056, "ceiling_condition_mean_max_T_C": 993},
    "same capacity metric, with the new check added", "CLAUDE.md finding 15", "DOC_ONLY")
add("L3.temp_hack", 3, "Reported temperatures vs the literature, 20 high-scoring answers",
    {"comparable": 18, "hotter": 15, "median_delta_C": 200, "hotter_by_200C_or_more": 10, "max_delta_C": 750,
     "ceiling_compliance": "100%"},
    "model max temperature minus published max temperature for the same target",
    "CLAUDE.md finding 6 (misc/validity_read_20.md, untracked)", "DOC_ONLY")
add("L3.prereg_june", 2, "Failure signs written before any practice run, and what was seen",
    {"spare_capacity": "LoRA capacity use flat across checkpoints",
     "advantage_collapse": "capacity 14%, 8 of 10 checks always tied",
     "representations": "CKA(Corpus SFT, GDPO-300) about 0.99 from layer 20 on; layer mean 0.96 (CLAUDE.md finding 5)",
     "drift_vs_reward": "KL peaked at 0.0026 (in the loosened-leash resume); reward 0.955 -> 0.988"},
    "", "chat 'Experimental framework review before GDPO run' (Volume I, sec. 3.5)", "CHAT_ONLY",
    note="The 0.99 is the late-layer value; the all-layer mean reported in CLAUDE.md is 0.96. "
         "The principal-angle figures (22 vs 35 degrees) were computed while a plotting bug (svdvals vs arccos) "
         "was open; do not quote them without a recheck.")

# ---------------------------------------------------------------------------
# Layer 1 — the instrument (chat and doc only; early data not tracked)
# ---------------------------------------------------------------------------
add("L1.six_checkpoints", 1, "Six separately trained checkpoints, identical scores",
    {"before_fix": "0.2846 +/- 0.0535", "after_fix": "0.853-0.869 on five of the six"},
    "mean checker reward on the early eval set", "chat 'MIRA project materials database integration' (Jun 2026)",
    "CHAT_ONLY", stated="the same six models scored between 0.853 and 0.869",
    note="The chat says five of the six were re-scored in that range; the tree said six. Corrected to five of six.")
add("L1.bugs", 1, "Bug ledger values quoted in the tree",
    {"sentinel_free_reward_max": "about 0.225", "truncation": "max_seq_len 1,024 vs median 4,092 tokens -> 100% of v3 sequences cut",
     "LFP_gradeability": "3.6% -> 61.7%", "corrupt_PD_shards": "8.5%, 1,691 refetched", "tests": 43,
     "corpus_gradeable_after_fixes": "85.4%"},
    "", "chat 'Experimental framework review before GDPO run' (Volume I bug ledger); CLAUDE.md", "CHAT_ONLY",
    note="max_seq_len 1,024 was chosen on an earlier dataset (mean 616 tokens, 0.11% truncated) and was not "
         "re-checked when the corpus switched to long teacher traces.")

# ---------------------------------------------------------------------------
# Layer 4 — ASTRAL as the outside referee
# ---------------------------------------------------------------------------
av = load("astral_validation_set.json")
T = av["targets"]
diff = [t["pred_best_purity"] - t["trad_best_purity"] for t in T]
add("L4.purity_split", 4, "Which recipe the robot lab measured as purer (best temperature of each)",
    {"rule_picked_purer": sum(d > 0 for d in diff), "conventional_purer": sum(d < 0 for d in diff),
     "ties": sum(d == 0 for d in diff),
     "at_0.10_margin": {"rule_picked_clearly": sum(d > 0.10 for d in diff), "within_0.10": sum(abs(d) <= 0.10 for d in diff),
                        "conventional_clearly": sum(d < -0.10 for d in diff)},
     "header_in_same_file": {k: v for k, v in av["headline_numbers"].items() if k.startswith("targets_where")}},
    "per-target best phase purity, rule-picked (predicted) vs conventional (traditional)",
    "results/astral_validation_set.json", "RECOMPUTED*",
    stated="'the rule-picked recipe usually came out purer' (early tree); header 15 / 16 / 4",
    note="The file's own header (15 better by 10%, 16 similar, 4 worse) cannot be reproduced from its per-target "
         "purities under either an absolute 0.10 or a relative 10% margin (13/16/6 and 15/12/8). Use the per-target "
         "counts: purer on 20 of 35; clearly purer (>0.10) on 13, within 0.10 on 16, clearly worse on 6.")
add("L4.lab_temperatures", 4, "Mean best temperature in the lab",
    {"conventional_recipe_C": round(st.mean(t["trad_best_T"] for t in T), 1),
     "rule_picked_recipe_C": round(st.mean(t["pred_best_T"] for t in T), 1)},
    "mean over 35 targets of the temperature at which each recipe reached its best purity",
    "results/astral_validation_set.json", "RECOMPUTED*", stated="'lab best, 763 C'",
    note="763 C is the conventional recipe's mean best temperature, not 'the lab best'. Overshoot numbers "
         "(+264 C etc.) are measured against this conventional figure.")
vc = load("astral_validator_correlation.json")["validator"]
xs, ys = [], []
agree = ties = dec = 0
for r in vc["rows"]:
    xs += [r["trad_score"], r["pred_score"]]
    ys += [r["trad_best_purity"], r["pred_best_purity"]]
    if r["trad_score"] == r["pred_score"]:
        ties += 1
        continue
    dec += 1
    agree += (r["pred_score"] > r["trad_score"]) == (r["pred_best_purity"] > r["trad_best_purity"])
rho, p = spearmanr(xs, ys)
add("L4.checker_vs_purity", 4, "Checker score vs measured purity, 70 recipes",
    {"spearman_rho": r4(rho), "p": r4(p), "pairs_agree": agree, "pairs_decided": dec, "pairs_tied": ties},
    "Spearman rank correlation; a pair agrees when the checker's higher-scored recipe is the purer one",
    "results/astral_validator_correlation.json", "RECOMPUTED", stated="rho 0.228, p 0.058; 17 of 34")
cc = load("corpus_coverage.json")
nov, tra = cc["novel_precursor_corpus_freq"], cc["traditional_precursor_corpus_freq"]
add("L4.corpus_freq", 4, "How often the starting powders appear in the training corpus",
    {"rule_picked_powders": nov, "conventional_powders": tra,
     "mean_rule_picked": round(st.mean(nov.values()), 2), "mean_conventional": round(st.mean(tra.values()), 1),
     "never_appear": [k for k, v in nov.items() if v == 0]},
    "mentions in the 17.6k-route Kononova corpus", "results/corpus_coverage.json", "RECOMPUTED",
    stated="2.6 vs 674.8; LiNbO3, KPO3, NaBO2 never")

# ---------------------------------------------------------------------------
# Layers 4-6 — model generations on ASTRAL, 35 targets x 32 samples
# ---------------------------------------------------------------------------
GEN = {"Base": "astral_gen_n32_base.json", "Corpus SFT": "astral_gen_n32_sft.json",
       "Format-only SFT": "astral_gen_n32_format_only.json", "RS-SFT": "astral_gen_n32_rs_sft.json",
       "GDPO-300": "astral_gen_n32_gdpo300.json", "GDPO from RS-SFT": "astral_gen_n32_gdpo_phase12_beta0.json",
       "E1 (GDPO from RS-SFT, ranker reward)": "astral_gen_n32_e1.json"}
gen = {m: {r["target"]: r for r in load(f)["results"]} for m, f in GEN.items()}
targets = list(gen["Base"])
overlap = set(load("phase16/astral_clean_subset.json")["overlapping_targets"])
clean = [t for t in targets if t not in overlap]
conv_bestT = st.mean(t["trad_best_T"] for t in T)


def hit(m, t, superset=False):
    ms = [s.get("match") for s in gen[m][t]["samples"]]
    return ("PREDICTED" in ms) or (superset and "PREDICTED_SUPERSET" in ms)


def conv(m, t):
    return "TRADITIONAL" in [s.get("match") for s in gen[m][t]["samples"]]


per_model = {}
for m in GEN:
    samples = [s for t in targets for s in gen[m][t]["samples"]]
    parsed = [s for s in samples if s.get("match") != "PARSE_FAILURE"]
    pr = [s["reward"] for s in samples if s.get("match") == "PREDICTED" and s.get("reward") is not None]
    tr = [s["reward"] for s in samples if s.get("match") == "TRADITIONAL" and s.get("reward") is not None]
    mt = [s["max_T"] for s in samples if s.get("max_T") is not None]
    per_model[m] = {
        "rule_picked_targets_exact": sum(hit(m, t) for t in targets),
        "rule_picked_targets_incl_superset": sum(hit(m, t, True) for t in targets),
        "rule_picked_targets_clean22_exact": sum(hit(m, t) for t in clean),
        "conventional_targets": sum(conv(m, t) for t in targets),
        "rule_picked_samples": sum(1 for s in samples if s.get("match") == "PREDICTED"),
        "conventional_samples": sum(1 for s in samples if s.get("match") == "TRADITIONAL"),
        "parse_rate": r4(len(parsed) / len(samples)),
        "checker_gap_rule_minus_conventional": r4(st.mean(pr) - st.mean(tr)) if pr and tr else None,
        "mean_max_T_C": round(st.mean(mt), 1) if mt else None,
        "overshoot_vs_conventional_best_T_C": round(st.mean(mt) - conv_bestT, 1) if mt else None,
    }
add("L5.per_model", 5, "Rule-picked and conventional recipes proposed, per model (35 targets x 32 samples)",
    per_model, "a target counts when at least one of 32 samples uses exactly the rule-picked (or conventional) powder set; "
               "'incl_superset' also counts a set that contains the rule-picked powders plus extras",
    "results/astral_gen_n32_*.json", "RECOMPUTED",
    stated="Base 10/24, Corpus SFT 0(1)/32, Format-only 11/31, RS-SFT 10/17, GDPO-300 3/31, GDPO from RS-SFT 14/17",
    note="Format-only's 11 and GDPO-300's 3 include one superset hit each; exact counts are 10 and 2.")
pairs = [("Base", "Corpus SFT", False, targets), ("Base", "Corpus SFT", True, targets),
         ("Base", "Corpus SFT", False, clean), ("Corpus SFT", "RS-SFT", False, targets),
         ("Corpus SFT", "RS-SFT", True, targets), ("Base", "Format-only SFT", False, targets),
         ("Base", "RS-SFT", False, targets), ("RS-SFT", "GDPO from RS-SFT", False, targets),
         ("RS-SFT", "GDPO from RS-SFT", False, clean), ("Corpus SFT", "GDPO-300", False, targets),
         ("RS-SFT", "E1 (GDPO from RS-SFT, ranker reward)", False, targets),
         ("RS-SFT", "E1 (GDPO from RS-SFT, ranker reward)", False, clean)]
mcn = {}
for a, b, sup, tg in pairs:
    only_a = sum(hit(a, t, sup) and not hit(b, t, sup) for t in tg)
    only_b = sum(hit(b, t, sup) and not hit(a, t, sup) for t in tg)
    res = mcnemar_exact(only_a, only_b)
    key = f"{a} -> {b} ({'clean 22' if tg is clean else 'all 35'}{', superset counts' if sup else ''})"
    mcn[key] = {"lost": only_a, "gained": only_b, "p_two_sided": r4(res["p_two_sided"]),
                "p_one_sided": r4(res["p_one_sided"])}
add("L5.mcnemar", 5, "Paired tests on rule-picked hits", mcn,
    "exact McNemar on targets where the two models disagree", "results/astral_gen_n32_*.json", "RECOMPUTED*",
    stated="Base->SFT p=0.0020 (35) / 0.0078 (22); SFT->RS-SFT 'p=0.004-0.012'; RS-SFT->GDPO p=0.125 (35), 'same p' on 22",
    note="RS-SFT -> GDPO is p=0.125 two-sided on 35 (one-sided 0.0625) but p=0.25 two-sided on the clean 22 "
         "(one-sided 0.125). The tree's 'same p on the clean 22' was wrong. CLAUDE.md's old 'p=0.004-0.012' for "
         "Base->SFT predates the exact recount; exact values are 0.0020 (exact match) and 0.0039 (superset counted).")
a12 = load("ammonium_ablation_check.json")
amm = {}
for m in ["Base", "RS-SFT", "GDPO from RS-SFT"]:
    amm[m] = {g: f"{sum(hit(m, t) for t in a12[k])}/{len(a12[k])}"
              for g, k in [("bug_can_touch", "ammonium_affected_targets"), ("bug_cannot_touch", "not_affected_targets")]}
add("L6.ammonium_split", 6, "Rule-picked hits split by whether the ammonium bug could touch the target",
    amm, "targets with at least one exact rule-picked sample", "results/ammonium_ablation_check.json + astral_gen_n32_*.json",
    "RECOMPUTED", stated="unaffected 47.1% -> 52.9% -> 70.6%; affected 11.1% -> 5.6% -> 11.1%")
gains = []
for t in targets:
    if hit("GDPO from RS-SFT", t) and not hit("RS-SFT", t):
        row = next(x for x in T if x["target"] == t)
        gains.append({"target": t, "clean": t not in overlap,
                      "rs_sft_samples": sum(s.get("match") == "PREDICTED" for s in gen["RS-SFT"][t]["samples"]),
                      "gdpo_samples": sum(s.get("match") == "PREDICTED" for s in gen["GDPO from RS-SFT"][t]["samples"]),
                      "purity_rule_picked": row["pred_best_purity"], "purity_conventional": row["trad_best_purity"]})
add("L6.gains", 6, "Targets GDPO from RS-SFT gained over RS-SFT", gains,
    "exact rule-picked hits", "results/astral_gen_n32_*.json + astral_validation_set.json", "RECOMPUTED",
    note="Only BaLiBO3 is a target where the rule-picked recipe was clearly purer; KLi(PO3)2 is within error; "
         "Li2TiSiO5 and NaSrBO3 were purer by the conventional recipe.")
p12 = load("phase12_astral_final_analysis.json")
add("L6.phase12", 6, "Phase 12 read at checkpoint 300",
    {"rs_sft": p12["rs_sft_baseline"]["any_predicted"], "gdpo": p12["phase12_gdpo_beta0_ckpt300"]["any_predicted"],
     "bar": "amplify > 12, preserve 10-12, collapse < 10", "verdict": p12["verdict"][:40]},
    "", "results/phase12_astral_final_analysis.json", "FROM_RESULTS")

# ---------------------------------------------------------------------------
# Layer 7 — judges against outside data
# ---------------------------------------------------------------------------
rv = load("ranker_v2_astral_gate.json")
add("L7.ranker_v2", 7, "Ranker v2 against ASTRAL", {"agree": rv["n_agree"], "of": rv["n_pairs"], "bar": rv["threshold_n"],
                                                   "spearman": r4(rv["stats"]["spearman_rho"])},
    "pair agreement with measured purity", "results/ranker_v2_astral_gate.json", "FROM_RESULTS",
    note="Three of its eight checks could not be graded on any ASTRAL pair (no amounts, no per-route temperatures).")
p13 = load("phase13_astral_scoring.json")["primary"]
p13b = load("phase13_astral_scoring_v2.json")
add("L7.comparator_astral", 7, "Comparator against ASTRAL",
    {"iteration1": {"prefer": p13["n_pref"], "against": p13["n_against"], "tie": p13["n_tie"], "bar": p13["bar"]},
     "iteration2_sign_agreement": f"{p13b['primary_sign_agreement']['agree']}/{p13b['primary_sign_agreement']['n_decided']}",
     "iteration2_prefers_rule_picked": f"{p13b['secondary_n_pref']['n_pref']}/35"},
    "", "results/phase13_astral_scoring*.json", "FROM_RESULTS",
    note="Every conventional ASTRAL recipe has 3 powders and every rule-picked one has 2, so a powder-count-sensitive judge cannot be tested here.")
ag = load("external/arrows_gate.json")["primary_endpoint_ybco"]
t4d = load("external/arrows_task4d_diagnostics.json")["thermo_analysis"]
t4e = load("external/arrows_task4e_additions.json")
add("L7.arrows", 7, "Judges against ARROWS3 robot-lab yields (YBCO, pairs differing by >= 5 wt%)",
    {"checker_vote": r4(ag["validator_gdpo_vote_agreement"]), "avoid_carbonates_rule": r4(ag["best_baseline_agreement"]),
     "checker_minus_rule": r4(ag["validator_diff"]), "checker_minus_rule_ci95": [r4(x) for x in ag["validator_diff_95ci"]],
     "comparator": r4(ag["comparator_agreement"]), "comparator_minus_rule_ci95": [r4(x) for x in ag["comparator_diff_95ci"]],
     "rule_minus_chance_ci95": [r4(x) for x in t4e["carbonate_free_baseline_significance"]["diff_95ci"]],
     "thermo_check_alone_all_pairs": r4(t4d["a_thermo_vs_best_baseline"]["thermo_agreement"]),
     "thermo_check_alone_decisive_pairs": {"agreement": r4(t4d["b_decisive_pairs_only"]["thermo_agreement"]),
                                           "n": t4d["b_decisive_pairs_only"]["n_decisive"]},
     "thermo_when_carbonate_status_equal": {"agreement": r4(t4d["c_carbfree_tie_pairs"]["thermo_agreement"]),
                                            "ci95": [r4(x) for x in t4d["c_carbfree_tie_pairs"]["thermo_agreement_95ci"]]},
     "comparator_gate_decided": {"n": t4e["gate_decided_direct"]["n"], "agreement": r4(t4e["gate_decided_direct"]["agreement"])}},
    "agreement = share of pairs where the judge prefers the higher-yield recipe; CIs by the gate script's own bootstrap",
    "results/external/arrows_gate.json, arrows_task4d_diagnostics.json, arrows_task4e_additions.json", "FROM_RESULTS",
    note="'Thermodynamic check alone 71.1%' holds only on the 521 pairs where that check is decisive; over all pairs it "
         "scores 0.558, below the avoid-carbonates rule (0.583).")

# ---------------------------------------------------------------------------
# Layer 8 — what practice changed
# ---------------------------------------------------------------------------
al = load("distributional/alkali_source_share.json")
cs = load("phase16/astral_clean_subset.json")["carbonate_shares"]
per_triple = {k: {kk: r4(v) for kk, v in al[k]["shares"].items() if kk != "missing"} | {"n": al[k]["n_triples"]}
              for k in ["base_on_astral", "rs_sft_on_astral", "gdpo_on_astral", "rs_sft_training_set"]}
per_triple["full_sft_on_astral"] = {kk: r4(v) for kk, v in cs["full_sft"]["full_35"]["shares"].items() if kk != "missing"} | {"n": cs["full_sft"]["full_35"]["n_triples"]}
bare = {"Li2O", "Na2O", "K2O", "Rb2O", "Cs2O"}
carb_any = {"Li2CO3", "Na2CO3", "K2CO3", "Rb2CO3", "Cs2CO3", "BaCO3", "SrCO3", "CaCO3", "MgCO3"}
whole = {}
for m in ["Base", "Corpus SFT", "RS-SFT", "GDPO from RS-SFT"]:
    s = [x for t in targets for x in gen[m][t]["samples"] if x.get("precursors")]
    whole[m] = {"any_bare_alkali_oxide_pct": round(100 * sum(any(p in bare for p in x["precursors"]) for x in s) / len(s), 1),
                "any_carbonate_incl_alkaline_earth_pct": round(100 * sum(any(p in carb_any for p in x["precursors"]) for x in s) / len(s), 1)}
add("L8.alkali_share", 8, "How the model sources lithium, sodium and potassium",
    {"per_alkali_choice": per_triple, "whole_sample_definition": whole},
    "per_alkali_choice: for each (target, sample, alkali element) in alkali-containing targets, is that element "
    "supplied by its bare oxide, its carbonate, or something else. whole_sample: share of parsed samples that "
    "contain any bare alkali oxide / any carbonate including Ba, Sr, Ca, Mg carbonates",
    "results/distributional/alkali_source_share.json, phase16/astral_clean_subset.json, astral_gen_n32_*.json",
    "RECOMPUTED*", stated="carbonate 74.7 -> 98.5 -> 48.0 -> 19.1 %, bare oxide 8.6 -> 0.5 -> 25.1 -> 56.3 %",
    note="The often-quoted 74.7/19.1 and 8.6/56.3 are the whole-sample numbers, whose carbonate set includes "
         "alkaline-earth carbonates. A sentence about how lithium, sodium and potassium are sourced must use the "
         "per-alkali-choice numbers: carbonate 72.9 -> 98.2 -> 45.9 -> 18.1 %, bare oxide 9.1 -> 0.5 -> 25.8 -> 56.9 %.")
sj = load("distributional/splithalf_jaccard.json")
add("L8.jaccard", 8, "Overlap of precursor sets between models vs each model with itself",
    {"self": {k: r4(v) for k, v in sj["self"].items()}, "cross": {k: r4(v) for k, v in sj["cross"].items()}},
    "Jaccard overlap of the precursor sets in two random halves of 16 samples per target, 200 splits",
    "results/distributional/splithalf_jaccard.json", "FROM_RESULTS",
    note="RS-SFT vs GDPO (0.525) sits between GDPO's self-overlap (0.523) and RS-SFT's (0.588): indistinguishable "
         "from resampling. Every pair involving Base is clearly lower (0.33-0.48).")
gw = load("distributional/gdpo_within_group_bareoxide.json")
add("L8.within_group", 8, "Inside GDPO training groups that mixed a bare-oxide and a carbonate answer",
    {"mixed_groups": gw["n_mixed_groups"], "of_groups": gw["n_groups_total"],
     "thermo_z_gap": r4(gw["per_channel"]["thermodynamic_favorable"]["mean_z_gap"]),
     "other_checks_z_gap": {c: r4(v["mean_z_gap"]) for c, v in gw["per_channel"].items() if c != "thermodynamic_favorable"}},
    "mean (bare-oxide z - carbonate z) per check", "results/distributional/gdpo_within_group_bareoxide.json", "FROM_RESULTS")
lee = load("external/lee2025_precursor_usage.json")
add("L8.lee", 8, "What chemists choose (Lee et al. 2025, 80,806 recipes)",
    {"carbonate_share": r4(lee["primary"]["carbonate_share"]), "bare_oxide_share": r4(lee["primary"]["bare_share"]),
     "n_alkali_choices": lee["primary"]["n_triples"],
     "impurity_diff_bare_minus_carbonate": r4(lee["secondary_pooled"]["difference"]),
     "impurity_diff_ci95": [r4(x) for x in lee["secondary_pooled"]["difference_95ci"]],
     "same_target_diff": r4(lee["secondary_restricted_both_sourcing_targets"]["difference"]),
     "same_target_ci95": [r4(x) for x in lee["secondary_restricted_both_sourcing_targets"]["difference_95ci"]]},
    "shares of (record, alkali element) choices; impurity = recipe reported with an impurity phase",
    "results/external/lee2025_precursor_usage.json", "FROM_RESULTS")
dr = load("dose_response/analysis.json")["q4_policy_carbonate"]
rb = load("dose_response/robustness.json")["part1_target_control"]
add("L8.teacher_forcing", 8, "Shift in log-probability, GDPO from RS-SFT minus RS-SFT, on 75 ARROWS3 recipes",
    {"carbonate_coef": r4(dr["coefficients"]["carbonate_status"]), "carbonate_ci95": [r4(x) for x in dr["carbonate_status_ci"]],
     "yield_coef_per_wt_pct": r4(dr["coefficients"]["best_wt_pct"]), "yield_ci95": [r4(x) for x in dr["target_wt_pct_ci"]],
     "with_target_fixed_effects": {"carbonate_coef": r4(rb["coefficients_with_target_fe"]["carbonate_status"]),
                                   "ci95": [r4(x) for x in rb["carbonate_status_ci_with_fe"]]}},
    "regression of the log-probability shift on carbonate status and measured yield (nats; empty <think> block)",
    "results/dose_response/analysis.json, robustness.json", "FROM_RESULTS",
    stated="-1.979 [-3.554, -0.352] (doc); -2.35 nats (decision tree)",
    note="-1.979 is the pre-registered model without target fixed effects; -2.35 is the robustness check with them. "
         "The tracked JSON's CI is [-3.531, -0.327]; PHASE15_DOSE_RESULTS.md prints [-3.554, -0.352] (a different "
         "bootstrap run). Quote the JSON.")

# ---------------------------------------------------------------------------
# Layer 9 — contamination; Layer 10 — Phase 16
# ---------------------------------------------------------------------------
ov = load("phase16/astral_overlap.json")
add("L9.overlap", 9, "ASTRAL targets that appear anywhere in project data",
    {"overlapping": ov["n_overlapping_targets"], "clean": ov["clean_subset_size"],
     "clean_hits": {m: per_model[m]["rule_picked_targets_clean22_exact"] for m in ["Base", "Corpus SFT", "RS-SFT", "GDPO from RS-SFT"]}},
    "", "results/phase16/astral_overlap.json + astral_gen_n32_*.json", "RECOMPUTED")
ef = load("phase16/e3_e3c_filter_report.json")
add("L10.filter", 10, "E3/E3c data filter", ef, "", "results/phase16/e3_e3c_filter_report.json", "FROM_RESULTS",
    note="Rare = one starting powder used in 17 or fewer corpus recipes, or a powder set never published for that target.")
vr = load("phase16/validator_v2_rescore.json")
add("L10.validator_v2", 10, "Recipes whose balance verdict flips under the ammonium fix",
    {m: f"{v['pct_flipped']}% ({v['n_flipped_v1_fail_v2_pass']}/{v['n_total_samples']})" for m, v in vr.items()},
    "", "results/phase16/validator_v2_rescore.json", "FROM_RESULTS")
add("L5.controls", 5, "How the two Phase 11 controls were built",
    {"format_only_sft": "300 random examples of the 1,217-example corpus set, 10 steps, fresh LoRA from Base; parse rate 1,112/1,120 = 99.3%",
     "rs_sft": "8 Base samples for each of 400 targets, kept if checker >= 0.9; 295 targets survived (73.75%); "
               "trained on 283, 12 held out for eval loss"},
    "", "docs/phases/PHASE11_RESULTS.md", "DOC_ONLY")
add("L6.run_diagnostics", 6, "Phase 12 run diagnostics",
    {"parse_failures": "18.2% -> 12.4% (69 -> 47 of 380) after two reader fixes; 46 of 47 remaining are 8,192-token truncations",
     "operation_order_smoke": "1.000 on all 311 scored completions; predicted to stay silent; it did",
     "kl_blowup": {"step": [25, 50, 75], "kl": [1.32, 97.2, 1.99e11], "grad_norm": [616.0, 1.84, 0.072]},
     "relaunch": "beta = 0; grad norm never above about 0.45"},
    "", "docs/phases/PHASE12_RESULTS.md", "DOC_ONLY")
E1 = "E1 (GDPO from RS-SFT, ranker reward)"
tempdev = {}
for m in ["Base", "RS-SFT", "GDPO from RS-SFT", E1]:
    d = []
    for t in targets:
        ts = [s["max_T"] for s in gen[m][t]["samples"] if s.get("max_T") is not None]
        row = next(x for x in T if x["target"] == t)
        best = row["pred_best_T"] if row["pred_best_purity"] >= row["trad_best_purity"] else row["trad_best_T"]
        if ts:
            d.append(st.mean(ts) - best)
    tempdev[m] = {"mean_signed_C": round(st.mean(d), 1), "mean_abs_C": round(st.mean(abs(x) for x in d), 1),
                  "targets_within_100C": sum(abs(x) <= 100 for x in d), "targets_too_hot_100C": sum(x > 100 for x in d),
                  "targets_too_cold_100C": sum(x < -100 for x in d)}
add("L10.e1_temperature", 10, "Model temperature vs the lab's best temperature, target by target",
    tempdev, "per target: model's mean max temperature minus the temperature at which the purer of the two lab recipes "
             "reached its best purity", "results/astral_gen_n32_*.json + astral_validation_set.json", "RECOMPUTED",
    stated="E1 overshoot +16.5 C (mean, vs conventional best)",
    note="E1's near-zero mean hides a wide spread: it cut mean absolute error from 189 C (RS-SFT) to 120 C, but undershoots "
         "by more than 100 C on 10 targets, which no earlier model did.")
sg = load("phase16/e1_support_growth.json")
add("L10.e1_support", 10, "E1 support growth vs RS-SFT's own resampling floor",
    {k: {kk: sg[k][kk].get("pooled_share") for kk in sg[k] if isinstance(sg[k][kk], dict) and "pooled_share" in sg[k][kk]}
     for k in sg if isinstance(sg[k], dict)},
    "share of E1's distinct precursor sets absent from RS-SFT's 200-sample reference, vs RS-SFT's own fresh 32-sample draw",
    "results/phase16/e1_support_growth.json", "FROM_RESULTS")
kb = load("kl_blowup_diagnostic.json")
top = sorted(kb["results"], key=lambda x: -x["max_r"])[:10]
add("L6.kl_tokens", 6, "Largest single-token log-ratios in the KL-blowup hunt",
    {"max_r_nats": r4(top[0]["max_r"]), "top10_tokens": [x["max_r_token"] for x in top],
     "top10_literal_digit_tokens": sum(x["max_r_token"].strip().isdigit() for x in top)},
    "RS-SFT used as a stand-in for the lost step-75 policy", "results/kl_blowup_diagnostic.json", "RECOMPUTED",
    stated="7 of top 10 are digits", note="4 of the top 10 are literal digit tokens; 7 of 10 sit inside numeric "
                                           "or stoichiometry arithmetic in their context. Say 'inside arithmetic', not 'digits'.")


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
def flat(v):
    return json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v


OUT.mkdir(parents=True, exist_ok=True)
json.dump({"generated_by": "scripts/build_claims_ledger.py", "seed": SEED, "entries": LEDGER},
          open(OUT / "claims_ledger.json", "w"), indent=1, ensure_ascii=False)
with open(OUT / "claims_ledger.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(LEDGER[0]), lineterminator="\n")
    w.writeheader()
    for e in LEDGER:
        w.writerow({k: flat(v) if v is not None else "" for k, v in e.items()})

LAYERS = {1: "Are our scores real?", 2: "Does practice teach new recipes?", 3: "Why only sharpening?",
          4: "Does the checker match a real lab?", 5: "Where did the rule-picked recipes go?",
          6: "Practice from RS-SFT", 7: "Better judges", 8: "What practice changed",
          9: "Contamination", 10: "Phase 16"}
md = ["# Claims ledger", "",
      "Generated by `scripts/build_claims_ledger.py`. Do not edit by hand; rerun the script.", "",
      "Status: **RECOMPUTED** = computed by the script from a tracked results file. **RECOMPUTED\\*** = recomputed, "
      "and an earlier statement of it was wrong or ambiguous (see note). **FROM_RESULTS** = a derived statistic "
      "copied from a tracked results file. **DOC_ONLY** / **CHAT_ONLY** = the raw data is not tracked.", "",
      "\"Points\" in any pass@k statement are percentage points of targets solved, never reward.", ""]
for L in sorted(LAYERS):
    rows = [e for e in LEDGER if e["layer"] == L]
    if not rows:
        continue
    md += [f"## Layer {L}: {LAYERS[L]}", ""]
    for e in rows:
        md += [f"### `{e['id']}` {e['claim']}", "",
               f"- **Status:** {e['status']}",
               f"- **Source:** {e['source']}"]
        if e["definition"]:
            md.append(f"- **Definition:** {e['definition']}")
        if e["stated_earlier"]:
            md.append(f"- **Stated earlier:** {e['stated_earlier']}")
        if e["note"]:
            md.append(f"- **Note:** {e['note']}")
        md += ["", "```json", json.dumps(e["value"], indent=1, ensure_ascii=False), "```", ""]
(OUT / "CLAIMS_LEDGER.md").write_text("\n".join(md))
print(f"wrote {len(LEDGER)} entries to {OUT}")
