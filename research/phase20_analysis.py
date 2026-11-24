#!/usr/bin/env python
"""
research/phase20_analysis.py -- every Phase 20 analysis (docs/phases/PHASE20_PREREG.md,
"Predictions" and "Measurements"). Each subcommand first asserts the spec is
committed (research/phase20_common.provenance(require_commit=True)) and writes
results/phase20/<name>.json (new files only). Old code is reused unchanged and
redirected (step tables, recipe cache, outputs); nothing old is rewritten.

Subcommands (CPU unless noted):
  transport  --stage 2b|final  P-20-1 + 18A rerun: research/phase18_a_transport_exact.py
             with the Phase 20 step table (S2 Base->sft-rssft-full, S3 Base->sft-e3-full,
             S4 Base->sft-e3c-full; S7 sft-rssft-full->E1-full@300 at final; S1, S5, S6 old).
  posterior  --stage 2b|final  18E rerun (research/phase18_e_posterior_matching.py),
             new models in place of the old SFT models (and E1 @300 at final).
  support                      P-20-2: research/support_growth_by_checkpoint.py (finding 33's
             procedure) with sft-rssft-full's reference as the reference and floor.
  reach      --stage 2b|final  P-20-3 (registered; Phase 19 exploratory reach made exact).
  location                     P-20-4 / P-20-5 read from results/phase20/p19_4/.
  coverage                     probe-set coverage, new vs the old model replaced.
  mmd_rssft                    old vs new RS-SFT per-target recipe distributions, 18E kernel MMD.
  hits                         ASTRAL rule-picked hits (finding 18/22 metric) per new model.
  alkali                       alkali-source shares (finding 24's per-alkali-choice wording rule).
  uniform    --stage 2b|final  does the old SFT +7-8 nat uniform shift appear for the new SFT?
  weights                      18C rerun (E64, effective rank) on the new adapters.
  tempfix                      temperature sensitivity: mixed comparisons with Stage 5 resamples.
  temp_vllm_e                  [E] Gowtham 2026-11-17: Base / old RS-SFT comparisons recomputed
             with their existing T 0.9 vLLM files (no new sampling).

Usage: PYTHONHASHSEED=0 PYTHONPATH=. uv run python research/phase20_analysis.py <cmd> [--stage ..]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))
sys.path.insert(0, str(REPO_ROOT / "research" / "distributional"))

import numpy as np  # noqa: E402

from research.phase20_common import (OUT_DIR, N_BOOT, SEED, NEW_SOURCES, T09_SOURCES,  # noqa: E402
                                     load_recipes20, notify, patch_phase18, provenance,
                                     write_json)

P19 = REPO_ROOT / "results" / "phase19"
NEW_SFT = ["sft_rssft_full", "sft_e3_full", "sft_e3c_full"]


def final_e1_ready() -> bool:
    return (OUT_DIR / "p19_4" / "dimension.json").exists() and NEW_SOURCES[("e1full_ckpt300", "astral")].exists()


def step_table(stage: str) -> dict:
    cache_has_corpus_full = NEW_SOURCES[("sft_corpus_full", "astral")].exists()
    s1_after = "sft_corpus_full" if (stage == "final" and cache_has_corpus_full) else "corpus_sft"
    return {
        "S1": ("base", s1_after),
        "S2": ("base", "sft_rssft_full"),
        "S3": ("base", "sft_e3_full"),
        "S4": ("base", "sft_e3c_full"),
        "S5": ("corpus_sft", "gdpo_v3_ckpt300"),
        "S6": ("rs_sft", "phase12_ckpt300"),
        "S7": ("sft_rssft_full", "e1full_ckpt300") if stage == "final" else ("rs_sft", "e1_ckpt300"),
    }


# ---------------------------------------------------------------- P-20-1 / 18A
def cmd_transport(stage: str):
    steps = step_table(stage)
    patch_phase18(steps, out_prefix=f"a18_{stage}_")
    import research.phase18_a_transport_exact as a
    a.main()
    doc = json.loads((OUT_DIR / f"a18_{stage}_transport_exact.json").read_text())
    prim = doc.get("primary") or doc.get("results_primary") or doc
    p2 = _find_key(prim, "P_18A_2")
    rows = p2.get("by_step", p2) if isinstance(p2, dict) else {}
    verdict = {}
    for s, m in (("S2", "sft_rssft_full"), ("S3", "sft_e3_full"), ("S4", "sft_e3c_full")):
        r = rows.get(s, {})
        ci = r.get("ci95") or r.get("ci")
        verdict[m] = {"step": s, "new_support_mass": r.get("new_support_mass"), "noise_floor": r.get("noise_floor"),
                      "diff": r.get("diff"), "ci95": ci,
                      "positive_with_ci_above_0": bool(ci and ci[0] > 0)}
    res = {**provenance(), "prediction": "P-20-1",
           "statement": "New-support mass minus Base's resampling floor (P-18A-2 measure, same code, "
                        "lambda, floor and targets) is positive with CI lower bound above 0 for each of "
                        "sft-rssft-full, sft-e3-full, sft-e3c-full.",
           "stage": stage, "step_table": steps, "by_model": verdict,
           "sft_rssft_full_samples": load_recipes20().get("sft_rssft_full", {}).get("astral", {}).get("source"),
           "passes": all(v["positive_with_ci_above_0"] for v in verdict.values()),
           "source": f"results/phase20/a18_{stage}_transport_exact.json (pool: primary, ASTRAL clean 22 + held-out 40)",
           "raw_P_18A_2_primary": p2}
    write_json(f"p20_1_{stage}.json", res)
    print(json.dumps({k: res[k] for k in ("by_model", "passes")}, indent=1, default=str))


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


# ---------------------------------------------------------------- 18E rerun
ALIAS_2B = {"rs_sft": "sft_rssft_full", "e3": "sft_e3_full", "e3c": "sft_e3c_full"}


def aliased_loader(alias: dict):
    def load():
        c = load_recipes20()
        for old, new in alias.items():
            if new in c:
                c[old] = c[new]
        return c
    return load


def cmd_posterior(stage: str):
    alias = dict(ALIAS_2B)
    if stage == "final":
        alias["e1_ckpt300"] = "e1full_ckpt300"
        if NEW_SOURCES[("sft_corpus_full", "astral")].exists():
            alias["corpus_sft"] = "sft_corpus_full"
    c = patch_phase18(step_table(stage), out_prefix=f"e18_{stage}_")
    c.load_recipes = aliased_loader(alias)
    import research.phase18_e_posterior_matching as e
    e.load_recipes = c.load_recipes
    e.write_json = c.write_json
    e.provenance = c.provenance
    e.main()
    p = OUT_DIR / f"e18_{stage}_posterior_matching.json"
    doc = json.loads(p.read_text())
    doc["phase20_aliases"] = {old: f"{new} (Phase 20 model, under the old label)" for old, new in alias.items()}
    p.write_text(json.dumps(doc, indent=1, default=str))
    print("aliases:", alias)


# ---------------------------------------------------------------- P-20-2
def cmd_support():
    import support_growth_by_checkpoint as sg
    P20 = REPO_ROOT / "data" / "novelty" / "phase20"
    ref = {ts: P20 / f"sft_rssft_full_reference_{ts}.jsonl" for ts in ("astral", "heldout")}
    sg.RSSFT_PATHS = ref
    sg.SOURCES = {("sft_rssft_full", ts): ("raw_reference", ref[ts]) for ts in ("astral", "heldout")}
    for c in (100, 200, 300):
        for ts in ("astral", "heldout"):
            sg.SOURCES[(f"e1full_ckpt{c}", ts)] = ("raw", P20 / f"e1full_ckpt{c}_{ts}_n32.jsonl")
    sg.CHECKPOINT_ORDER = ["sft_rssft_full", "e1full_ckpt100", "e1full_ckpt200", "e1full_ckpt300"]
    out_tmp = OUT_DIR / "support_growth_by_checkpoint_e1full.json"
    sg.OUT_PATH = out_tmp
    sg.main()
    rep = json.loads(out_tmp.read_text())
    bc = rep["by_checkpoint"]
    verdict = {}
    for ts in ("astral", "heldout"):
        floor = bc["sft_rssft_full"][ts]
        for c in (100, 200, 300):
            r = bc[f"e1full_ckpt{c}"][ts]
            above_point = r["point"] > floor["point"]
            ci_above_floor_ci = r["ci95"][0] > floor["ci95"][1]
            verdict[f"{ts}_ckpt{c}"] = {"point": r["point"], "ci95": r["ci95"],
                                        "floor_point": floor["point"], "floor_ci95": floor["ci95"],
                                        "point_above_floor": above_point,
                                        "intervals_overlap": not ci_above_floor_ci,
                                        "exceeds_floor": ci_above_floor_ci}
    res = {**provenance(), "prediction": "P-20-2",
           "statement": "E1-full at steps 100, 200 and 300 does not exceed sft-rssft-full's resampling floor "
                        "for new recipes (finding 33's procedure).",
           "operationalisation": ("finding 33's procedure (support_growth_by_checkpoint.py): pooled share of "
                                  "distinct recipes absent from the init's 200-sample reference, cluster-bootstrap "
                                  "CI. A checkpoint 'exceeds the floor' only if its CI lies entirely above the "
                                  "floor's CI (finding 33: 'every interval overlaps the floor's'); the point "
                                  "comparison is reported alongside."),
           "by_checkpoint": verdict,
           "passes": not any(v["exceeds_floor"] for v in verdict.values()),
           "hits_and_carbonate": {k: v.get("hits_and_carbonate") for k, v in bc.items()},
           "raw": "results/phase20/support_growth_by_checkpoint_e1full.json"}
    write_json("p20_2.json", res)
    print(json.dumps({k: res[k] for k in ("by_checkpoint", "passes")}, indent=1, default=str))


# ---------------------------------------------------------------- P-20-3
def _boot_median(vals, seed):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    idx = rng.integers(0, len(v), (N_BOOT, len(v)))
    bs = np.median(v[idx], axis=1)
    return float(np.median(v)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def reach_step(scores, rows_by_t, before, after):
    per_t = {}
    for t, rows in rows_by_t.items():
        d = []
        for r in rows:
            lb = scores.get((before, t, r["recipe_key"]))
            la = scores.get((after, t, r["recipe_key"]))
            if lb is None or la is None:
                break
            d.append((la - lb, r["counts"][before] > 0))
        else:
            mu = sum(x for x, _ in d) / len(d)
            uns = [x - mu for x, s in d if not s]
            smp = [x - mu for x, s in d if s]
            if len(uns) >= 3 and len(smp) >= 1:
                per_t[t] = {"diff": sum(uns) / len(uns) - sum(smp) / len(smp),
                            "n_unsampled": len(uns), "n_sampled": len(smp)}
    return per_t


def cmd_reach(stage: str):
    d = OUT_DIR / ("p19_2b" if stage == "2b" else "p19_4")
    scores = {}
    for line in (d / "string_scores.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            scores[(r["model"], r["target"], r["recipe_key"])] = r["l"]
    rows_by_t = defaultdict(list)
    for line in (d / "probe_strings_with_counts.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows_by_t[r["target"]].append(r)
    srcs = json.loads((d / "count_sources.json").read_text())
    steps = {"base->sft_e3_full": ("base", "sft_e3_full", ">0"),
             "base->sft_e3c_full": ("base", "sft_e3c_full", ">0")}
    if stage == "final":
        steps["sft_rssft_full->e1full_ckpt300"] = ("sft_rssft_full", "e1full_ckpt300", "<0")
        if "reference" not in srcs.get("sft_rssft_full", ""):
            raise SystemExit("sft_rssft_full counts are not from its 232-sample reference")
    res = {}
    for name, (b, a, want) in steps.items():
        per_t = reach_step(scores, rows_by_t, b, a)
        med, ci = _boot_median([v["diff"] for v in per_t.values()], SEED)
        ok = (ci[1] < 0) if want == "<0" else (ci[0] > 0)
        res[name] = {"median": med, "ci95": ci, "n_usable_targets": len(per_t),
                     "predicted": "CI entirely below 0" if want == "<0" else "CI entirely above 0",
                     "holds": ok, "per_target": per_t}
        print(f"  {name}: median {med:+.3f} CI {ci} usable={len(per_t)} holds={ok}", flush=True)
    g = json.loads((d / "gate.json").read_text())
    expl = {m: not v["passes_0.4"] for m, v in g["by_model"].items()}
    doc = {**provenance(), "prediction": "P-20-3", "stage": stage,
           "definition": ("Delta = l_after - l_before (Phase 19 G/E exact string scores), centred within target; "
                          "'sampled' = before-model's 232-sample reference contains the recipe; usable target: "
                          ">=3 unsampled and >=1 sampled strings; per target mean centred Delta(unsampled) - "
                          "mean(sampled); statistic = median over usable targets; 10,000-draw cluster "
                          "bootstrap over targets, seed 20261116."),
           "by_step": res,
           "passes": all(v["holds"] for v in res.values()) if stage == "final" else None,
           "gate_G_failed_models_label_exploratory": [m for m, x in expl.items() if x],
           "count_sources": srcs}
    write_json(f"p20_3_{stage}.json", doc)


# ---------------------------------------------------------------- P-20-4 / P-20-5
def cmd_location():
    d = OUT_DIR / "p19_4"
    loc = json.loads((d / "location.json").read_text())
    dim = json.loads((d / "dimension.json").read_text())
    g = json.loads((d / "gate.json").read_text())
    L = {s: loc["by_step"][s]["L50"] for s in ("S3", "S4", "S7")}
    big = lambda x: 99 if x is None else x  # noqa: E731
    p4 = {"S7>S3": big(L["S7"]) > big(L["S3"]), "S7>S4": big(L["S7"]) > big(L["S4"])}
    R4 = {s: dim["by_step"][s]["R"]["4"] for s in ("S3", "S4", "S7")}
    p5 = {"S3>S7": R4["S3"] > R4["S7"], "S4>S7": R4["S4"] > R4["S7"]}
    expl = [m for m, v in g["by_model"].items() if not v["passes_0.4"]]
    doc = {**provenance(), "steps": {"S3": "Base -> sft-e3-full", "S4": "Base -> sft-e3c-full",
                                     "S7": "sft-rssft-full -> E1-full @300"},
           "P_20_4": {"statement": "L50(sft-rssft-full -> E1-full@300) > L50(Base -> sft-e3-full) and > "
                                   "L50(Base -> sft-e3c-full); Block E code unchanged.",
                      "L50": L, "L50_ci95": {s: loc["by_step"][s]["L50_ci95"] for s in L},
                      "orderings": p4, "passes": all(p4.values())},
           "P_20_5": {"statement": "R(4) for each SFT step (S3, S4) > R(4) for sft-rssft-full -> E1-full@300; "
                                   "F1 code unchanged, L* from this phase's Block E.",
                      "R4": R4, "L_star": {s: dim["by_step"][s]["L_star"] for s in R4},
                      "orderings": p5, "passes": all(p5.values())},
           "gate_G_failed_models_label_exploratory": expl,
           "sources": ["results/phase20/p19_4/location.json", "results/phase20/p19_4/dimension.json"]}
    write_json("p20_4_5.json", doc)
    print(json.dumps({"P_20_4": doc["P_20_4"], "P_20_5": doc["P_20_5"]}, indent=1, default=str))


# ---------------------------------------------------------------- measurements
def cmd_coverage():
    from research.phase18_common import recipes
    cache = load_recipes20()
    probe = defaultdict(set)
    for line in (P19 / "probe_strings.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            probe[(r["target"], r["target_set"])].add(tuple(r["recipe"]))
    pairs = [("sft_rssft_full", "rs_sft"), ("sft_e3_full", "e3"), ("sft_e3c_full", "e3c"),
             ("e1full_ckpt300", "e1_ckpt300"), ("sft_corpus_full", "corpus_sft")]
    out = {}
    for new, old in pairs:
        row = {}
        for m in (new, old):
            if m not in cache:
                row[m] = None
                continue
            tot = cov = 0
            for (t, ts), ys in probe.items():
                for y in recipes(cache, m, ts, t):
                    tot += 1
                    cov += tuple(y) in ys
            row[m] = {"share_in_probe_set": cov / tot if tot else None, "n_samples": tot}
        out[f"{new}_vs_{old}"] = row
    write_json(f"coverage_{STAGE}.json", {**provenance(), "measurement": "probe-set coverage",
                                 "definition": "share of the model's parsed samples on the 40 probe targets whose "
                                               "canonical recipe is one of Phase 19's 671 probe recipes",
                                 "by_pair": out})
    print(json.dumps(out, indent=1))


def _mmd(p_samples, q_samples):
    from research.phase18_common import jaccard_distance
    ys = sorted(set(p_samples) | set(q_samples))
    K = np.array([[1 - jaccard_distance(a, b) for b in ys] for a in ys])
    cp, cq = Counter(p_samples), Counter(q_samples)
    p = np.array([cp[y] / len(p_samples) for y in ys])
    q = np.array([cq[y] / len(q_samples) for y in ys])
    d = p - q
    return float(np.sqrt(max(0.0, d @ K @ d)))


def cmd_mmd_rssft():
    from research.phase18_common import recipes, target_sets
    cache = load_recipes20()
    sets = target_sets()
    rng = random.Random(SEED)
    out = {}
    for label, ts, targets in (("astral_35", "astral", sets["astral_35"]), ("heldout_40", "heldout", sets["heldout_40"])):
        per = {}
        for t in targets:
            old = [tuple(y) for y in recipes(cache, "rs_sft", ts, t, group="reference")]
            new = [tuple(y) for y in recipes(cache, "sft_rssft_full", ts, t, group="reference")]
            o1 = [tuple(y) for y in recipes(cache, "rs_sft", ts, t, group="resampling_rate")]
            if len(old) < 2 or len(new) < 2:
                continue
            per[t] = {"mmd_old_vs_new": _mmd(old, new), "mmd_old_ref_vs_old_resample_floor": _mmd(old, o1) if o1 else None,
                      "n_old": len(old), "n_new": len(new)}
        ts_list = sorted(per)
        vals = [per[t]["mmd_old_vs_new"] for t in ts_list]
        fl = [per[t]["mmd_old_ref_vs_old_resample_floor"] for t in ts_list if per[t]["mmd_old_ref_vs_old_resample_floor"] is not None]
        bs = []
        for _ in range(N_BOOT):
            s = [vals[rng.randrange(len(vals))] for _ in vals]
            bs.append(sum(s) / len(s))
        out[label] = {"mean_mmd_old_vs_new": sum(vals) / len(vals), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      "mean_floor_old_200_vs_old_32": sum(fl) / len(fl) if fl else None,
                      "n_targets": len(vals), "per_target": per}
    write_json("mmd_rssft_old_vs_new.json", {**provenance(), "measurement": "old vs new RS-SFT, 18E kernel MMD",
                                            "kernel": "k(y,y') = 1 - Jaccard distance (18E)",
                                            "samples": "each model's 200-sample 'reference' group (vLLM T 0.9); floor = "
                                                       "old RS-SFT's 200 vs its own 32 resampling draw [E]",
                                            "by_set": out})
    print({k: (v["mean_mmd_old_vs_new"], v["ci95"], v["mean_floor_old_200_vs_old_32"]) for k, v in out.items()})


def cmd_hits():
    from research.phase18_common import astral_lab, recipes, target_sets
    cache = load_recipes20()
    lab, sets = astral_lab(), target_sets()
    clean = set(sets["astral_clean_22"])
    models = [("sft_rssft_full_n32", None), ("sft_e3_full", None), ("sft_e3c_full", None),
              ("e1full_ckpt100", None), ("e1full_ckpt200", None), ("e1full_ckpt300", None),
              ("sft_corpus_full", None), ("rs_sft", "resampling_rate"), ("base", "resampling_rate"),
              ("e3", None), ("e3c", None)]
    out = {}
    for m, grp in models:
        if m not in cache or "astral" not in cache[m] or "samples" not in cache[m]["astral"]:
            continue
        ex, sup = {}, {}
        for t in sets["astral_35"]:
            ys = [set(y) for y in recipes(cache, m, "astral", t, group=grp)]
            pred = set(lab[t]["phase16_hit_set"])
            ex[t] = any(y == pred for y in ys)
            sup[t] = any(pred <= y for y in ys)
        out[m] = {"exact_full": f"{sum(ex.values())}/35", "exact_clean": f"{sum(ex[t] for t in clean)}/22",
                  "superset_incl_full": f"{sum(sup.values())}/35",
                  "superset_incl_clean": f"{sum(sup[t] for t in clean)}/22",
                  "group": grp or "all (32/target)", "hit_targets_exact": sorted(t for t, v in ex.items() if v)}
    write_json(f"astral_hits_{STAGE}.json", {**provenance(), "measurement": "ASTRAL rule-picked hits (finding 18/22 metric)",
                                    "definition": "hit = any of the model's 32 samples has exactly the ASTRAL rule-picked "
                                                  "precursor set (canonical); superset-inclusive variant reported. All "
                                                  "rows are vLLM T 0.9 draws (Base, old RS-SFT: their 32-sample "
                                                  "resampling group).", "by_model": out})
    print(json.dumps({m: {k: v[k] for k in ("exact_full", "exact_clean")} for m, v in out.items()}, indent=1))


def _astral_dump(cache, m, grp=None) -> Path:
    from research.phase18_common import recipes, target_sets
    res = [{"target": t, "samples": [{"precursors": list(y)} for y in recipes(cache, m, "astral", t, group=grp)]}
           for t in target_sets()["astral_35"]]
    p = OUT_DIR / f"_astral_dump_{m}{'_' + grp if grp else ''}.json"
    p.write_text(json.dumps({"results": res}))
    return p


def cmd_alkali():
    from alkali_source_share import tally_astral_dump
    cache = load_recipes20()
    out = {}
    for m, grp in (("sft_rssft_full_n32", None), ("e1full_ckpt300", None), ("rs_sft", "resampling_rate"),
                   ("e1_ckpt300_t09", None), ("sft_e3_full", None), ("sft_e3c_full", None)):
        if m not in cache or "astral" not in cache[m]:
            continue
        out[m] = tally_astral_dump(_astral_dump(cache, m, grp))
    write_json(f"alkali_source_share_{STAGE}.json", {**provenance(), "measurement": "alkali-source shares, ASTRAL 35",
                                            "wording_rule": "finding 24: per-alkali-choice shares (each alkali element "
                                                            "a sample must source counts once); not whole-sample shares",
                                            "by_model": out})
    print(json.dumps(out, indent=1, default=str)[:2000])


def cmd_uniform(stage: str):
    d = OUT_DIR / ("p19_2b" if stage == "2b" else "p19_4")
    loc = json.loads((d / "location.json").read_text())
    old = json.loads((P19 / "location.json").read_text())
    out = {s: {"new": {k: loc["by_step"][s][k] for k in ("before", "after", "delta_mean", "delta_rms")},
               "old_phase19": {k: old["by_step"][s][k] for k in ("before", "after", "delta_mean", "delta_rms")}
               if s in old["by_step"] else None}
           for s in loc["by_step"]}
    for s, v in out.items():
        n = v["new"]
        v["new"]["mean_over_rms"] = n["delta_mean"] / n["delta_rms"] if n["delta_rms"] else None
    write_json(f"uniform_shift_{stage}.json", {**provenance(), "measurement": "uniform shift on answer strings",
                                              "definition": "Block E before/after: mean and rms of l_after - l_before over "
                                                            "the 671 probe strings; old SFT: mean +7.2 to +7.6, mean ~ rms",
                                              "by_step": out})
    print(json.dumps(out, indent=1))


def cmd_weights():
    import research.phase18_common as c18
    c18.ADAPTERS = {**c18.ADAPTERS}
    import research.phase20_common as c20
    prov = provenance()
    import research.phase18_c_weight_geometry as w
    w.ADAPTERS = {**w.ADAPTERS, "rs_sft": c20.NEW_ADAPTERS["sft_rssft_full"],
                  "e3": c20.NEW_ADAPTERS["sft_e3_full"], "e3c": c20.NEW_ADAPTERS["sft_e3c_full"]}
    def last_ckpt(run):
        cs = sorted((REPO_ROOT / run).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        return str(cs[-1].relative_to(REPO_ROOT))
    w.FT_INTERMEDIATE = {**w.FT_INTERMEDIATE,
                         "rs_sft": last_ckpt("runs/sft-qlora-sft-rssft-full"),
                         "e3": last_ckpt("runs/sft-qlora-sft-e3-full"),
                         "e3c": last_ckpt("runs/sft-qlora-sft-e3c-full")}
    # S6 (old RS-SFT -> Phase 12) stays as an unchanged reference step on its own
    # (old) init, so the 18C code and its P-18C-2 test run as written.
    w.ADAPTERS["rs_sft_old"] = c18.ADAPTERS["rs_sft"]
    w.TRAJ = {"S5": w.TRAJ["S5"], "S6": ("rs_sft_old", w.TRAJ["S6"][1]),
              "S7": ("rs_sft", [c20.NEW_ADAPTERS[f"e1full_ckpt{k}"] for k in (100, 200, 300)])}
    w.STEPS = {**step_table("final"), "S6": ("rs_sft_old", "phase12_ckpt300")}
    w.PRACTICE_STEPS = ("S5", "S6", "S7")
    w.provenance = lambda: {**prov, "phase20_note": "18C rerun: rs_sft/e3/e3c = sft-*-full adapters; S7 = "
                                                     "sft-rssft-full -> E1-full; S6 = old RS-SFT -> Phase 12 "
                                                     "(old init, unchanged reference)"}
    w.write_json = lambda name, obj: write_json("c18_" + name, obj)
    w.main()


def cmd_tempfix():
    """Recompute each directly mixed comparison with the Stage 5 resamples in place
    of the four HF T=1.0 ASTRAL files; old and new side by side."""
    from research.phase18_common import astral_lab, recipes, target_sets
    cache = load_recipes20()
    lab, sets = astral_lab(), target_sets()
    out = {"hits": {}, "alkali": {}}
    from alkali_source_share import tally_astral_dump
    old_files = {"corpus_sft": "astral_gen_n32_sft.json", "gdpo_v3_ckpt300": "astral_gen_n32_gdpo300.json",
                 "phase12_ckpt300": "astral_gen_n32_gdpo_phase12_beta0.json", "e1_ckpt300": "astral_gen_n32_e1.json"}
    for m in T09_SOURCES:
        row = {}
        for tag, key in (("old_hf_t1.0", m), ("new_vllm_t0.9", f"{m}_t09")):
            if key not in cache:
                row[tag] = None
                continue
            ex = {t: any(set(y) == set(lab[t]["phase16_hit_set"]) for y in recipes(cache, key, "astral", t))
                  for t in sets["astral_35"]}
            row[tag] = {"exact_full": f"{sum(ex.values())}/35",
                        "exact_clean": f"{sum(ex[t] for t in sets['astral_clean_22'])}/22"}
        out["hits"][m] = row
        out["alkali"][m] = {"old_hf_t1.0": tally_astral_dump(REPO_ROOT / "results" / old_files[m]),
                            "new_vllm_t0.9": tally_astral_dump(_astral_dump(cache, f"{m}_t09"))
                            if f"{m}_t09" in cache else None}
    # 18A / 18E with the four models' ASTRAL entries replaced by their T 0.9 resamples
    for prefix, mod in (("a18", "research.phase18_a_transport_exact"), ("e18", "research.phase18_e_posterior_matching")):
        out[f"{prefix}_rerun"] = f"results/phase20/{prefix}_tempfix_*.json (old: results/phase18/)"
    write_json("tempfix_hits_alkali.json", {**provenance(), "measurement": "temperature sensitivity (hits, alkali)", **out})
    print(json.dumps(out["hits"], indent=1))


def cmd_tempfix_blocks():
    """18A and 18E rerun with the four HF ASTRAL entries replaced by Stage 5 resamples
    (old step table, old models otherwise) -- the 'directly mixed' 18A/18E rows."""
    def load():
        c = json.loads(json.dumps(load_recipes20()))
        for m in T09_SOURCES:
            if f"{m}_t09" in c:
                c[m] = {**c[m], "astral": c[f"{m}_t09"]["astral"]}
        return c
    from research.phase18_common import STEPS as OLD_STEPS
    c = patch_phase18(dict(OLD_STEPS), out_prefix="tempfix_")
    c.load_recipes = load
    import research.phase18_a_transport_exact as a
    a.load_recipes = load
    a.main()
    import research.phase18_e_posterior_matching as e
    e.load_recipes, e.write_json, e.provenance = load, c.write_json, c.provenance
    e.main()


def cmd_temp_vllm_e():
    """[E] (Gowtham, 2026-11-17): comparisons that set Base or old RS-SFT (HF T=1.0 ASTRAL
    files) against vLLM T=0.9 models, recomputed with Base's and old RS-SFT's existing T 0.9
    vLLM files (base_reference_astral, rssft_reference_astral; their 32-sample
    'resampling_rate' group, so n = 32 like every other row). No new sampling."""
    from research.phase18_common import astral_lab, recipes, target_sets
    from alkali_source_share import tally_astral_dump
    cache = load_recipes20()
    lab, sets = astral_lab(), target_sets()
    clean = sets["astral_clean_22"]

    def hits(m, grp=None, src=None):
        if src is not None:
            doc = json.loads((REPO_ROOT / "results" / src).read_text())
            by = {r["target"]: [set(s.get("precursors") or []) for s in r["samples"]] for r in doc["results"]}
        else:
            by = {t: [set(y) for y in recipes(cache, m, "astral", t, group=grp)] for t in sets["astral_35"]}
        ex = {t: any(y == set(lab[t]["phase16_hit_set"]) for y in by.get(t, [])) for t in sets["astral_35"]}
        return {"exact_full": f"{sum(ex.values())}/35", "exact_clean": f"{sum(ex[t] for t in clean)}/22",
                "hit_targets": sorted(t for t, v in ex.items() if v)}

    f31 = {"E3 (vLLM 0.9, unchanged)": hits("e3"), "E3c (vLLM 0.9, unchanged)": hits("e3c"),
           "Base old: astral_gen_n32_base.json (HF 1.0)": hits(None, src="astral_gen_n32_base.json"),
           "Base new: base_reference_astral resampling_rate (vLLM 0.9)": hits("base", "resampling_rate"),
           "Base new: base_reference_astral reference 200 (vLLM 0.9), any of 200": hits("base", "reference"),
           "RS-SFT old: astral_gen_n32_rs_sft.json (HF 1.0)": hits(None, src="astral_gen_n32_rs_sft.json"),
           "RS-SFT new: rssft_reference_astral resampling_rate (vLLM 0.9)": hits("rs_sft", "resampling_rate")}
    rs_alk = {"old_hf_t1.0 (astral_gen_n32_rs_sft.json)": tally_astral_dump(REPO_ROOT / "results" / "astral_gen_n32_rs_sft.json"),
              "new_vllm_t0.9 (rssft_reference_astral resampling_rate)": tally_astral_dump(_astral_dump(cache, "rs_sft", "resampling_rate"))}
    f33 = json.loads((REPO_ROOT / "results" / "support_growth_by_checkpoint.json").read_text())["by_checkpoint"]["rs_sft"]
    doc = {**provenance(), "label": "[E]", "decided": "2026-11-17 by Gowtham, after the prereg commit",
           "finding_31_hits": f31,
           "finding_33_rssft_row": {"as_reported (computed 2026-10-15 from the vLLM rebuild)": f33.get("hits_and_carbonate"),
                                    "alkali_per_choice_hf_vs_vllm": rs_alk,
                                    "rssft_vllm_hits": f31["RS-SFT new: rssft_reference_astral resampling_rate (vLLM 0.9)"]},
           "a18_e18_note": ("18A/18E already read Base and RS-SFT from their vLLM reference files; their mixing is "
                            "via the four HF models, recomputed in tempfix_* with the Stage 5 resamples.")}
    write_json("temp_vllm_E.json", doc)
    print(json.dumps(f31, indent=1, default=str))


def cmd_report():
    """Collect every registered verdict and measurement into results/phase20/summary.json
    (the dated PHASE20_RESULTS.md entry is written from it)."""
    pick = {
        "P-20-1": ("p20_1_final.json", ("by_model", "passes")),
        "P-20-2": ("p20_2.json", ("by_checkpoint", "passes")),
        "P-20-3": ("p20_3_final.json", ("by_step", "passes", "gate_G_failed_models_label_exploratory")),
        "P-20-4/5": ("p20_4_5.json", ("P_20_4", "P_20_5", "gate_G_failed_models_label_exploratory")),
        "gate_G_stage4": ("p19_4/gate.json", ("gate_passes", "by_model")),
        "coverage": ("coverage_final.json", ("by_pair",)),
        "mmd_rssft": ("mmd_rssft_old_vs_new.json", ("by_set",)),
        "hits": ("astral_hits_final.json", ("by_model",)),
        "alkali": ("alkali_source_share_final.json", ("by_model",)),
        "uniform": ("uniform_shift_final.json", ("by_step",)),
        "tempfix": ("tempfix_hits_alkali.json", ("hits", "alkali")),
        "temp_vllm_E": ("temp_vllm_E.json", ("finding_31_hits", "finding_33_rssft_row")),
    }
    out = {}
    for k, (f, keys) in pick.items():
        p = OUT_DIR / f
        if not p.exists():
            out[k] = {"missing": f}
            continue
        d = json.loads(p.read_text())
        out[k] = {kk: d.get(kk) for kk in keys}
        for kk in ("by_model", "by_step"):
            if isinstance(out[k].get(kk), dict):
                for v in out[k][kk].values():
                    if isinstance(v, dict):
                        v.pop("per_target", None)
    write_json("summary.json", {**provenance(), **out})
    print(json.dumps({k: (v.get("passes") if isinstance(v, dict) else None) for k, v in out.items()}, indent=1))


STAGE = "final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--stage", default="final", choices=["2b", "final"])
    a = ap.parse_args()
    global STAGE
    STAGE = a.stage  # suffix for outputs of commands without a --stage of their own
    provenance(require_commit=True)
    fn = globals()[f"cmd_{a.cmd}"]
    import inspect
    if "stage" in inspect.signature(fn).parameters:
        fn(a.stage)
    else:
        fn()
    notify(f"MIRA phase20 analysis {a.cmd} ({a.stage}) done")


if __name__ == "__main__":
    main()
