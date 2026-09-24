#!/usr/bin/env python
"""
research/distributional/dose_response_robustness.py — Phase 15 Task 5
robustness round, requested on review of PHASE15_DOSE_RESULTS.md. All
[E]. Reads results/dose_response/teacher_forcing_raw.json (already
committed); does not touch the GPU except for the new ASTRAL-prompt
rank/logp check (part 4), which loads base/RS-SFT/GDPO-300 for four
named targets, no sampling.

Parts:
1. Q4 target control: per-target carbonate counts (narrow, standing
   definition), Q4 regression with target fixed effects, does
   carbonate_status survive.
2. Q4 broader carbonate definition (any precursor containing the "CO3"
   substring, catching CuCO3 / Y2(CO3)3 which the standing alkali/
   alkaline-earth-only set excludes): how many routes reclassify, does
   the coefficient survive, with and without target FE.
3. Q4 length covariate: add precursor-segment token count (base
   tokenizer, already in the raw output) to the regression; check
   whether carbonate routes are longer on average, to establish the
   "conservative with respect to length" direction claimed in the doc.
4. Original Q3 on ASTRAL [E]: for 4 named ASTRAL targets, prompt-only
   (no teacher-forced route), report rank + log p of the ASTRAL-
   predicted first precursor's first token and the traditional first
   precursor's first token, under base / RS-SFT / GDPO-300. Same
   rendering as Task 5 (dose_response_teacher_forcing.py).

Usage (tmux, GPU for part 4 only):
  uv run python research/distributional/dose_response_robustness.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW_PATH = Path("results/dose_response/teacher_forcing_raw.json")
ASTRAL_PATH = Path("results/astral_validation_set.json")
OUT_JSON = Path("results/dose_response/robustness.json")

SEED = 20260925
N_BOOT = 10000
CARBONATES_NARROW = {"Li2CO3", "Na2CO3", "K2CO3", "BaCO3", "SrCO3", "CaCO3", "MgCO3"}

ASTRAL_TARGETS = ["NaSrBO3", "BaLiBO3", "KLi(PO3)2", "Li2TiSiO5"]


def ols_coeffs(rows, y_key, x_keys):
    import numpy as np
    X = np.array([[1.0] + [float(r[k]) for k in x_keys] for r in rows])
    y = np.array([float(r[y_key]) for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = {"intercept": beta[0]}
    for i, k in enumerate(x_keys):
        out[k] = beta[i + 1]
    return out


def cluster_bootstrap_ci(rows, stat_fn, rng, n_boot=N_BOOT):
    n = len(rows)
    boot_stats = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        boot_stats.append(stat_fn(sample))
    boot_stats.sort()
    lo = boot_stats[int(0.025 * n_boot)]
    hi = boot_stats[int(0.975 * n_boot) - 1]
    return (lo, hi)


def build_rows():
    raw = json.loads(RAW_PATH.read_text())
    rows = []
    for r in raw:
        route = r["route"]
        ck = r["checkpoints"]
        precs = route["precursors"]
        rows.append({
            "precursor_set": route["precursor_set"],
            "target": route["arrows_target"],
            "has_carbonate_narrow": route["has_carbonate"],
            "has_carbonate_broad": any("CO3" in p for p in precs),
            "best_wt_pct": route["best_wt_pct"],
            "logp_rs_sft": ck["rs_sft"]["total_logp"],
            "logp_gdpo": ck["gdpo_300"]["total_logp"],
            "n_tokens_base": ck["base"]["n_tokens"],
        })
        rows[-1]["delta_gdpo_rssft"] = rows[-1]["logp_gdpo"] - rows[-1]["logp_rs_sft"]
    return rows


def part1_target_control(rows, rng):
    print("\n=== Part 1: Q4 target control ===")
    per_target_counts = {}
    for t in ["YBCO", "LTOPO", "NTMO"]:
        sub = [r for r in rows if r["target"] == t]
        n_carb = sum(r["has_carbonate_narrow"] for r in sub)
        per_target_counts[t] = {"n_total": len(sub), "n_carbonate": n_carb, "n_non_carbonate": len(sub) - n_carb}
        print(f"  {t}: {n_carb}/{len(sub)} carbonate (narrow definition)")

    rows_fe = [dict(r, carbonate_status=1 if r["has_carbonate_narrow"] else 0,
                    is_ybco=1 if r["target"] == "YBCO" else 0,
                    is_ltopo=1 if r["target"] == "LTOPO" else 0) for r in rows]
    x_keys = ["carbonate_status", "best_wt_pct", "is_ybco", "is_ltopo"]
    coeffs = ols_coeffs(rows_fe, "delta_gdpo_rssft", x_keys)

    def carb_coef_fe(subset):
        if len({r["carbonate_status"] for r in subset}) < 2:
            return coeffs["carbonate_status"]
        c = ols_coeffs(subset, "delta_gdpo_rssft", x_keys)
        return c["carbonate_status"]

    ci = cluster_bootstrap_ci(rows_fe, carb_coef_fe, rng)
    result = {
        "per_target_counts": per_target_counts,
        "coefficients_with_target_fe": coeffs,
        "carbonate_status_ci_with_fe": ci,
        "excludes_zero": not (ci[0] <= 0 <= ci[1]),
        "survives": not (ci[0] <= 0 <= ci[1]),
    }
    print(f"  carbonate_status coefficient (with target FE) = {coeffs['carbonate_status']:.4f}, "
          f"95% CI = {ci}, survives = {result['survives']}")
    return result


def part2_broad_definition(rows, rng):
    print("\n=== Part 2: Q4 broader carbonate definition (CO3 substring) ===")
    n_flip = sum(1 for r in rows if r["has_carbonate_broad"] and not r["has_carbonate_narrow"])
    n_narrow = sum(r["has_carbonate_narrow"] for r in rows)
    n_broad = sum(r["has_carbonate_broad"] for r in rows)
    print(f"  narrow definition: {n_narrow}/{len(rows)} carbonate")
    print(f"  broad definition (CO3 substring): {n_broad}/{len(rows)} carbonate")
    print(f"  routes reclassified non-carbonate -> carbonate: {n_flip}")

    rows_b = [dict(r, carbonate_status=1 if r["has_carbonate_broad"] else 0,
                   is_ybco=1 if r["target"] == "YBCO" else 0,
                   is_ltopo=1 if r["target"] == "LTOPO" else 0) for r in rows]

    coeffs_no_fe = ols_coeffs(rows_b, "delta_gdpo_rssft", ["carbonate_status", "best_wt_pct"])

    def carb_coef_no_fe(subset):
        if len({r["carbonate_status"] for r in subset}) < 2:
            return coeffs_no_fe["carbonate_status"]
        c = ols_coeffs(subset, "delta_gdpo_rssft", ["carbonate_status", "best_wt_pct"])
        return c["carbonate_status"]

    ci_no_fe = cluster_bootstrap_ci(rows_b, carb_coef_no_fe, rng)

    x_keys_fe = ["carbonate_status", "best_wt_pct", "is_ybco", "is_ltopo"]
    coeffs_fe = ols_coeffs(rows_b, "delta_gdpo_rssft", x_keys_fe)

    def carb_coef_fe(subset):
        if len({r["carbonate_status"] for r in subset}) < 2:
            return coeffs_fe["carbonate_status"]
        c = ols_coeffs(subset, "delta_gdpo_rssft", x_keys_fe)
        return c["carbonate_status"]

    ci_fe = cluster_bootstrap_ci(rows_b, carb_coef_fe, rng)

    result = {
        "n_narrow": n_narrow, "n_broad": n_broad, "n_reclassified": n_flip,
        "coefficients_no_fe": coeffs_no_fe, "ci_no_fe": ci_no_fe,
        "survives_no_fe": not (ci_no_fe[0] <= 0 <= ci_no_fe[1]),
        "coefficients_with_fe": coeffs_fe, "ci_with_fe": ci_fe,
        "survives_with_fe": not (ci_fe[0] <= 0 <= ci_fe[1]),
    }
    print(f"  broad def, no FE: coef={coeffs_no_fe['carbonate_status']:.4f}, CI={ci_no_fe}, "
          f"survives={result['survives_no_fe']}")
    print(f"  broad def, with target FE: coef={coeffs_fe['carbonate_status']:.4f}, CI={ci_fe}, "
          f"survives={result['survives_with_fe']}")
    return result


def part3_length_covariate(rows, rng):
    print("\n=== Part 3: Q4 length covariate ===")
    carb = [r for r in rows if r["has_carbonate_narrow"]]
    non_carb = [r for r in rows if not r["has_carbonate_narrow"]]
    mean_tok_carb = sum(r["n_tokens_base"] for r in carb) / len(carb)
    mean_tok_non = sum(r["n_tokens_base"] for r in non_carb) / len(non_carb)
    print(f"  mean n_tokens_base: carbonate={mean_tok_carb:.2f} (n={len(carb)}), "
          f"non-carbonate={mean_tok_non:.2f} (n={len(non_carb)})")

    rows_l = [dict(r, carbonate_status=1 if r["has_carbonate_narrow"] else 0) for r in rows]
    x_keys = ["carbonate_status", "best_wt_pct", "n_tokens_base"]
    coeffs = ols_coeffs(rows_l, "delta_gdpo_rssft", x_keys)

    def carb_coef(subset):
        if len({r["carbonate_status"] for r in subset}) < 2:
            return coeffs["carbonate_status"]
        c = ols_coeffs(subset, "delta_gdpo_rssft", x_keys)
        return c["carbonate_status"]

    def len_coef(subset):
        if len({r["carbonate_status"] for r in subset}) < 2:
            return coeffs["n_tokens_base"]
        c = ols_coeffs(subset, "delta_gdpo_rssft", x_keys)
        return c["n_tokens_base"]

    ci_carb = cluster_bootstrap_ci(rows_l, carb_coef, rng)
    ci_len = cluster_bootstrap_ci(rows_l, len_coef, rng)

    result = {
        "mean_n_tokens_carbonate": mean_tok_carb, "mean_n_tokens_non_carbonate": mean_tok_non,
        "carbonate_routes_longer": mean_tok_carb > mean_tok_non,
        "coefficients": coeffs,
        "carbonate_status_ci": ci_carb, "carbonate_survives": not (ci_carb[0] <= 0 <= ci_carb[1]),
        "n_tokens_coef_ci": ci_len, "n_tokens_positive_and_significant": coeffs["n_tokens_base"] > 0 and not (ci_len[0] <= 0 <= ci_len[1]),
    }
    print(f"  carbonate_status coef (with length) = {coeffs['carbonate_status']:.4f}, CI={ci_carb}")
    print(f"  n_tokens_base coef = {coeffs['n_tokens_base']:.4f}, CI={ci_len}")
    return result


def part4_astral_rank_logp():
    print("\n=== Part 4: Original Q3 on ASTRAL [E] -- prompt-only rank/logp ===")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import dose_response_teacher_forcing as tf

    astral = json.loads(ASTRAL_PATH.read_text())["targets"]
    target_info = {t["target"]: t for t in astral if t["target"] in ASTRAL_TARGETS}
    for t in ASTRAL_TARGETS:
        info = target_info[t]
        print(f"  {t}: predicted[0]={info['predicted'][0]}  traditional[0]={info['traditional'][0]}")

    CHECKPOINTS = ["base", "rs_sft", "gdpo_300"]
    results = {t: {} for t in ASTRAL_TARGETS}

    for ckpt_name in CHECKPOINTS:
        print(f"\nloading checkpoint: {ckpt_name}...", flush=True)
        model, tok = tf.load_checkpoint(ckpt_name)
        for target in ASTRAL_TARGETS:
            info = target_info[target]
            prompt_text = tf.closed_prompt(target)
            rendered_prompt = tok.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
            shared_prefix = rendered_prompt + tf.FIXED_PREFIX + '[{"formula": "'

            with torch.no_grad():
                enc = tok(shared_prefix, return_tensors="pt").to(model.device)
                out = model(**enc)
                logits = out.logits[0, -1]  # last position predicts the first formula token
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                sorted_idx = torch.argsort(log_probs, descending=True)
                rank_of = {tid.item(): i + 1 for i, tid in enumerate(sorted_idx)}

            per_candidate = {}
            for label, formula in [("predicted", info["predicted"][0]), ("traditional", info["traditional"][0])]:
                full_text = shared_prefix + formula + '"'
                enc2 = tok(full_text, return_tensors="pt", return_offsets_mapping=True)
                offsets = enc2["offset_mapping"][0].tolist()
                del enc2["offset_mapping"]
                first_char = len(shared_prefix)
                first_tok_id = None
                for idx in range(1, enc2["input_ids"].shape[1]):
                    if offsets[idx][0] == first_char:
                        first_tok_id = enc2["input_ids"][0, idx].item()
                        break
                lp = log_probs[first_tok_id].item() if first_tok_id is not None else None
                rank = rank_of.get(first_tok_id) if first_tok_id is not None else None
                per_candidate[label] = {
                    "formula": formula, "first_token_id": first_tok_id,
                    "first_token_str": tok.decode([first_tok_id]) if first_tok_id is not None else None,
                    "logp": lp, "rank": rank,
                }
            results[target][ckpt_name] = per_candidate
            print(f"  {target} [{ckpt_name}]: predicted({info['predicted'][0]}) rank={per_candidate['predicted']['rank']} "
                  f"logp={per_candidate['predicted']['logp']:.3f}  |  "
                  f"traditional({info['traditional'][0]}) rank={per_candidate['traditional']['rank']} "
                  f"logp={per_candidate['traditional']['logp']:.3f}")
        tf.unload(model)
        print(f"  {ckpt_name}: DONE, GPU freed", flush=True)

    return results


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def part4b_divergence_point():
    """[E] addendum, not literally requested: part4 showed predicted and
    traditional share an identical first token for 3/4 targets (same
    leading cation symbol), making the literal first-token rank/logp
    uninformative (they are always equal). This finds the first token
    at which the two formula strings actually diverge and reports
    rank/logp there instead -- the point where the model's choice
    between predicted and traditional is actually made."""
    print("\n=== Part 4b [E, added beyond the literal ask]: divergence-point rank/logp ===")
    import torch
    import dose_response_teacher_forcing as tf

    astral = json.loads(ASTRAL_PATH.read_text())["targets"]
    target_info = {t["target"]: t for t in astral if t["target"] in ASTRAL_TARGETS}

    CHECKPOINTS = ["base", "rs_sft", "gdpo_300"]
    results = {}
    for target in ASTRAL_TARGETS:
        info = target_info[target]
        pred, trad = info["predicted"][0], info["traditional"][0]
        cp_len = common_prefix_len(pred, trad)
        if cp_len == min(len(pred), len(trad)) and pred[:cp_len] == trad[:cp_len] and pred == trad:
            results[target] = {"note": "predicted and traditional first precursor are identical; no divergence point", "predicted": pred, "traditional": trad}
            print(f"  {target}: predicted == traditional == {pred}, skipping")
            continue
        results[target] = {"common_prefix": pred[:cp_len], "predicted": pred, "traditional": trad, "checkpoints": {}}

    for ckpt_name in CHECKPOINTS:
        print(f"\nloading checkpoint: {ckpt_name}...", flush=True)
        model, tok = tf.load_checkpoint(ckpt_name)
        for target in ASTRAL_TARGETS:
            if "checkpoints" not in results[target]:
                continue
            info = target_info[target]
            pred, trad = info["predicted"][0], info["traditional"][0]
            cp_len = common_prefix_len(pred, trad)
            common_prefix = pred[:cp_len]
            prompt_text = tf.closed_prompt(target)
            rendered_prompt = tok.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
            shared_prefix = rendered_prompt + tf.FIXED_PREFIX + '[{"formula": "' + common_prefix

            with torch.no_grad():
                enc = tok(shared_prefix, return_tensors="pt").to(model.device)
                out = model(**enc)
                logits = out.logits[0, -1]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                sorted_idx = torch.argsort(log_probs, descending=True)
                rank_of = {tid.item(): i + 1 for i, tid in enumerate(sorted_idx)}

            per_candidate = {}
            for label, formula in [("predicted", pred), ("traditional", trad)]:
                full_text = shared_prefix + formula[cp_len:] + '"'
                enc2 = tok(full_text, return_tensors="pt", return_offsets_mapping=True)
                offsets = enc2["offset_mapping"][0].tolist()
                del enc2["offset_mapping"]
                first_char = len(shared_prefix)
                first_tok_id = None
                for idx in range(1, enc2["input_ids"].shape[1]):
                    if offsets[idx][0] == first_char:
                        first_tok_id = enc2["input_ids"][0, idx].item()
                        break
                lp = log_probs[first_tok_id].item() if first_tok_id is not None else None
                rank = rank_of.get(first_tok_id) if first_tok_id is not None else None
                per_candidate[label] = {
                    "divergent_continuation": formula[cp_len:],
                    "first_token_id": first_tok_id,
                    "first_token_str": tok.decode([first_tok_id]) if first_tok_id is not None else None,
                    "logp": lp, "rank": rank,
                }
            results[target]["checkpoints"][ckpt_name] = per_candidate
            print(f"  {target} [{ckpt_name}] after common prefix '{common_prefix}': "
                  f"predicted('{per_candidate['predicted']['divergent_continuation']}') "
                  f"rank={per_candidate['predicted']['rank']} logp={per_candidate['predicted']['logp']:.3f}  |  "
                  f"traditional('{per_candidate['traditional']['divergent_continuation']}') "
                  f"rank={per_candidate['traditional']['rank']} logp={per_candidate['traditional']['logp']:.3f}")
        tf.unload(model)
        print(f"  {ckpt_name}: DONE, GPU freed", flush=True)

    return results


def main():
    rng = random.Random(SEED)
    rows = build_rows()

    p1 = part1_target_control(rows, rng)
    p2 = part2_broad_definition(rows, rng)
    p3 = part3_length_covariate(rows, rng)
    p4 = part4_astral_rank_logp()
    p4b = part4b_divergence_point()

    OUT_JSON.write_text(json.dumps({
        "part1_target_control": p1,
        "part2_broad_carbonate_definition": p2,
        "part3_length_covariate": p3,
        "part4_astral_prompt_only_rank_logp": p4,
        "part4b_divergence_point_e": p4b,
    }, indent=1, default=str))
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
