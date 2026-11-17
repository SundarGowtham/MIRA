# Phase 20 findings audit (Stage 0)

Written 2026-11-17, before any Phase 20 training. Spec: `docs/phases/PHASE20_PREREG.md` (blob `15d89d8a`).

## Tags

Each numbered finding in CLAUDE.md (1–34) gets one or more tags:

- **U**: unaffected. The finding concerns the validator, the comparator, external data, or Base only.
- **S**: SFT-side affected. It describes or uses a model trained by the truncated SFT path: Corpus SFT, RS-SFT, E3, E3c or format-only. Every such model trained on the prompt plus the first ~180 or ~550 reasoning tokens, with loss on the prompt, and never saw an answer.
- **R**: RL from a truncated-SFT init only. It describes an RL run (GDPO run 1/2 from Corpus SFT; Phase 12 and E1 from RS-SFT). The RL itself was not truncated, but its starting point was.
- **T**: uses temperature-mixed ASTRAL sample files. It sets a temperature-1.0 HF ASTRAL file next to a temperature-0.9 vLLM one (list below).

## Findings

| # | short name | tags | note |
|---|---|---|---|
| 1 | reward capacity 14% | S R | measured on GDPO run-1 groups from the Corpus SFT init |
| 2 | diversity collapse structural | S R | "an SFT property": SFT here is the truncated Corpus SFT |
| 3 | closed-book ≡ open-book | S | measured on the SFT policy |
| 4 | pass@k n=200: SFT 0.870 vs GDPO-300 0.880 | S R | |
| 5 | activation geometry null | S R | base part U |
| 6 | validity read / +200 °C temperature hack | R | completions from the RL policy; the validator's 100–2000 °C window is U |
| 7 | β=0.001, lr=1e-5 (run 2) | R | |
| 8 | gradeability flap | U | a validator property; measured on SFT samples |
| 9 | hardening probe | S | SFT policy |
| 10 | SFT redistributes capability by stratum | S | the headline "SFT" effect is the truncated SFT's |
| 11 | hard-zero pocket | S R | base pass@16 is U |
| 12 | interp probe A | S R | base row U |
| 13 | interp probe B (SFT rotates frac/int direction) | S | |
| 14 | interp probe C (hard-zeros) | S | the hard-zero set is defined using SFT and GDPO-300 |
| 15 | low-temperature objective probe | S | SFT policy |
| 16 | three-model ASTRAL n=32 | S R | base/SFT/GDPO-300 are all HF T=1.0: internally consistent, not T |
| 17 | geometric consequence / KL ref = base | U | the ref-model identity and the on-policy argument are unaffected; the "SFT lost the routes" premise is S |
| 18 | Phase 11 RS-SFT retains | S | base, format-only, RS-SFT, full SFT, GDPO-300 all HF T=1.0: not T |
| 19 | ranker v2 external gate | U | no model samples |
| 20 | KL estimator blowup | R | |
| 21 | Phase 12 pre-launch diagnostics | R | the parse-failure fix itself is U |
| 22 | Phase 12 AMPLIFIED (10→14/35) | R | RS-SFT and Phase 12@300 ASTRAL files both HF: not T. The ammonium split is HF-only |
| 23 | Phase 13 comparator, zero power on ASTRAL | U | |
| 24 | bare-oxide / carbonate shift | S R | shares from HF files (base/SFT/RS-SFT/GDPO); the Lee and ARROWS comparisons are U |
| 25 | ammonium bug vs bare-oxide | S R | the bug is U; the share trajectory is S R |
| 26 | RS-SFT does not amplify its bias; GDPO does | S R | split-half Jaccard HF-only |
| 27 | external ARROWS gates fail | U | |
| 28 | teacher-forcing dose response | S R | |
| 29 | `precursors_exist` = database coverage | U | |
| 30 | Phase 16 E1 | R T | hits 10→14 HF-only; **new recipes 8.3% vs floor 10.5% are T**: E1@300 HF ASTRAL vs RS-SFT vLLM reference |
| 31 | E3 vs E3c | S T | novelty vs Base's vLLM reference is consistent; **the "E3 13/35 vs Base/RS-SFT 10/35" hit comparison is T** (E3 vLLM, Base and RS-SFT HF) |
| 32 | ranker capacity 46.3% | S | on RS-SFT's vLLM draws, not T |
| 33 | new-recipe rate flat across checkpoints | R T | @100/@200 vLLM, @300 HF, floor vLLM. "Carbonate 48.2 → 25.3%" is vLLM→vLLM; step-300 points are HF |
| 34 | Phase 17 reward audit / signature regression | S R T | fine-tuning and practice signatures are S and R; Price equation and reachability back-test T; Hodge, information bound and tie stratum U except where pooled (see list) |

Phase 18 and Phase 19 results have no CLAUDE.md finding numbers. Their SFT steps (S1–S4) are S; S5–S7 are R; 18A and 18E are T (list below).

## Results files that mix HF T=1.0 and vLLM T=0.9 ASTRAL samples (Stage 0 item 3)

**The spec's premise is incomplete.** It says four models' ASTRAL files are HF T=1.0. In fact **seven** `results/astral_gen_n32_*.json` files came from `research/astral_model_generations.py` (HF, `temperature=1.0`, top-p 0.95, seed 42): base, sft (Corpus SFT), gdpo300, rs_sft, format_only, gdpo_phase12_beta0 (Phase 12@300), and e1 (E1@300). The vLLM T=0.9 ASTRAL sets are e3, e3c, every @100/@200 file, and every `data/novelty/*astral*.jsonl` (including `base_reference_astral` and `rssft_reference_astral`).

The four in the spec are the ones Phase 18/19 read through HF files. There, base and RS-SFT come from vLLM reference files. But the finding 16/18/22/30/31 hit tables use the HF files for **base and RS-SFT too**. Stage 5 resamples only the four, as registered. Base, RS-SFT and format-only stay HF-only, and comparisons against them inside HF are consistent.

Also: `results/support_growth_by_checkpoint.json`'s RS-SFT trajectory row was computed on 2026-10-15 from a vLLM rebuild of `astral_gen_n32_rs_sft.json`. That rebuild overwrote the tracked file, and the file was later restored from git. Its carbonate share (48.21%) does not match the restored HF file (45.94%).

**Directly mixed (one statistic compares an HF-group model with a vLLM-group model on ASTRAL):**

| file | script | HF models | vLLM models | quantity |
|---|---|---|---|---|
| `results/support_growth_by_checkpoint.json` | `research/support_growth_by_checkpoint.py` | e1@300, phase12@300 | rs_sft reference and trajectory row, e1/phase12 @100/@200 | support growth vs floor; hits and carbonate share by checkpoint (finding 33) |
| `results/phase16/e1_support_growth.json` | `research/phase16_e1_support_growth.py` | e1@300 | rs_sft reference and floor | finding 30's 8.3% vs 10.5% |
| `results/review/rulepicked_sample_share.json` | `research/review_1_rulepicked_sample_share.py` | base, corpus_sft, rs_sft, gdpo300, phase12@300, e1@300 | e3, e3c | sample-level rule-picked share; e3/e3c minus base paired bootstrap |
| `results/phase17/rc_price_equation.json` (+ `phase17_rc_followup.py`) | `research/phase17_rc_price_equation.py` | rs_sft, phase12, e1 (observed shift) | rs_sft reference (predicted covariance, novelty feature) | Price-equation sign check |
| `results/phase18/transport_exact.json`, `figdata_transport_exact.json` | `research/phase18_a_transport_exact.py` | corpus_sft, gdpo_v3@300, phase12@300, e1@300 | base, rs_sft, e3, e3c | 18A transport; mixed steps on ASTRAL: S1, S6, S7 |
| `results/phase18/posterior_matching.json` | `research/phase18_e_posterior_matching.py` | same four | base, rs_sft, e3, e3c | 18E lab-valid mass, coverage, MMD, hits |
| `results/phase17/audit_reach.json` (`l6_gains_backtest`) | `research/phase17_rh_reachability.py` | 4 gained targets chosen from HF rs_sft vs HF phase12 | RS-SFT probability from vLLM traces | reachability back-test |
| `results/phase17/b1_tail_calibration.json` (`l6_gains_detail`), copied in `rh_recount_no_tail.json` | `research/phase17_b1_tail_calibration.py` | same 4 targets | rssft_reference_astral | e.g. BaLiBO3: 10/200 in the vLLM reference, 0/32 in the HF file |
| `results/phase18/recipes_by_model.json` | `research/phase18_common.py` | 4 HF | 8 vLLM | the shared recipe cache behind 18A, 18E and Phase 19's probe strings |

**Pooled only (HF and vLLM ASTRAL samples pooled to choose which recipes get scored; the statistic is teacher-forced or from vLLM traces):** `results/phase17/universe.jsonl`, `universe_v2.jsonl` and everything built on them (`logp_universe.jsonl`, `signature.json`, `b1_split_trace_control*.json`, `geodesic.json`, `leash.json`, `figdata_ternary.json`, `transport.json`, `hodge_comparator.json`, `rh_recount_no_tail.json`, `review/c_calibrated_range.json`, `instrument_check.json`); Phase 19's `probe_strings.jsonl` (per-model counts from both groups) and through it `gate.json`, `kernel_predictions.json`, `reach.json`, `spillover.json`, `location.json`, `dimension.json`.

**Mixed ledger entries** (`results/ledger/CLAIMS_LEDGER.md`): `L10.e1_support`, `L10.support_by_checkpoint`, `L11.price`, `L11.reach`. Borderline: `L11.tail_calibration`, and `L11.b1_signature` and `L11.gate_g1` through the pooled universe. Finding 31's prose (E3 13/35 next to Base 10/35) is mixed even though the `L10.e3_*` entries are vLLM-only.

**HF-only on ASTRAL (internally consistent):** `astral_5model_n32.json`, `ammonium_ablation_check.json`, `phase12_astral_final_analysis.json`, `phase16/astral_clean_subset.json`, `phase16/validator_v2_rescore.json`, `phase17/block_a_*`, `distributional/{alkali_source_share, splithalf_jaccard, per_target, carbonate_penalty_by_channel, rs_sft_bareoxide_step}.json`, `rssft_ammonium_propagation_test.json`.

The final analysis's temperature-sensitivity measurement recomputes every row of the "directly mixed" table with the Stage 5 resamples in place of the four HF files.

## Draft CLAUDE.md finding (for Gowtham to paste)

> 35. **Every SFT model before Phase 20 was trained on a truncated prompt-plus-reasoning prefix, never on an answer (found 2026-11-16, Phase 19 Block 0; retrained in Phase 20).** `experiments/sft.py` passed `max_length=1024` (TRL `truncation_mode=keep_start`), set on 2026-05-17 from an audit of `sft_v2`, which had no reasoning traces (p99 858 tokens). Every later dataset has `<think>` traces: median full length 4,825 (Corpus SFT), 5,468 (RS-SFT) and 5,843 (E3/E3c) tokens; max 14,648. Each run therefore trained on the prompt plus the first ~550 (Corpus SFT) or ~180 (RS-SFT, E3, E3c) reasoning tokens. The loss covered the prompt too (~82% of trained tokens for RS-SFT/E3/E3c), and 0 of 1,217 / 283 and 2 of 505 examples reached their answer. TRL's default `max_length` is also 1024. RL was not truncated (TRL 1.8.0 GRPO has no prompt cap; completion cap 8,192 = the sampling cap), and the self-generated SFT data are unaffected. **Affected**: SFT-side, findings 1–5, 9–16, 18, 24–26, 28, 31, 32, 34 and Phase 18/19 SFT steps S1–S4. RL from a truncated-SFT init: 1, 2, 4–7, 11, 12, 16, 20–22, 24–26, 28, 30, 33, 34 and S5–S7. Unaffected: 8, 17 (except its premise), 19, 23, 27, 29. **Separate issue found in the same audit**: seven ASTRAL n=32 files (base, Corpus SFT, GDPO-300, RS-SFT, format-only, Phase 12@300, E1@300) were drawn by HF at temperature 1.0; every other ASTRAL set is vLLM at 0.9. Comparisons that cross the two (findings 30, 31's hit comparison, 33, 34's Price and reachability checks, Phase 18A/18E) are listed in `docs/phases/PHASE20_FINDINGS_AUDIT.md`. Phase 20 retrains RS-SFT, E3 and E3c with loss on the assistant turn only and no truncation (`train.py sft --sft-full`), reruns E1 from the corrected RS-SFT, and resamples four of the HF models at 0.9. Results: `docs/phases/PHASE20_RESULTS.md`.
