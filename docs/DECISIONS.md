# Decision record — pre-registered predictions and what happened

*Every row below states a prediction (or threshold) that was written down
**before** the result was seen, then the actual result, then the verdict.
This is the project's own discipline made visible: several of these went
against the hypothesis that motivated them, and are kept rather than
revised after the fact. Source file given for every number — see
[`results/`](../results/) for the underlying data and
[`TIMELINE.md`](TIMELINE.md) for the narrative.*

| # | decision point | pre-registered rule | result | verdict | source |
|---|---|---|---|---|---|
| 1 | pass@k, n=200, SFT vs. GDPO-300 | gap ≥ 5 points → RL expanded capability | +1.0 pt (188/200 ties, McNemar p=0.77); gap *shrinks* with k (+2.4→+1.0) | **FAIL — sharpening, not expansion** | `results/passk_n200.json`, CLAUDE.md finding 4 |
| 2 | Hardening probe, 5 conditions × 40 targets × 8 samples | capacity 14% → 40%+ under score-relevant variation | best condition 17.7% (temp_ceiling); inventory restriction *reduced* diversity (2.17→1.65 routes/group) | **FAIL — prediction backwards** | `results/hardening.json`, CLAUDE.md finding 9 |
| 3 | Low-temperature objective probe, 40 × 8 | >40% capacity → proceed to run 4; <25% → fifth falsification | 23.8% (11-channel, incl. `temperature_economy`) | **Marginal fail** — just under the stop line, but the first intervention with a real effect (`temperature_economy` z-var 0.72) | CLAUDE.md finding 15 |
| 4 | Interp probe C, hard-zero separability (reduced set) | AUC > 0.75 at layer ~22 | 0.716 | **NOT MET** — signal survives (36/37 layers p<0.001) but peaks at layer 3, not 22; weaker and shallower than v1 suggested | CLAUDE.md finding 14 |
| 5 | Ranker v1 capacity probe (RANKER_SPEC.md) | >40% → train run 4; 25–40% → fix and re-probe; <25% → stop | 51.12% (`ranker_capacity_recheck.json`, properly grouped) | **PASS** → run 4 (Arm B, ranker-scored) launched | `run_gdpo_run4.sh` header |
| 6 | Ranker v2 external gate vs. ASTRAL robot data | ≥24/35 (68.6%) pairwise agreement → proceed; <24/35 → stop, do not train | 21/35 = 60.0%, Spearman ρ=0.133 | **FAIL, stopped as instructed** — caveat: 3/8 channels ungradeable on this dataset (missing molar ratios), so the gate tested roughly half the ranker | `results/ranker_v2_astral_gate.json`, CLAUDE.md finding 19 |
| 7 | Format-only SFT (300 ex., ~1 epoch, from base) | parse rate ≳95% AND retains base's ASTRAL hit rate (10/35) | parse rate 99.3%; hit rate 11/35 | **PASS, both conditions** | `results/astral_gen_n32_format_only.json`, `docs/phases/PHASE11_RESULTS.md` §2 |
| 8 | RS-SFT from base (rejection-sampling SFT, validator bar 0.9) | retains base's support-preserving capability | 10/35 (ties base exactly), conventional-set hits *drop* to 17/35 (lowest of any model), highest mean reward (0.972), lowest max-T overshoot (+171 °C) | **PASS, exceeds base on diversity/temperature** | `results/astral_gen_n32_rs_sft.json`, CLAUDE.md finding 18 |
| 9 | GDPO from RS-SFT-from-base (Phase 12), ASTRAL N/35 at checkpoint 300 | **>12/35** → RL amplified support, first positive result. **10–12/35** → preserved, not amplified. **<10/35** → on-policy collapse is intrinsic | *(run not yet launched — smoke gate passed 2026-09-08, full run pending explicit go-ahead)* | **PENDING** | `misc/PHASE12_INSTRUCTIONS.md`, `run_gdpo_phase12.sh` |

## Corrections made along the way

Two of the above were revised after being acted on, and the correction is
kept rather than quietly fixed:

- **Row 5's "run 4" was launched, then its premise was reconsidered.** The
  ranker that passed the capacity gate later failed the external ASTRAL
  gate (row 6) — internal reward-capacity and external agreement with
  experiment are different questions, and passing one doesn't imply the
  other.
- **The Phase 11 "re-anchor the KL reference to base" plan (originally row
  9's Step 1) was verified empirically to be a config no-op**: the KL
  reference had always been base Qwen3-8B, never SFT, because no adapter
  is ever registered under the name `"ref"`. Caught by direct measurement
  (`run_debug_and_analysis/verify_ref_model_identity.py`: disabling the
  adapter reproduces a freshly-loaded base model's logits exactly, max
  diff 0.000000) before ~5 GPU-days were spent on it. CLAUDE.md finding 17
  documents the corrected mechanism.
