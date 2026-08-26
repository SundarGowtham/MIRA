# CLAUDE.md — MIRA orientation for a coding agent

Read this first. It is the map, not the territory: the deep doc family lives in
`misc/some_claude_files/` (read on demand, order at the bottom). This file is
accurate as of **2026-08-25**; if the calendar says otherwise, verify against
the repo before trusting specifics.

## What this project is

Fine-tune Qwen3-8B (QLoRA r16/α32) to generate **solid-state synthesis routes**
(precursors + operations with T/time/atmosphere as JSON after a `<think>` block)
for inorganic compounds. Two stages: SFT on DeepSeek teacher traces (filtered
≥0.65 by the validator), then RL (GRPO/**GDPO** = per-check z-normalized
multi-reward, TRL 1.8.0 `multi_objective_aggregation`) against **`validator.py`**
— a deterministic physics oracle (pymatgen + Materials Project phase-diagram
shard cache), NOT reference-recipe matching. The validator emits a factored
10-check reward vector + per-check gradeability tags; sentinel-tagged
("ungradeable") checks are excluded from reward by None-propagation at every
level. The scientific question: does RL expand the synthesis-capability
boundary or merely sharpen SFT's distribution — and can a factored verifier
show *which* subskills moved?

**Answer as of 2026-08-25: it sharpens, and the factored verifier cannot show
which subskills moved because it has no within-group variance to show them
with.** The project's contribution has shifted from "a better synthesis model"
to "a diagnostic account of why a physically-decomposed verifier fails to
support factored-reward RL." See the Central Diagnosis below.

## Central diagnosis (read this before proposing anything)

**The validator measures validity, not quality.** RLVR works where the verifier
can *rank* candidates — math has abundant wrong answers, code has failing tests.
MIRA's checks are validity gates that a well-trained 8B on a textbook-route
corpus almost always passes:

| check | mean | model fails it |
|---|---|---|
| target_match | 1.000 | never |
| temperature_plausible | 0.999 | never |
| precursors_exist | 0.998 | ~never |
| operation_order | 0.994 | rarely |
| charge_neutrality | 0.909 | across targets, never *within* a group |

**A channel is live only when the model fails it sometimes.** This is why
reward capacity is stuck at ~14–17% regardless of what is done to the prompt
(finding 9), why channel-aligned activation directions do not exist (finding 5),
and why GDPO could only sharpen (finding 4). Capacity is a property of the
**reward–policy pair**; interventions that change only the policy cannot move it.

Corollary for design: the single measured quantity with genuine within-group
variance in the whole hardening probe was **inventory adherence at 85.4%** —
a 15% failure rate, measured post-hoc and never rewarded. Any future reward
channel must be something the model fails at a meaningful rate.

## Repo map

- `validator.py` — the oracle. `VALIDATOR_VERSION` bumps on scoring changes. **The user owns this file; check before touching it.**
- `core/` — data/model/reward/observability. `core/reward.py::make_check_reward_fns` builds the RL reward vector (currently the 5-channel `RUN3_CHECKS`, uniform weights).
- `experiments/` — sft.py, grpo.py (base), gdpo.py (thin override), sft_grpo.py. `train.py` is the CLI dispatcher.
- `data/` — `sft_v3/` (1,217/157/157), `rl/` (6,568 prompts, SFT-disjoint), `rl_run3/` (run-3's filtered set), `cache/` (pd shards + formula set — **never delete anything here**), `raw/synthesis_clean.json` (17.6k Kononova literature routes).
- `runs/` — checkpoints. `sft-qlora-sft-v3-2nd-rank16/` (SFT, the RL-init), `gdpo-qlora-gdpo-v3/checkpoint-{100,200,300}` (run 1), `gdpo-qlora-beta-ablation-probe/` (run 2). `gdpo-qlora-gdpo-v4/` = killed run 3, dead artifact.
- `manifold_visualization/act_geo/` — activation cache (`cache/*_probe.npz`, `cache/*_completions.npz`) + `pass1_geometry.json`, `pass2_channels.json`. **Reusable: the interp probes below need no new forward passes for the probe-set work.**
- `misc/` — **gitignored scratch**: probes, results, the doc family.
- `run_logs/`, `run_*.sh` — tmux runners with ntfy pings.
- `tests/test_validator.py` — 43 assertions; run after ANY validator change: `uv run python tests/test_validator.py`.

## Verified findings (the load-bearing numbers)

1. **Reward capacity = 14%** (z-variance 1.43/10): only `amount_accuracy` (0.60) and `thermodynamic_favorable` (0.36) are alive within-group. 8/10 channels contribute no gradient (saturated, prompt-determined, or diversity-limited).
2. **Diversity collapse is structural**: ~2.0 distinct precursor sets/group of 8, flat across T=1.0–1.5 and across runs. Temperature is not a lever; it was an SFT property, not RL-caused.
3. **Closed-book ≡ open-book** (|Δp̂| ≤ 0.04): the model never used the PD context (attention "read early, discounted late"). Closed-book is default: 2× cheaper.
4. **pass@k n=200, k=16 (2026-08-23, THE decision result)**: SFT 0.870 vs GDPO-300 0.880 → paired diff **+1.0 pt** (median 0; 188/200 ties, 7 vs 5 decisive). The pre-set bar was ≥5 pts. **300 steps of GDPO did not expand the boundary.** The n=15 +6.6pt signal was small-n noise. Gap *shrinks* with k (+2.4 at k=1 → +1.0 at k=16), McNemar p=0.77 — the textbook **sharpening** signature.
5. **Activation geometry (2026-08-21)**: channel-aligned activation directions (z-weighted) do NOT separate from a within-group permutation null at any layer in base/SFT/GDPO-300 (median p 0.2–0.4, 0/37 layers p<0.05). CKA base→SFT 0.92, SFT→GDPO 0.96. No linear channel-subspace signal at pooled-completion resolution. **Caveat: mean-pooled over ~4,700 tokens; a localized-pooling redo is cheap and unrun.**
6. **Validity read (2026-08-18)**: 20 validator-≥0.9 completions vs literature (`misc/validity_read_20.md`) — mostly plausible route families, BUT confirmed the temperature hack: median **+200 °C** above literature across 18 comparable routes (15/18 hotter, extremes to +750 °C). AgCa₂Mg₂V₃O₁₂ scored 1.000 with a 1300 °C sinter (Ag volatilizes); `temperature_plausible` only checks a 100–2000 °C window.
7. β=0.001, lr=1e-5 confirmed (run-2 ablation: KL rose 3.7×, no collapse, clip_ratio=0).
8. Gradeability flap: up to 60% of fractional targets get graded by different channel sets across samples — run-3 dataset filters to stable-tag targets (strict = constant tag vector).
9. **Hardening probe — PRE-REGISTERED PREDICTION FAILED (2026-08-25)**. 5 conditions × 40 targets × 8 samples on the SFT policy. Predicted capacity 14% → 40%+; observed:

   | condition | capacity | routes/grp | % identical | p̂@0.9 |
   |---|---|---|---|---|
   | baseline | 16.4% | 2.17 | 37.5% | 0.646 |
   | temp_ceiling | **17.7%** | 2.15 | 37.5% | 0.639 |
   | inventory | 16.4% | **1.65** | **62.5%** | 0.658 |
   | atmosphere | 16.3% | 2.10 | 35.0% | 0.702 |
   | combined | 16.1% | 1.75 | 55.0% | 0.641 |

   **The constraints bound** (temp adherence 100%, inventory 85.4%) — this is a
   real null, not a prompt failure. Two sub-findings:
   - **Constraining the action space REDUCED diversity** (inventory: 2.17 → 1.65 routes/group, 37.5% → 62.5% identical). Restricting reagents removed the alternatives the model had been varying over. The prediction was backwards.
   - **The temperature hack is instrumental, not ignorance.** Under a ceiling the model complies 100% (mean max-T 993 °C) vs +200 °C median unconstrained. It *can* control T precisely; it inflates only because the validator grades ΔG at the model's own reported T. Confirmed by intervention, not correlation. Ironically `temperature_plausible` stayed 100% dead under the condition designed to revive it — perfect compliance means zero variance.
   - Four channels are 100% zero-std in *every* condition: `charge_neutrality`, `target_match`, `target_stability`, `temperature_plausible`. Nothing done to the prompt moves them.

10. **SFT REDISTRIBUTES CAPABILITY, IT DOES NOT UNIFORMLY IMPROVE IT (2026-08-23)** — the largest clean differential signal in the project, and currently the most publishable finding. pass@1 by stratum, SFT vs base, n=200:

    | stratum | Δ pass@1 | n |
    |---|---|---|
    | frac × interpolated | **+0.149** | 74 |
    | frac × ungradeable | **+0.106** | 20 |
    | int × discrete | −0.008 | 56 |
    | int × interpolated | −0.028 | 31 |
    | int × ungradeable | **−0.089** | 19 |

    A **24-point swing** between strata. The aggregate +5.1 pt gain is a weighted
    average of a real gain on doped/solid-solution targets and a real loss on
    integer-stoichiometry ones — consistent with distillation moving the model
    toward the corpus distribution (Kononova is fractional-heavy) and away from
    what the base model already knew. GDPO by contrast is ~uniform across strata
    (+0.01 to +0.05), consistent with sharpening. **Nobody reports this because
    nobody breaks pass@1 out by structural stratum.**

11. **Hard-zero pocket**: 12/104 targets (11.5%) score 0/16 at bar 0.9, and they are chemically coherent — complex site-substituted perovskites (`Ba(Zn₀.₃₃Ta₀.₆₇)O₃`, `Ca(Mg₀.₃₃Nb₀.₆₇)O₃`, `BaCe₀.₇Zr₀.₁Y₀.₁Yb₀.₁O₃`), non-oxides (`Si₀.₀₅Al₀.₉₅N`, `Li₂PO₂N`), alloys (`Mg₉₈.₅Gd₁Zn₀.₅`), unusual oxidation states (`Li₂NiO₂`, `SrBiO₂.₅`). Base pass@16 = 0.835 — there were only 16.5 pts of headroom to begin with; SFT captured 21%, GDPO 27%.

## Journey (why things are the way they are)

1. Six checkpoints once "scored identically" — root cause: silent parse-failure fallback; then the sentinel-payout bug. Validator now has None-propagation end to end + 43 tests.
2. SFT v3 (real corpus) → p̂ probes: saturated at bar 0.65, headroom at 0.9 → RL's job is top-decimal precision.
3. Run 1 (GDPO β=0.04) + run 2 (β-ablation): produced the capacity/diversity diagnostics above, and taught us every per-check trend curve was target-sampling noise (0/612 targets repeated).
4. Run-3 prep (5-channel uniform reward, closed-book data filtered to mid-band p̂ + gradeability-stable, fixed 30-target probe eval, `epsilon_high=5.0`, EvalModeGuard for a TRL eval bug) — all implemented and smoke-verified.
5. **Run 3 was launched, then killed** after a strategy review: pass@k n=200 came first and returned null (finding 4), so run 3's capacity verdict answers a question nobody needs answered *yet*. `run_gdpo_run3.sh` remains ready.
6. **Hardening probe run, prediction failed** (finding 9). The hypothesis was that dead channels were caused by score-irrelevant policy variation; making variation score-relevant should raise capacity. It didn't, and constraining *reduced* diversity. This falsified the diversity-first diagnosis and produced the Central Diagnosis above: the bottleneck is that the verifier cannot distinguish good from great, not that the policy fails to vary.
7. **Now**: two live directions — (a) mechanistic interpretability on the *stratum redistribution* signal (finding 10), which unlike the reward channels has large measured variance to explain; (b) a reward redesign that converts validity-checking into quality-optimization (below). Direction (a) is ~30× cheaper than a training run and is the current priority.

## Active work / next steps (ordered)

**Why interp before more RL:** every reward-side axis is blocked by the same
bottleneck (you cannot find representation structure aligned to a channel with
no variance — that is what finding 5's null means). Finding 10 is the one axis
with a large, clean, differential effect to explain. Cost comparison: a GDPO run
is ~1,400 s/step × 300 ≈ **5 days**; pass1+pass2 activation geometry took
**1–2 hours**. The probes below are hours, not days.

1. **Interp probe A — failure prediction (highest value to the chemistry community).**
   Linear probe on layer-L activations predicting whether the model will succeed
   on a target, trained on the 200-target pass@k success rates across
   base/SFT/GDPO-300. A calibrated "don't trust this one" signal is more useful
   in a lab than +2 pts of pass@1. Reuses `act_geo/cache/*_probe.npz`; CPU after
   the forward pass.
2. **Interp probe B — the redistribution direction.** Contrast frac vs int
   targets in activation space per checkpoint. Does SFT create or amplify a
   separating direction, and does its magnitude track the +0.149/−0.089 swing?
   This localizes what distillation did. Directly explains finding 10.
3. **Interp probe C — the hard-zero pocket.** Are the 12 never-solved targets
   (finding 11) representationally distinct in the **base** model, before any
   fine-tuning? If the base model already "knows" it cannot do these, that is a
   striking result and it feeds probe A.
4. **Reward redesign — the low-temperature objective (the one capability-side
   idea that plausibly moves capacity).** Convert the confirmed hack into the
   objective: reward = ΔG favorable *at the reported T* **minus** a term in T.
   Reporting a high T no longer helps because T is penalized directly. This
   works where inventory constraints failed because different precursor families
   have genuinely different minimum feasible temperatures (carbonates decompose
   lower than oxides, nitrates lower than carbonates) — so the Li₂CO₃↔LiOH swap
   the model already makes, which currently leaves `charge_neutrality` invariant,
   becomes **score-relevant**. The objective is continuous and unsaturable.
   Probe it with `probe_hardening.py` as a new condition; primary metric capacity.
5. **Multi-objective Pareto (extension of 4).** Minimize T *and* step count *and*
   precursor cost. No single right answer → genuine within-group variance by
   construction → the GDPO-vs-GRPO comparison finally means something, since
   scalar aggregation genuinely destroys Pareto structure.
6. **RS-SFT control arm** (scripted, unlaunched): `data_curation/build_rs_sft_dataset.py`
   — isolates "RL objective" from "verifier-filtered data". The standard RLVR
   control; reviewers will ask for it. Note: plain 24k-step SFT continuation
   (~200 epochs) is an overfit strawman — use RS-SFT instead.
7. **Validator temperature fix** (run-4 prep; the hardening probe is done so
   comparability no longer blocks it): decouple ΔG evaluation T from the model's
   reported T; replace `temperature_plausible` with a target-conditioned check.
   Subsumed by item 4 if that is adopted. User owns validator edits.
8. **Run 4** = winning condition + run-3 reward/config stack (`run_gdpo_run3.sh`
   pattern, new data). Only after 4/5 shows capacity >40%.
9. **Article drafting (user + Claude chat)**: the instrument-paper arc needs no
   new results — capacity 14%, the diversity mechanism, the temperature hack
   confirmed by intervention, the pass@k sharpening null, and the failed
   pre-registered hardening prediction. Finding 10 is the second paper.

**Structural limitation to state in the writeup:** Kononova contains ~32k routes
that *worked* and no information about what doesn't. A verifier built only from
successes can check validity but cannot rank quality. The precedent for fixing
this is Raccuglia et al. (Nature 2016), who mined *failed* hydrothermal
syntheses and found the failures carried more information than the successes.
That is a data-acquisition project, not a modeling one.

**Do NOT do:** SAE / dictionary-learning feature extraction on 8B residuals.
It is a project in itself, feature interpretation needs chemistry expertise the
user does not claim, and an SAE assumes the ontology it is testing (it can only
report directions; if the structure is curved it returns a fan of directions and
calls it feature splitting). Linear probes on a *measured behavioral contrast*
test a specific hypothesis instead of fishing.

## Operational playbook (each rule was learned the hard way)

- Run everything: `uv run python ...`; scripts outside repo root need `PYTHONPATH=.`; GPU work: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Secrets: `.env` (`set -a; source .env; set +a`) — never print it.
- **Long jobs ALWAYS in tmux** with tee + auto-restart (`run_gdpo.sh`, `run_gdpo_run3.sh`, `run_passk.sh` patterns; ntfy topic `mira-g5x7k2-status`). Agent-launched raw background processes die when the user logs off — the user has called this out repeatedly.
- **GPU memory rule** (32 GB card): full-vocab logits = batch × seq × 152k × bytes dominate. Per-device batch 1 at ~8–10k tokens; tune accum, never effective batch (16) or G (8). Generation ≈65% of RL step time.
- **git**: never commit/push/reset/rebase without the user explicitly asking. Stale 0-byte `.git/index.lock` reappears (editor plugin) — `rm` it if no git process runs. Committer identity is auto-guessed (fine).
- **Validator**: user is rewriting it on the side — check before big changes; small fixes + tests OK; bump `VALIDATOR_VERSION` on scoring-behavior changes.
- **Known traps**: `load_validator(formula_set, pd_index_path, project_root)` signature (old `pd_cache_path` kwarg dead); pd_index paths are repo-root-relative (pass `project_root=Path(".")`); `data/sft/` doesn't exist (`data/sft_old/`); `--model` is the BASE model — always pass `Qwen/Qwen3-8B`, never a checkpoint path (`--checkpoint` carries adapters); data/generation files are GBs — sample with `head -c`, never read fully.
- **TRL 1.8.0 + continuous batching + eval bug**: generation calls `unwrapped_model.train()` unconditionally (grpo_trainer.py:1685), misrouting later eval batches into the train branch. `EvalModeGuard` (core/observability.py) works around it — do not remove on TRL 1.8.0; re-test on any TRL upgrade.
- **Statistics rules** (learned from two misread runs):
  - Per-step W&B curves are **target-sampling noise**. With 2 groups/step and 0/612 targets repeated, no per-check trend is identifiable. Trends require a **fixed probe set re-evaluated every N steps**.
  - Model-vs-model claims must be **paired** with McNemar on discordant targets + bootstrap CI. Marginal means hide everything: SFT 0.870 vs GDPO 0.880 is 188 ties and 12 discordant targets.
  - pass@k at k = n_samples is **not an estimate** — it degenerates to "did any sample succeed." Usable k ≤ n/2.
  - `frac_reward_zero_std` is an aggregate and read 0 through both runs while 8 channels were dead. Log **per-check** within-group std.
  - Before claiming a capability change, check whether the **gap shrinks with k** (sharpening) or holds (expansion).
- User preferences: explicit permission before edits in new areas; verify by execution, no fabrication; be candid about null/bad results; minimal diffs; dense replies.

## Getting up to speed (token-efficient read order)

1. This file — especially the **Central Diagnosis** and findings 9/10.
2. `misc/some_claude_files/KIMI_NEW_CHAT_PRIMER.md` — the deep primer (58 lines): validator contract, probe inventory, run-3 spec detail, doc map.
3. On demand only: `gdpo_v4_next_steps_claude_recommendation.md` (capacity diagnostics D1–D5), `CLAUDE_CLOSED_BOOK_ANALYSIS.md` (closed-book + gradeability flap), `CLAUDE_RESPONSE_TO_RUN3_JUSTIFICATION.md` + `KIMI_ITEMS_2_3_5_PLAN.md` (the resequencing logic), `ANALYSIS_CHAIN_RESULTS.md` (temp sweep/attention/pass@k-15).
4. Data/results files (`runs/*/generations.jsonl`, `misc/*.json`) are large — inspect with `head -c`, `jq`-style one-liners, or the existing analyzers (`reward_geometry.py`, `analyze_passk.py`, `run_debug_and_analysis/`).
5. Before proposing anything, grep the repo for prior art — most good ideas have a script already (`probe_*.py`, `reward_geometry.py`, `run_debug_and_analysis/*`).