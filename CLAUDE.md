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

**Answer as of 2026-08-26: it sharpens, and the factored verifier cannot show
which subskills moved because it has no within-group variance to show them
with.** The project's contribution has shifted from "a better synthesis model"
to two things: (a) a diagnostic account of why a physically-decomposed verifier
fails to support factored-reward RL (Central Diagnosis below), and (b) — newer
and stronger — **three positive interpretability results on a model whose
behavioral story is a null** (findings 12–14). The representational axis is
where the signal is: success is predictable at AUC ~0.75, SFT's capability
redistribution has a mechanistic depth signature, and the base model already
encodes which targets are unsolvable. The RL null becomes context for that,
not the headline.

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

12. **Interp probe A — success IS predictable from representation (2026-08-26, hardened)**. Linear probe on prompt activations, 200 targets, 5-fold CV, 2000-sample bootstrap. Continuous pass@1 is NOT predictable (R² negative at every layer, every checkpoint). Binary hit/miss IS, at **mid-late layers**:

    | checkpoint | best layer | AUC | 95% CI |
    |---|---|---|---|
    | base | 26 | 0.745 | [0.653, 0.828] |
    | sft | 21 | 0.767 | [0.662, 0.857] |
    | gdpo300 | 20 | 0.729 | [0.598, 0.840] |

    All CIs clear 0.5. **CIs overlap heavily across checkpoints — do NOT claim SFT's 0.767 beats base's 0.745.** A usable calibrated-confidence signal: the representation knows, before generating, whether the model is likely to fail. (v1 reporting bug, now fixed: "best layer" selected on R², printing layer 0 — the worst layer by AUC — and making a real result look null.)
13. **Interp probe B — SFT rotates the frac/int direction progressively with depth (2026-08-26, hardened)**. Diff-of-means direction between fractional and integer targets (frac=94, int=106), perm_p=0.000 at 36/37 layers. `cos(d_base, d_sft)` by layer:

    | layer | 1 | 4 | 10 | 15 | 20 | 25 | 30 | 36 |
    |---|---|---|---|---|---|---|---|---|
    | cos | 0.997 | 0.971 | 0.858 | 0.647 | 0.535 | 0.465 | 0.379 | **0.244** |

    **Random-direction null (d=4096, 2000 pairs): mean 0.0002, 5–95% band [−0.025, +0.026].** So 0.244 is ~9× the null's upper band → **SFT rotates the direction ~76° by the final layer but does NOT replace it.** Progressive reorganization, not overwriting. ‖d_sft‖/‖d_base‖ = 0.85. Held-out projection-vs-Δpass@1 Spearman ρ≈0.31, p<1e-5, consistent across all three checkpoints (layer 3). Mechanistic signature of finding 10's redistribution, localized to depth; same architectural shape as the PD-attention "read early, discounted late" result. Caveat: early-layer alignment (0.997) is trivially lexical — fractional formula strings contain decimal points. **OPEN BUG: `held_out_auc_frac_vs_int` and `held_out_cohens_d` are None at all 37 layers — the held-out separation refit did not run or failed silently. The held-out Spearman did run. Chase this: held-out separation is what shows the direction generalizes rather than fitting the 200 targets it was built from.**
14. **Interp probe C — hard-zeros are separable, but shallower and weaker than v1 suggested (2026-08-26, with malformed-formula ablation)**. 17/200 targets score 0/16 on base AND sft AND gdpo300. Logistic probe on **base** activations, LOO AUC:

    | set | best layer | AUC there | AUC at layer 22 | sig layers |
    |---|---|---|---|---|
    | full 17 | 22 | 0.821 [0.706, 0.913] | 0.821 | 36/37 |
    | reduced 12 (malformed dropped) | **3** | 0.786 [0.629, 0.916] | **0.716** | 36/37 |

    **The pre-registered rule (AUC >0.75 at layer ~22 on the reduced set) was NOT met: 0.716.** Separability survives everywhere (36/37 layers p<0.001, layer-0 control at chance), but the deep-layer peak in the full set was substantially driven by the 5 malformed corpus artifacts (`BaCrO`, `MgSnZnO`, `LiFeBO3C`, `Eu4Y1`, `ReBa2Cu3O`); on chemically-real hard-zeros the signal is strongest at **layer 3**. Honest claim: *hard-zero targets are linearly separable in base activations (AUC 0.79 at layer 3), but the effect is closer to a formula-family feature than to deep synthesis reasoning.* **The script's auto-verdict "FINDING STANDS" is too strong — do not quote it.** The earlier layer-22 / CKA-dip convergence does NOT survive; drop it. **OPEN BUG: `layer0_negative_control` reports AUC = 0.000, not ~0.5. A perfectly-inverted classifier ≠ chance; likely a sign convention or a degenerate constant-prediction case. `perm_p`=1.000 gives the right conclusion (no signal) but the 0.000 must be explained before publication.**

15. **Low-temperature objective probe — PARTIAL, below the pre-registered bar (2026-08-27)**. 40 targets × 8 samples, SFT policy, closed-book. Soft prompt preference for lower processing temperature + a `temperature_economy` channel gated on thermodynamic feasibility.

    | | baseline | low_temp |
    |---|---|---|
    | capacity (10 original channels) | 16.4% | 19.0% |
    | **capacity (incl. `temperature_economy`, 11 ch)** | — | **23.8%** |
    | routes/group | 2.17 | 2.15 |
    | p̂@0.9 | 0.646 | 0.610 |
    | mean reward | 0.919 | 0.911 |

    **Pre-registered rule: >40% → run 4; <25% → fifth falsification. Result 23.8% — just under the failure line.** But it is the first intervention that did anything real:
    - `temperature_economy` z-var **0.721** — the healthiest channel ever produced here, beating `amount_accuracy` (0.62). Within-group std 0.141; only **6/37 groups** flat (84% informative). Mean 0.340, spread across the range (98 completions at 0.0, 21 at 1.0).
    - **Two untouched channels partially revived**: `operation_order` z-var 0.227→0.394 (+74%, zero-std groups 72.5%→55.0%), `precursors_exist` 0.022→0.064 (+186%, 97.5%→92.5%). This is the predicted causal chain — temperature pressure forces different precursor families (carbonates decompose below oxides) → different operation sequences → previously-blind checks can see it.
    - **Why it stalled**: mean reported max-T under the soft objective = **1056 °C**, *higher* than the 993 °C produced under a hard ceiling (finding 9). ~1/3 of completions scored 0.0 on temperature_economy, i.e. blew past `T_ref` entirely. The model treated the preference as advisory. This is the spec's first listed failure mode — a prompt-strength problem, not a null.
    - Cost: paired Δp̂@0.9 = −0.037 (95% CI [−0.077, +0.001]), Δmean-reward = −0.008 (CI [−0.016, −0.001]). Small, marginal, expected direction.
    - Note: `adherence` is empty for baseline records (no `max_T_reported`), so the baseline-vs-low_temp temperature comparison uses finding 9's ceiling condition as the reference point.

16. **THREE-MODEL ASTRAL AT n=32 — SFT DESTROYS A BASE CAPABILITY THE VERIFIER WOULD HAVE REWARDED (2026-09-03)**. 35 targets × 32 samples, base / SFT / GDPO-300, identical code path.

    | model | proposes better set | proposes conventional | validator: better | validator: conventional | **gap** | mean max-T |
    |---|---|---|---|---|---|---|
    | **base** | **10/35** | 24/35 | 0.993 (n=22) | 0.972 (n=309) | **+0.021** | 1026.5 °C |
    | **SFT** | **1/35** | 32/35 | 0.852 (n=2) | 0.905 (n=525) | **−0.053** | 963.2 °C |
    | **GDPO-300** | **3/35** | 31/35 | 0.967 (n=4) | 0.918 (n=530) | **+0.049** | 1017.1 °C |

    - **base → SFT loss is significant**: paired exact McNemar **p = 0.004–0.012**.
    - **SFT → GDPO recovery is NOT**: p ≈ 0.5–0.6, recovers ~22% of what was lost. Report as "consistent with partial recovery," never as recovery.
    - base vs GDPO borderline (p = 0.016–0.092) — after RL the model is plausibly still below where it started.
    - **n=8 was severely underpowered**: base read 3/35 at n=8 vs 10/35 at n=32. The earlier three-model table (old finding 16 / H4) is superseded.
    - **THE DECISIVE COLUMN IS THE GAP.** SFT is the *only* model that scores the experimentally-superior routes **below** conventional ones. Base prefers them; GDPO prefers them most. **The verifier points the right way; the policy cannot get there.** This is a support problem, not a reward problem.
    - Temperature overshoot vs the lab's best condition: base **+264 °C**, SFT **+200**, GDPO **+254**. SFT moved toward realistic temperatures; RL pushed back up — finding 6's hack as a three-point arc on unseen data.

17. **THE GEOMETRIC CONSEQUENCE — REVISED 2026-09-06, the original attribution was wrong.**

    **Correction first, verified empirically not read from source:** the GDPO KL reference has **always been base Qwen3-8B, never SFT**. No adapter is ever named `"ref"`, so TRL's `use_adapter` resolves to `disable_adapter()`, which is bit-identical to a freshly loaded base model (`run_debug_and_analysis/verify_ref_model_identity.py`: max |logit diff| vs fresh base = **0.000000**; vs the SFT policy = 10.05). PHASE11_REVISED's "re-anchor the KL to base" Step 1 was therefore a **config no-op** — caught before launch, saving ~5 GPU-days.

    **So the support freeze was never caused by the KL anchor.** The correct mechanism: π\* ∝ π_ref·exp(r/β) is the **optimum, not the trajectory**. Policy gradient does local ascent with 𝔼_{y∼π_θ}[∇log π_θ(y)·A(y)] — the expectation is over the **current policy**, not the reference. A route π_θ never samples is never scored and never reinforced, whatever mass π_ref assigns it; at β=0.001 the pull toward base is negligible. This is the **absorbing boundary** (Vol II Part 1): the gradient for any output is gated by its own sampling probability.

    **The binding axis is the sampling distribution (axis 4), not the reference policy (axis 3).** The revision makes the claim *stronger* — it holds for on-policy RL generally, not only KL-anchored RL — and it predicts the Phase 11 results exactly: changing what gets sampled (the init) worked; changing the reference would not have.

    **Axes MIRA has moved on:** the reward (five interventions) and β (run 2) — both **within-face**, i.e. they reweight inside the current support without changing it. **Support-changing axes:** the **sampling/initialization distribution** (axis 4 — RS-SFT, format-only SFT), the **projection type** (inserting an m-projection; forward KL is *forced* to cover its data's support), and the **sample space Ω** itself (schema / tool-call changes, e.g. Chain-of-Abstraction). The simplex is stratified by support — distributions on a subset S form a face Δ(S) — and e-geodesics cannot leave a face while m-geodesics (mixtures) can. SFT is m-projection and **can** change support; RL is e-projection and, on-policy, **cannot**.

18. **PHASE 11 — RS-SFT FROM BASE RETAINS WHAT CORPUS FINE-TUNING DESTROYS (2026-09-06)**. ASTRAL, 35 targets × 32 samples:

    | model | predicted | conventional | mean reward | max-T | overshoot |
    |---|---|---|---|---|---|
    | base | 10/35 | 24/35 | 0.948 | 1026.5 | +264 |
    | format-only SFT | 11/35 | 31/35 | 0.937 | 1070.8 | +308 |
    | **RS-SFT from base** | **10/35** | **17/35** | **0.972** | **934.2** | **+171** |
    | full SFT | 1/35 | 32/35 | 0.888 | 963.2 | +200 |
    | GDPO-300 (from full SFT) | 3/35 | 31/35 | 0.894 | 1017.1 | +254 |

    - **RS-SFT vs full SFT on predicted hits: p = 0.004–0.012.** Same significance as base-vs-SFT. 295 survivors from 400 base-generated targets, validator bar 0.9, 73.75% survival, fine-tuned from base.
    - **Conventional 17/35 is the lowest of anything including base** (p = 0.016 best case) — it moved *away* from convention rather than merely avoiding collapse.
    - **Lowest temperature overshoot (+171 °C).** The strongest number in the table, because temperature is externally checkable while the validator reward is not (H1: chance agreement with experiment). Do not lead with the 0.972.
    - **Format-only is an interesting negative**: retains base's hit rate (11/35, p = 1.0 vs base) but conventional climbed 24→31 and temperature got *worse* (+308, worst of all). A few hundred examples for one epoch still pulled it toward convention. Format installation is not free.
    - **Confirms the central claim**: it is fine-tuning *on the external corpus* that closed the loop, not fine-tuning per se.

19. **Ranker v2 FAILED the external gate — with a caveat that must be stated (2026-09-06).** Agreement 21/35 = 60.0% against a pre-registered 24/35 bar; Spearman 0.133, weaker than the plain validator's 0.228. Stopping was correct.

    **But three of eight channels were ungradeable on all 35 pairs** (`temperature_economy`, `slice_competing_phases`, `precursor_decomposition_match`) because ASTRAL supplies precursor *species* without molar ratios or per-route temperature sweeps. The gate tested a ranker running on roughly half its channels. **Honest claim: "failed the gate as evaluated, on a dataset that cannot exercise several of its channels"** — a data-compatibility failure as much as a design failure. Do **not** write "two verifier generations failed external validation" without this caveat.

    **Do not lean on the `n_precursors` result** (9/17 disagreements, reported as "worse than chance"). 9/17 is *exactly* chance at n=17 and is not evidence against ASTRAL's principle 1. A separate real confound: ASTRAL's missing molar ratios make 3-precursor stoichiometry harder for the balance solver to close, systematically favouring 2-precursor routes at the gate stage independent of chemistry.

## Journey (why things are the way they are)

1. Six checkpoints once "scored identically" — root cause: silent parse-failure fallback; then the sentinel-payout bug. Validator now has None-propagation end to end + 43 tests.
2. SFT v3 (real corpus) → p̂ probes: saturated at bar 0.65, headroom at 0.9 → RL's job is top-decimal precision.
3. Run 1 (GDPO β=0.04) + run 2 (β-ablation): produced the capacity/diversity diagnostics above, and taught us every per-check trend curve was target-sampling noise (0/612 targets repeated).
4. Run-3 prep (5-channel uniform reward, closed-book data filtered to mid-band p̂ + gradeability-stable, fixed 30-target probe eval, `epsilon_high=5.0`, EvalModeGuard for a TRL eval bug) — all implemented and smoke-verified.
5. **Run 3 was launched, then killed** after a strategy review: pass@k n=200 came first and returned null (finding 4), so run 3's capacity verdict answers a question nobody needs answered *yet*. `run_gdpo_run3.sh` remains ready.
6. **Hardening probe run, prediction failed** (finding 9). The hypothesis was that dead channels were caused by score-irrelevant policy variation; making variation score-relevant should raise capacity. It didn't, and constraining *reduced* diversity. This falsified the diversity-first diagnosis and produced the Central Diagnosis above: the bottleneck is that the verifier cannot distinguish good from great, not that the policy fails to vary.
7. **Now**: two live directions — (a) mechanistic interpretability on the *stratum redistribution* signal (finding 10), which unlike the reward channels has large measured variance to explain; (b) a reward redesign that converts validity-checking into quality-optimization (below). Direction (a) is ~30× cheaper than a training run and is the current priority.

## Active work / next steps (ordered)

**Why interp before more RL — now confirmed empirically.** Every reward-side
axis is blocked by the same bottleneck (you cannot find representation structure
aligned to a channel with no variance — that is finding 5's null). Finding 10
was the one axis with a large differential effect to explain, and probing it
worked: findings 12–14 are all positive, cost ~2 h of GPU, versus ~5 days for a
GDPO run. **The pattern to keep: probe contrasts that are behaviourally
measured and large, not reward channels that are constant.** Items 1–3 below
harden the three results; items 4+ are the capability-side track, unblocked but
lower priority now.

## PHASE 12 — GDPO FROM RS-SFT

Full plan: `misc/some_claude_files/PHASE12_INSTRUCTIONS.md`.

Findings 17–18 set this up. Every prior GDPO run started from a policy that had already
lost the good routes; RS-SFT-from-base has them in support (10/35), and the validator
already scores them **+0.049 above conventional**. The reward points the right way — the
open question is whether on-policy sampling can now reach them.

1. **THE RUN: GDPO initialised from `runs/rs-sft-from-base/final`**, existing
   **validator** (not the ranker — it has not passed a gate), everything else identical
   to Arm A: β=0.001, lr=1e-5, G=8, batch 1 × accum 16, `epsilon_high=5.0`, closed-book,
   `data/rl_run3`, `EvalModeGuard`, checkpoints every 100, fixed 30-target probe every
   50 steps, per-check std logged individually. **One variable: the initialisation.**
   Evaluate at checkpoint 300 for the matched Arm A comparison.

   **Pre-registered (ASTRAL predicted hits, N/35 at n=32, ckpt-300):**
   **>12/35** → RL amplified what was in support; the project's first positive result.
   **10–12/35** → preserved but not amplified. **<10/35** → RL degrades even a good
   starting point; the on-policy collapse is intrinsic, not inherited. All three are
   findings.

   Secondary: conventional hits (RS-SFT baseline 17/35 — does RL pull back toward
   convention?), max-T overshoot (baseline +171 — **does the temperature hack
   reassert?** full-SFT→GDPO went +200→+254), validator gap, reward capacity, routes per
   group, pass@k n=200.

2. **Smoke gate, 20 steps, hard asserts** — including **assert the init is RS-SFT, not
   full SFT**, with the resolved checkpoint path logged.

3. **CPU work while the GPU is busy:** the two open probe bugs; re-report the ranker gate
   restricted to gradeable channels (finding 19); record RS-SFT provenance (bar 0.9,
   73.75% survival, 295/400) — it is now a headline method and must be reproducible from
   the paper alone.

**HEADLINE METRIC CHANGED.** pass@1 against our own validator is inside the closed loop
(H1: chance agreement with experiment). Primary is now **"proposes the
experimentally-superior route, N/35" at n=32**. Bar to beat: **base 10/35**. SFT 1, GDPO 3.

4. **Article drafting (IN PROGRESS)** — draft at
   `misc/some_claude_files/PAPER_DRAFT.md`. Needs no new results; the low-T
   probe outcome slots into the "what would have to change" section either way.
5. **Reward redesign detail — the low-temperature objective (the one capability-side
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
- **Never delete files.** The user has previously lost analysis outputs and scripts. Every cleanup move is `git mv`; suspected dead code moves to `research/`, it does not get removed.
- **Repo layout (post-cleanup, 2026-09-08)**: root holds only the shared library surface — `train.py`, `validator.py`, `evaluate_batched.py`, `stratified_difficulty_eval.py`, `gibbs_corrector.py`, `probe_hardening.py`, `reward_geometry.py` (each imported by 5-25 files; moving them breaks call sites). `research/` (was `run_debug_and_analysis/`) holds all diagnostics and probes. `scripts/` holds tmux launchers. `results/` tracks the analysis JSONs that are evidence for every headline number — **untracked artifacts are `generations.jsonl`, `runs/`, triage blobs, chat dumps.** `docs/` holds the tracked narrative (timeline, phase results, decision record); `misc/` stays gitignored wholesale as the working scratch directory.
- SonarCloud was considered and **skipped** — it measures production-software metrics that do not apply to a research repo. The 43-assertion `tests/test_validator.py` is the quality signal to surface instead.

## Getting up to speed (token-efficient read order)

1. This file — especially the **Central Diagnosis** and findings 9/10.
2. `misc/some_claude_files/KIMI_NEW_CHAT_PRIMER.md` — the deep primer (58 lines): validator contract, probe inventory, run-3 spec detail, doc map.
3. On demand only: `gdpo_v4_next_steps_claude_recommendation.md` (capacity diagnostics D1–D5), `CLAUDE_CLOSED_BOOK_ANALYSIS.md` (closed-book + gradeability flap), `CLAUDE_RESPONSE_TO_RUN3_JUSTIFICATION.md` + `KIMI_ITEMS_2_3_5_PLAN.md` (the resequencing logic), `ANALYSIS_CHAIN_RESULTS.md` (temp sweep/attention/pass@k-15).
4. Data/results files (`runs/*/generations.jsonl`, `misc/*.json`) are large — inspect with `head -c`, `jq`-style one-liners, or the existing analyzers (`reward_geometry.py`, `analyze_passk.py`, `run_debug_and_analysis/`).
5. Before proposing anything, grep the repo for prior art — most good ideas have a script already (`probe_*.py`, `reward_geometry.py`, `run_debug_and_analysis/*`).