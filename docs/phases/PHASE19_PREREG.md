## Spec: Phase 19 pre-registration

### Why this phase exists

Phase 18A counted it exactly:
- every SFT step adds recipes its starting model never wrote;
- no RL step does.

The geometric argument says why. RL multiplies each recipe's probability by $e^{t\,r(y)}$, and zero times anything is zero. That argument treats every recipe as moving on its own.

In a real network recipes are coupled. Pushing up one string through shared weights also moves every string whose gradient points the same way. That coupling is the only route by which RL could create a new recipe. Phase 17 pre-registered the blocks that measure it (D, E, F) but never ran them.

Phase 19 runs them, with two changes learned since:
1. **Exact quantities only.** Phase 17 found that recipe-level probabilities from teacher forcing are biased low by about 20× to 10⁶× below $\pi=10^{-2}$ (`b1_tail_calibration.json`). The cause is that a recipe's probability is a sum over many ways of writing it.
   - Phase 19 therefore measures the log-probability of **one fixed string per recipe**, which is computed exactly.
   - It ties strings back to recipes only through sample counts.
2. **Location by weight swapping, not only activation patching.** Under LoRA, each layer's change is a separate adapter. Swapping adapters layer by layer asks directly which layers carry the change.

### Rules (frozen)

- **No training.** F2 generates new samples and needs explicit approval.
- **Recipe** means the canonical precursor set from `precset_of` in `research/support_growth_by_checkpoint.py`.
- **Probe set:** `results/phase17/probe_targets.json` (20 ASTRAL clean, 20 held out).
- **String for each recipe.** For each probe target, take the top 30 recipes by pooled sample count. Represent each by its most frequent exact answer string across all models' samples.
  - Context: prompt plus an empty `<think></think>` (answer-only), identical for every model.
  - Score: $\ell_m(y)=\log p_m(\text{string}\mid\text{context})$, summed over the answer tokens only.
- **Sampled status** comes from counts. A recipe is *sampled* by a model if it appears at least once in that model's existing samples for the target (232 for Base and RS-SFT, 32 for the others). Otherwise it is *unsampled*.
- **Steps** are as in `PHASE17_PREREG.md`:

  | Step | Before → after | Method |
  |---|---|---|
  | S3 | Base → E3 | SFT |
  | S4 | Base → E3c | SFT |
  | S6 | RS-SFT → Phase 12 @300 | RL, checker |
  | S7 | RS-SFT → E1 @300 | RL, ranker |

  S2 and S5 are secondary, run only if Block 0's time estimate allows.
- **Statistics:** bootstraps cluster by target, 10,000 resamples, seed 20261116.
- **Files:** code in `research/phase19_*.py`, results in `results/phase19/`, new files only. Do not edit `validator.py`, `core/comparator.py` or ranker code.
- **Jobs:** tmux with `tee` and ntfy, user-space environments. No `sudo`, no `apt`.
- **Reporting:** a failed prediction is reported as failed. Anything decided after the commit is labeled **[E]**.

---

### Block 0 — inventory (CPU)

1. **Probe strings.** Build `results/phase19/probe_strings.jsonl` with: target, recipe, string, pooled count, and per-model counts and sampled status.
2. **Adapters.** Confirm the adapter paths for S2–S7 and that each adapter's LoRA layers cover all 36 transformer blocks.
3. **Training data.** Locate the training data that D needs:
   - S3 and S4: their SFT training files, and the overlap between the probe targets and the E3 / E3c training targets (needed by P-19D-3);
   - S6 and S7: the first 50 training groups (400 completions) with their scores.
4. **Time estimate.** Estimate the GPU time per block.

### Block G — gate: does the exact string score track behavior? (GPU, forward only)

For each model in the step table:
- compute $\ell_m(y)$ for every probe string;
- per target, compute the Spearman correlation between $\ell_m(y)$ and the recipe's count in that model's samples.

Phase 17's answer-only medians were 0.45–0.57.

**Gate:** median ≥ 0.4 for every model.

**If it fails,** run E and D anyway, label every result **[E]**, and say so first in every report.

**Output:** `results/phase19/gate.json`.

---

### Block E — where is the change written? (GPU, forward only)

**Primary method: adapter swapping.** For step $s$ and depth $L$, build a hybrid model:
- the after-model's adapter in blocks $0..L$;
- the before-model's adapter above $L$.

For SFT steps, the before-model is Base, so "before" means no adapter. Build the reverse hybrid too: before-model's adapter in blocks $0..L$, after-model's above.

For each $L\in\{0,2,4,\dots,34,35\}$:
1. Score every probe string.
2. Compute recovery, pooled over targets:
$$R_s(L)=1-\frac{\sum_{x,y}\big(\Delta^{\text{hybrid}}_s(y)-\Delta_s(y)\big)^2}{\sum_{x,y}\Delta_s(y)^2},\qquad \Delta_s(y)=\ell_{\text{after}}(y)-\ell_{\text{before}}(y).$$
3. Define $L_{50}$ and $L_{80}$ as the first depths where bottom-up recovery reaches 0.5 and 0.8.

**Single-layer contribution [E]:** swap one block at a time and report its recovery.

**Secondary [E]:** residual-stream activation patching as in Phase 17 Block E, for comparison.

**Predictions [C]:**
- **P-19E-1.** $L_{50}$ is later for each RL step (S6, S7) than for each SFT step (S3, S4). All four orderings are required.
- **P-19E-2.** $L_{50}(S6)\ge 27$ and $L_{50}(S7)\ge 27$: the RL change sits in the last quarter of the network.

**Does not support** "RL is written late": P-19E-2 fails for both RL steps.

**Outputs:** `results/phase19/location.json`, `results/phase19/figdata_location.json` (recovery curves).

---

### Block D — the coupling: can RL reach an unsampled recipe through shared weights? (GPU, backward at fixed weights)

**Background.** One small training step changes a string's log-probability by approximately
$$\Delta\ell(y')\approx\eta\,\langle\nabla_\theta\ell(y'),\,g_s\rangle,\qquad g_s=\sum_i w_i\,\nabla_\theta\log p(y_i).$$
Here $y_i$ are the step's training examples and $w_i$ their weights:
- **SFT:** $w_i=1/|D|$ for every training example.
- **RL:** $w_i=A_i$, the GDPO advantage recomputed exactly as training computed it.

The inner product is the empirical kernel. It is the coupling.

**Method:**
- **Gradients.** Probe-string gradients at the before-model's weights, using a newly written Kronecker sketch following the description in `PHASE17_PREREG.md` lines 286–293 (Phase 17 D1 never ran, so no code is reused); validated by the ≥ 0.95 sketch check below. Every LoRA-targeted module, $k=8$ per side, seeded.
- **Sketch check.** Exact vs sketched inner products on 50 random pairs in 4 modules must reach Pearson ≥ 0.95.
- **Update directions** $g_s$:
  - S3 and S4: their SFT training completions as trained.
  - S6 and S7: the first 50 groups (400 completions), with exact GDPO advantages.
- **Predicted change:** $\hat\Delta_s(y')=\langle\nabla\ell(y'),g_s\rangle$, up to scale.
- **Lazy check.** Compute the kernel's centered alignment between the before-model and the after-model. Below 0.8, D-1 describes only the start of training, and the write-up says so.

**Predictions [C]:**
- **P-19D-1 (first-order validity).** Per target, compute the Spearman correlation of $\hat\Delta_s$ with the exact $\Delta_s$. The median is above 0, with the CI excluding 0, for S3, S4, S6 and S7.
- **P-19D-2 (RL's reach).** For S6 and S7, define the reach ratio
$$\rho_s=\frac{\text{mean}\,|\hat\Delta_s(y)|\ \text{over recipes the before-model never sampled}}{\text{mean}\,|\hat\Delta_s(y)|\ \text{over recipes it sampled}} .$$
  The median over targets is below 1, with the CI excluding 1: RL's update moves unsampled recipes less than sampled ones.
  - **Usable targets.** A target is usable only if the before-model sampled at least one recipe for it AND at least 3 distinct recipes in its top-30 set have zero count in the before-model's samples. Unusable targets are dropped. Report how many remain per step.
  - **Untestable.** If fewer than 10 usable targets remain for a step, P-19D-2 for that step is reported as untestable, not failed.
  - **[E]** (exploratory, not part of the pass/fail): a length-normalized version ($|\hat\Delta_s(y)|$ divided by the string's token count), and the partial Spearman correlation between $|\hat\Delta_s(y)|$ and unsampled status controlling for string length and gradient norm.
- **P-19D-3 (SFT spillover, from Phase 17 D5).** Use held-out targets absent from S3's and S4's training data (E3 and E3c training sets; overlap checked in Block 0). Overlapping targets are dropped. If fewer than 10 targets remain, P-19D-3 is reported as untestable.
  - Positives: recipes E3 gained that Base never sampled.
  - Negatives: all other recipes.
  - The AUC of $\hat\Delta_{S3}$ for separating them is above 0.5, with the CI excluding 0.5.

**[C-measure], no direction:**
- **Positive vs negative samples.** Split RL's update into its positive-advantage and negative-advantage halves. Report which half narrows the model, measured by the slope of $\hat\Delta$ against $\ell_{\text{before}}$. Two published papers disagree: Ren and Sutherland (ICLR 2025) and Zhu et al. (NeurIPS 2025).
- **Effective dimension.** The participation ratio of the kernel's spectrum at each before-model: how many independent directions the model can move its answers in.

**[E]:** the kernel's block structure by alkali-source family (carbonate, oxide, other), with a 1,000-permutation null.

**Outputs:**
- `results/phase19/kernel_meta.json`
- `results/phase19/kernel_predictions.json`
- `results/phase19/reach.json`
- `results/phase19/spillover.json`
- `results/phase19/figdata_kernel.json` (sorted kernel heatmaps for 3 targets)

---

### Block F1 — how many directions does the change need? (GPU, forward only)

**Method:**
1. **Depth.** Use $L^*=L_{80}$ from Block E, or the last block if there is no $L_{80}$.
2. **Directions.** On half the probe targets, collect $\delta=h_{\text{after}}-h_{\text{before}}$ at $L^*$ at ALL positions of the teacher-forced sequence (prompt positions and answer-string positions), using the same sequence for the before and after models, and take its principal components.
3. **Recovery.** On the other half, run the before-model with $h_{\text{before}}+P_k\delta$ patched at ALL positions of the teacher-forced sequence (prompt and answer string), where $P_k$ projects onto the top $k$ components. $R(k)$ is measured on the answer-string log-probability $\ell(y)$, with the recovery formula of Block E. Report $R(k)$ for $k\in\{1,2,4,8,16,32,64\}$.
4. **Null.** Repeat with 20 random $k$-dimensional subspaces.

**Prediction [C]:**
- **P-19F-1.** $k_{80}$, the smallest $k$ with $R\ge0.8$, is smaller for S6 and S7 than for S3 and S4.

**[E]:** recovery with one global steering vector (the mean $\delta$), per step.

**Output:** `results/phase19/dimension.json`.

---

### Block F2 — dose: scale the change and count recipes (GPU, generation; needs approval)

**Method:**
- **Models.** Scale each step's weight change by $\alpha\in\{0,0.5,1,1.5,2\}$:
  - SFT: one adapter at weight $\alpha$;
  - RL: PEFT `add_weighted_adapter` with weights $[\alpha,1-\alpha]$ on (after, before), `combination_type="cat"`.
- **Sampling.** 32 samples per probe target at the training sampling settings.
- **Counts.** Report:
  - new-recipe share against the before-model's samples, with the Phase 18A noise floor;
  - effective number of recipes $\exp H$ from counts;
  - ASTRAL rule-picked hits.

Report the generation-time estimate before asking for approval.

**Predictions [C]:**
- **P-19F2-1.** For S6 and S7, new-recipe share minus floor stays within its CI of 0 at every $\alpha$ up to 2. Scaling RL up does not create recipes.
- **P-19F2-2.** For S6 and S7, the effective number of recipes falls monotonically over $\alpha\in[1,2]$.
- **P-19F2-3.** For S3 and S4, the new-recipe share rises from $\alpha=0$ to $\alpha=1$.

**Note for the write-up.** $\alpha=2$ is a first-order stand-in for "train longer," not a substitute for it.

**Output:** `results/phase19/dose.json`.

---

## What each outcome means for the paper

- **E holds.** The RL change is written in the last quarter of the network; SFT's starts earlier. This is the location figure.
- **D-2 holds.** Through the network's own coupling, RL's update barely touches recipes the model never samples. This is the mechanism behind 18A's count: the zero-times-anything argument survives in a real network, measured rather than assumed.
- **D-2 fails.** RL does push unsampled recipes through shared weights, but not far enough to cross 18A's noise floor in 300 steps. Then the limit is reach per step, not a hard wall, and the paper says so.
- **D-3 holds.** SFT's new recipes for unseen compounds come from shared gradient directions.
- **F1 and F2.** These say how many directions RL's change uses, and whether scaling it narrows the model further.