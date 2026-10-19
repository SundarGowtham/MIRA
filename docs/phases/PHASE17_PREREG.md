# Phase 17 — instructions and pre-registration: the shape, cause and location of support change

Written 2026-10-16, before any Phase 17 computation. Once Gowtham approves, this file is committed unedited and its commit hash is written as `prereg_commit` into every Phase 17 results JSON. Predictions do not change after the commit. Anything decided later is labelled **[E]** (exploratory) in `docs/phases/PHASE17_RESULTS.md`.

**No training runs.** Every block uses forward passes, gradients at fixed weights, or interventions on existing checkpoints. The one exception (H3) needs Gowtham's explicit approval.

---

## Why this phase exists

Phase 16 settled the behaviour, under two different rewards:

- Practice reweights inside the model's existing range of recipes.
  - E1 (ranker reward, capacity 46.3%) and Phase 12 (checker reward) both reached 14/35 rule-picked hits.
  - Neither cleared RS-SFT's resampling floor for new recipes at step 100, 200 or 300.
- Fine-tuning on the model's own valid samples adds recipes.
  - On ASTRAL, E3 is +11.1 points above Base's floor (95% CI [6.3, 15.8]) and E3c is +8.5 [3.1, 13.9].
  - Choosing rare recipes adds nothing: on held-out targets, E3 − E3c = −0.8 points (95% CI [−3.2, +1.6]).

The same behaviour is already reported on maths:
- *RL Squeezes, SFT Expands*, arXiv 2509.21128;
- *What pass@k Cannot Measure*, NeurIPS 2026;
- *Differential Smoothing*, arXiv 2511.19942.

What has not been measured directly inside one trained model is **why**:
- what shape each training step's change has;
- which gradient terms produce that shape;
- where in the network the change is written.

Phase 17 measures those three things.

### The thesis, stated as geometry

Let $\pi_0$ be a model's distribution over recipes $y$ for a target $x$.

- **Practice is an exponential tilt** (an e-geodesic): $\pi_t(y) \propto \pi_0(y)\,e^{t f(y)}$.
  - A tilt multiplies, so it cannot create mass where there was none: $\operatorname{supp}(\pi_0 e^{f}) = \operatorname{supp}(\pi_0)$.
  - On the log scale it moves every recipe by $t f(y)$, so a recipe can gain at most $t(\max f - \min f)$ nats relative to any other. That is the leash.
- **Fine-tuning on data $q$ is a mixture** (an m-geodesic): $\pi_\lambda = (1-\lambda)\pi_0 + \lambda q$.
  - A mixture adds, so $\operatorname{supp}(\pi_\lambda) = \operatorname{supp}(\pi_0)\cup\operatorname{supp}(q)$.
  - Rare recipes in $q$ gain the most.

The two flows are exactly what natural-gradient (Fisher-metric) descent produces for the two objectives, on the simplex of categorical distributions with logits $\theta$:

$$\text{reward } J=\textstyle\sum_y \pi_\theta(y) r(y):\qquad \dot\theta = G^{-1}\nabla_\theta J = r \;\Rightarrow\; \log\pi_t = \log\pi_0 + t\,r - \psi(t)\quad\text{(e-geodesic)}$$

$$\text{forward KL } \mathrm{KL}(q\,\|\,\pi_\theta):\qquad \dot\theta = -G^{-1}\nabla_\theta \mathrm{KL} \;\Rightarrow\; \pi_t = e^{-t}\pi_0 + (1-e^{-t})\,q\quad\text{(m-geodesic)}$$

Here $G=\operatorname{diag}(\pi)-\pi\pi^\top$ is the Fisher metric in logit coordinates.

Plain (Euclidean) gradient ascent on the logits instead gives

$$\dot\theta_y = \pi(y)\big(r(y)-\bar r\big),$$

so a recipe's gain scales with its current probability. This is the rich-get-richer force of Differential Smoothing, Prop. 3.2.

A real LLM does not move tabular logits. Its recipes are coupled through shared weights (the kernel in Block D). Blocks B and C test which idealised picture each real training step resembles. Block D tests which gradient terms cause the resemblance. Blocks E and F intervene on the network to find where the change lives and how many directions it needs. Block G asks whether a pairwise judge can be a reward at all.

---

## Scope (frozen)

| Block | Question | Kind | Compute |
|---|---|---|---|
| 0 | Inventory, recipe universe | setup | CPU |
| A | Can we read $\log\pi(\text{recipe})$ reliably? | **gate** | GPU, forward |
| B | How does each step's change depend on prior probability and reward? | geometry | CPU |
| C | Does each step move mass to new recipes, or grow it in place? | geometry (unbalanced OT) | CPU |
| D | Which gradient terms produce the shape? | mechanism (empirical kernel) | GPU, backward at fixed weights |
| E | Where in the network is the change written? | causal intervention (patching) | GPU, forward |
| F | How many directions does the change need, and what does scaling it do? | causal intervention (steering, weight dose) | GPU, forward |
| G | Can a pairwise judge be a reward? | structure (Hodge decomposition) | CPU |
| R | Would a principled audit have predicted every training outcome before any GPU was spent? | reward audit, back-tested on six rewards | CPU (+ existing samples) |
| H | Optional, needs approval | — | — |

**Order:** 0 → A (gate G1) → B, C (CPU) → D → E → F. G and R depend on nothing and can start at any time; R uses only existing scored samples and CPU.

After each block, report and append a dated entry to `docs/phases/PHASE17_RESULTS.md`.

---

## Steps under study

A **step** is a pair (model before, model after).

| Step | Before → after | Kind | Checkpoint of "after" |
|---|---|---|---|
| S1 | Base → Corpus SFT | fine-tune, external corpus | `runs/sft-qlora-sft-v3-2nd-rank16/final` |
| S2 | Base → RS-SFT | fine-tune, own filtered samples | `runs/sft-qlora-rs-sft-from-base/final` |
| S3 | Base → E3 | fine-tune, own valid + rare, deduplicated | per `docs/phases/PHASE16_RESULTS.md` |
| S4 | Base → E3c | fine-tune, own valid, deduplicated | per `docs/phases/PHASE16_RESULTS.md` |
| S5 | Corpus SFT → GDPO-300 | practice, checker | `runs/gdpo-qlora-gdpo-v3/checkpoint-300` |
| S6 | RS-SFT → Phase 12 | practice, checker, β = 0 | `runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-{100,200,300}` |
| S7 | RS-SFT → E1 | practice, ranker v2, β = 0 | E1 `checkpoint-{100,200,300}` |

- **Fine-tuning steps:** S1–S4.
- **Practice steps:** S5–S7.
- **Own-sample fine-tuning steps** (the main contrast): S2–S4.

GDPO runs continue their init adapter (`core/model.py::load_with_adapter`, `is_trainable=True`). So a practice step's weight change is $\Delta W = B_1A_1 - B_0A_0$. Block 0 confirms this per run.

## Target sets

- **ASTRAL:** clean 22 primary, full 35 secondary (`results/phase16/astral_overlap.json`).
- **Held-out 40:** `data/novelty/heldout_targets.json`. These are held out from E3 and E3c. Block 0 records whether any were in RS-SFT's or Corpus SFT's training data. Spillover claims use only targets absent from the step's training data.
- **Probe set** for the GPU-heavy blocks D–F: 20 ASTRAL (all from the clean 22) and 20 held-out targets, fixed by seed 20261016 in Block 0 and written to `results/phase17/probe_targets.json`.

## Definitions

- **Recipe** $y$: an unordered precursor set, canonicalised with `precset_of` in `research/support_growth_by_checkpoint.py`. Every script uses the same canonicaliser.
- **Universe** $U(x)$: every distinct recipe in any existing sample for target $x$. This covers all models in the step table, including the Base and RS-SFT 200-sample references.
  - For Blocks D–G, cap at the top 30 by pooled frequency.
  - Always include every recipe any model sampled at least twice.
- **Surface forms.** One recipe is written in several ways (precursor order, amounts). $\log\pi(y)$ is the logsumexp over the up-to-8 most frequent observed surface forms of $y$. This is a lower bound on the set's probability. The top-1-form value is also recorded, as a sensitivity check.
- **Estimators of** $\pi_m(y\mid x)$:
  - **Marginal (primary for A–C):**
    $$\hat\pi_m(y\mid x)=\frac1K\sum_{k=1}^{K}\pi_m\big(y\mid x,\tau_k\big),\qquad \tau_k \sim \pi_m(\cdot\mid x),$$
    where the $\tau_k$ are $K=8$ of the model's own existing reasoning traces for that target. This is unbiased for the probability of the answer, and it removes Phase 15's empty-think caveat (finding 28).
  - **Answer-only (used in D–F for cost):** empty `<think>`, exactly as `research/distributional/dose_response_teacher_forcing.py` does it. Only the precursor array span is scored, never the shared prefix.
- **Step change:** $\Delta_s(y)=\log\pi_{\text{after}}(y)-\log\pi_{\text{before}}(y)$.
- **Recipe reward** $r_s(y)$: the reward the step itself trained on, averaged over the existing scored samples that produced $y$ (pooled across models).
  - S5 and S6: the 5-channel checker aggregate.
  - S7: ranker v2.
  - S1–S4: the checker aggregate, used as a covariate.
  - Report coverage: the share of $U(x)$ with at least one scored sample.
  - Within-target advantage: $\tilde r(y)=r(y)-\sum_{y'}\tilde\pi_{\text{before}}(y')\,r(y')$, with $\tilde\pi$ renormalised on $U(x)$.
- **Effective support:** $N_{\text{eff}}(\pi)=\exp H(\tilde\pi)$ on $U(x)$.
  - **$\varepsilon$-support:** $|\{y:\pi(y)>\varepsilon\}|$ with $\varepsilon=1/200$, one draw in a 200-sample reference.
- **Fisher–Rao distance** on $U(x)$ (renormalised): $d_{FR}(p,q)=2\arccos\sum_y\sqrt{p(y)\,q(y)}$. Under the map $y\mapsto\sqrt{p(y)}$, the simplex becomes the positive part of a unit sphere and $d_{FR}$ is arc length. Support boundaries are the coordinate planes.

---

## Block 0 — inventory (CPU)

1. For every model in the step table, record:
   - adapter path and storage (merged or stacked; continued from init or fresh);
   - intermediate checkpoints, if any exist for S1–S4;
   - the LoRA config.
2. For every model and target set, find the files that hold raw completions with `<think>` traces and per-sample checker (and, for E1, ranker) scores. Likely locations: `data/novelty/*.jsonl`, `results/astral_gen_n32_*.json`, `runs/*/generations.jsonl`.
3. Record training-set membership of every ASTRAL and held-out target, per step.
4. Build $U(x)$ and the probe set.

**Stop and report** if any model lacks raw completions for either target set.

**Outputs:**
- `results/phase17/inventory.json`
- `results/phase17/universe.jsonl` (target, recipe, surface forms with counts, per-model sample counts, reward stats)
- `results/phase17/probe_targets.json`

## Block A — the instrument (GPU, forward only) — GATE

**What.** Compute $\log\hat\pi_m(y\mid x)$ under both estimators.
- **Models (13):** Base, Corpus SFT, RS-SFT, E3, E3c, GDPO-300, Phase 12 @100/200/300, E1 @100/200/300.
- **Recipes:** every $y\in U(x)$.
- **Targets:** ASTRAL 35 + held-out 40.
- **Cost control:** prefill each reasoning trace once and score all continuations from the cached prefix.
- Estimate the cost first, then run in tmux.

**Gate G1 (instrument validity).**
- Hold out the samples not used as $\tau_k$.
- Per model and target, compute the Spearman correlation between $\hat\pi_m(y\mid x)$ and the empirical frequency of $y$ among the held-out samples.
- **Pass:** the median over targets is ≥ 0.5 for every model under the marginal estimator.
- Report the same number for the answer-only estimator. It says how far Phase 15's empty-think numbers can be trusted, and how much to trust D–F.
- **If G1 fails, stop.** Report, and do not run B–F.

**Consistency check.** The count of recipes entering the $\varepsilon$-support per step must agree in sign with Phase 16's support growth (`results/support_growth_by_checkpoint.json`, `results/phase16/e3_e3c_final_battery.json`). A disagreement is reported before anything else.

**Outputs:**
- `results/phase17/logp_universe.parquet` (target, recipe, model, estimator, logp, logp_top1form, n_forms, n_tokens)
- `results/phase17/instrument_check.json`

---

## Block B — the shape of each step (CPU)

### B1. Signature regression

For each step $s$, over all $(x,y)$ with $y\in U(x)$, using the marginal estimator, fit

$$\Delta_s(y)=a_x+b_s\,\tilde r(y)+c_s\,\log\pi_{\text{before}}(y)+d_s\,\tilde\pi_{\text{before}}(y)\,\tilde r(y)+\epsilon .$$

- $a_x$ are target fixed effects.
- Use a cluster bootstrap by target: 10,000 resamples, seed 20261016.

What each idealised geometry predicts:

| Geometry | $b$ | $c$ | $d$ |
|---|---|---|---|
| m-flow toward data (fine-tuning) | ≈ 0 | **< 0** (rare training recipes gain most) | ≈ 0 |
| Fisher e-flow (natural-gradient practice) | **> 0** | 0 | 0 |
| Euclidean practice (plain gradient on logits) | > 0 | 0 | **> 0** (gain ∝ probability × advantage) |

**Predictions [C]:**
- **P-B1a.** For S2, S3 and S4: $c_s<0$, with the 95% CI excluding 0.
- **P-B1b.** For S5, S6 and S7: $b_s>0$ with the CI excluding 0, and the CI of $c_s$ not lying entirely below 0.
- **P-B1c.** The CI of $c_{S3}-c_{S4}$ includes 0. Both steps deduplicate their training recipes, so both pull toward the same flat target over valid recipes. This would explain the Phase 16 E3/E3c null.

**Measured, no direction pre-registered [C-measure]:**
- the sign of $d_s$ for S5–S7, i.e. whether practice followed the Fisher geometry or the Euclidean one. Adam is approximately a diagonal preconditioner, so either is plausible;
- $c_{S1}$.

**Does not support the thesis:**
- $c_s<0$ (CI) for any practice step; or
- the CI of $c_s$ lying above 0 for any own-sample fine-tuning step.

**Curve fit.** Also fit the m-flow curve directly for S2–S4 on recipes in each step's training set:

$$\Delta(y)=\log\!\big(1-\lambda+\lambda\,q(y)/\pi_{\text{before}}(y)\big).$$

- Here $q$ is uniform over that target's training recipes and $\lambda$ is estimated.
- Report $\lambda$ and $R^2$, and compare against the e-flow alternative $\Delta=t\,\tilde r-\psi$.

### B2. Geodesic test along checkpoints

Applies to S6 and S7, and to S1–S4 if Block 0 finds intermediate checkpoints. Per target, take $\pi_0,\pi_{100},\pi_{200},\pi_{300}$ on $U(x)$ and fit two paths, with $s_k$ free:

- **e-path:** $\log\pi_k=(1-s_k)\log\pi_0+s_k\log\pi_{\text{end}}-\psi_k$
- **m-path:** $\pi_k=(1-s_k)\pi_0+s_k\pi_{\text{end}}$

Score each fit by the Fisher–Rao distance from the fitted point to $\pi_k$, and take $\rho=\mathrm{RSS}_e/\mathrm{RSS}_m$.

- **Prediction [C]:** median $\rho<1$ for S6 and S7. Practice trajectories are closer to e-geodesics.
- **[E]:** $\rho$ for fine-tuning steps, if checkpoints exist.

### B3. The leash

- From B1's $b_s$, the leash length for practice is $L_s=b_s\cdot(\max\tilde r-\min\tilde r)$: the largest relative gain a tilt of that strength allows.
- Report $L_s$ and the observed largest relative gain among recipes that started below $\varepsilon$.
- Count recipes that cross from below $\varepsilon$ to above it, per step. Compare against a noise floor: the same count between two independent halves of the $\tau_k$ for the "before" model.

**Prediction [C]:** practice steps are within the noise floor; S3 and S4 are above it.

### B4. Data for figures (no predictions)

- **Ternary:** for targets with an alkali element, the probability mass on carbonate, bare oxide and other alkali sources. Families come from `classify_alkali_source` in `research/distributional/alkali_source_share.py`. One point per model, and one arrow per step.
- **Fisher sphere:** the same three-family distributions mapped by $\sqrt{\pi}$ onto the unit sphere's positive octant. Each step is an arc of length $d_{FR}$.
- **Signature scatter:** $\Delta_s$ against $\log\pi_{\text{before}}$ per step, with the three theory curves.

**Outputs:**
- `results/phase17/signature.json` (coefficients, CIs, curve fits)
- `results/phase17/geodesic.json`
- `results/phase17/leash.json`
- `results/phase17/figdata_ternary.json`
- `results/phase17/figdata_sphere.json`

---

## Block C — transport or growth (CPU, unbalanced optimal transport)

**Why.** A tilt can only grow or shrink mass where mass already is. Creating new support means moving mass to new places. Unbalanced optimal transport splits any change of a measure into exactly these two parts:
- **reaction:** growth in place;
- **transport:** movement across recipe space.

This is the Wasserstein–Fisher–Rao (Hellinger–Kantorovich) split. Its reaction part is the Fisher–Rao geometry of Block B.

**Method.** Per target, set up:
- **Measures:** $\mu=\pi_{\text{before}}$ and $\nu=\pi_{\text{after}}$ restricted to $U(x)$. Do not renormalise them: total mass on $U$ may change, and that is information.
- **Ground cost:** $C(y,y')$ = Jaccard distance between precursor sets. As a check, also use the number of precursor substitutions.
- **Solver:** entropic unbalanced OT (POT `ot.unbalanced.sinkhorn_unbalanced`) with KL marginal relaxation $\tau\in\{0.05,0.1,0.5\}$ and entropic regulariser $10^{-2}$.
- **Transport share:**
  $$T=\frac{\langle C,P\rangle}{\langle C,P\rangle+\tau\,\mathrm{KL}(P\mathbf 1\|\mu)+\tau\,\mathrm{KL}(P^\top\mathbf 1\|\nu)}.$$

**Prediction [C]:** at every $\tau$, the median $T$ over targets is higher for S2–S4 than for S5–S7. The cluster-bootstrap CI on the difference of medians must exclude 0.

**[E]:**
- The distance histogram of transported mass for fine-tuning steps. Does new mass arrive from one-precursor swaps (short moves) or from unrelated sets?
- The mass arriving at recipes that started below $\varepsilon$ ("support-creation mass"), per step.

**Outputs:** `results/phase17/transport.json`, `results/phase17/figdata_transport.json` (per-target plans for 6 example targets, with a 2-D MDS layout of $U(x)$ under $C$).

---

## Block D — the cause: learning dynamics through the empirical kernel (GPU)

**Why.** Block B shows the shape. Block D shows which gradient terms produce it. One small gradient step changes log-probabilities by (Ren & Sutherland, ICLR 2025)

$$\Delta\log\pi(y')\approx\eta\,\big\langle\nabla_\theta\log\pi(y'),\,g\big\rangle,\qquad g=\sum_i w_i\,\nabla_\theta\log\pi(y_i).$$

So every training method is a choice of weights $w_i$ on one shared object, the empirical kernel

$$K(y',y_i)=\langle\nabla_\theta\log\pi(y'),\,\nabla_\theta\log\pi(y_i)\rangle .$$

- **SFT on a set $D$:** $w_i=1/|D|$ for every example, whatever its probability. Equivalently, an on-policy gradient with importance weight $1/\pi$ (Wu et al., ICLR 2026).
- **GDPO:** $w_i=A_i$, the advantage, on the model's own samples. It splits into positive-sample reinforcement (PSR, $A_i>0$) and negative-sample reinforcement (NSR, $A_i<0$) (Zhu et al., NeurIPS 2025).

**D1. Gradients.**
- **Kernel:** a Kronecker-sketched full-weight kernel over the LoRA-targeted modules, LoGra-style (Choe et al. 2024).
  - For each linear module, project the input activations $x_t$ and output gradients $\delta_t$ with seeded Gaussian matrices $R_{\text{in}},R_{\text{out}}$ ($k=8$ each).
  - Accumulate $\sum_t(R_{\text{out}}^\top\delta_t)(R_{\text{in}}^\top x_t)^\top$.
  - The full gradient is never materialised. The sketch is 64 dims × number of modules.
  - Why this kernel: it is identical across steps, so comparisons between steps are fair. For a fresh LoRA adapter ($B=0$), the LoRA kernel equals this kernel up to a random input projection.
- **Secondary, for S6 and S7 only:** the exact LoRA-parameter kernel, sketched per module.
- **Sketch check:** compare exact and sketched inner products on 50 random pairs for 4 modules. Require Pearson ≥ 0.95.
- **Weights:** the step's "before" model.
- **Probe gradients:** probe-set targets, top 30 recipes each, answer-only.
- **Update gradients:**
  - S3 and S4: their training recipes, full completions as trained.
  - S6 and S7: 50 training groups (400 completions) from steps 1–50 of each run. Recompute advantages exactly as GDPO computed them: per-check z-normalisation within the group, then summed.

**D2. Lazy-regime check** (Malladi et al., ICML 2023).
- Compute the centred kernel alignment between $K$ at the "before" weights and at the "after" weights, and at 100 and 200 for S6 and S7.
- CKA ≥ 0.8 means the linearised picture describes the whole step. Below that, D3–D5 describe only its first steps, and the write-up says so.

**D3. First-order predictions.** For each step, compute the predicted change $\hat\Delta_s(y')=\sum_i K(y',y_i)w_i$ (up to scale), and the per-target Spearman correlation with the actual $\Delta_s$ (Block A, answer-only).
- **Prediction [C]:** median Spearman > 0, with the CI excluding 0, for S3, S4, S6 and S7.

**D4. Which half of practice narrows.** Two published hypotheses disagree, so no direction is pre-registered. Compute B1's slope $c$ and interaction $d$ on $\hat\Delta$ for PSR only, NSR only, full GDPO, and SFT.
- Ren & Sutherland's squeezing effect predicts that NSR narrows ($c$ or $d$ > 0).
- Zhu et al. predict that PSR narrows and NSR preserves breadth.
- Report which holds here.
- **[C] for SFT:** $c(\hat\Delta_{\text{SFT}})<0$.

**D5. Spillover: why E3 writes new recipes for compounds it never trained on.** Use held-out targets absent from S3's training data.
- Positives: recipes absent from Base's 200-sample reference that E3 gains ($\Delta>0$ and $\pi_{\text{after}}\ge\varepsilon$).
- Negatives: all other recipes.
- Compute the AUC of $\hat\Delta_{S3}$ for separating them.
  - **Prediction [C]:** AUC > 0.5, with the CI excluding 0.5.
- **[E]:** kernel block structure. Compare mean $K$ between recipes sharing an alkali/alkaline-earth source family with mean $K$ across families, using a permutation null (1,000). This is the chemistry in the kernel.

**D6 [E].** The spectrum of $K$ at each "before" model, reported as a participation ratio. $K=JJ^\top$ and the pullback Fisher $J^\top J$ share their nonzero eigenvalues, so this counts the directions the model can move its answers in.

**Outputs:**
- `results/phase17/kernel_meta.json` (sketch check, CKA)
- `results/phase17/kernel_predictions.json`
- `results/phase17/psr_nsr.json`
- `results/phase17/spillover.json`
- `results/phase17/figdata_kernel.json` (a sorted kernel heatmap for 3 targets)

---

## Block E — location: activation patching (GPU, forward)

**Why.** This block is causal. Copy the trained model's internal state at layer $L$ into its starting model. If the change in $\log\pi$ appears, the change is written at or below layer $L$.

**Method.** For each step and each layer $L\in\{0,2,4,\dots,36\}$, on the probe set, answer-only:
1. Run the "before" model.
2. Replace its residual stream after layer $L$ at every prompt and prefix position with the "after" model's.
3. Score $U(x)$ continuations. The continuation positions run in the "before" model.
4. Compute recovery, pooled over probe targets:

$$R_s(L)=1-\frac{\sum_{x,y}\big(\Delta^{\text{patch}}_s(y)-\Delta_s(y)\big)^2}{\sum_{x,y}\Delta_s(y)^2}.$$

Also run the reverse patch (the "after" model with the "before" state). $L_{50}$ and $L_{80}$ are the first layers with $R\ge0.5$ and $R\ge0.8$.

**Predictions [C]:**
- **P-E1.** $L_{50}$ of each practice step (S6, S7) is later than $L_{50}$ of each own-sample fine-tuning step (S3, S4). That is four pairwise orderings, all required.
- **P-E2.** $L_{50}(S6)$ and $L_{50}(S7)$ are ≥ 27 (three-quarters of depth).

This is consistent with the 2026 fixed-SAE result: RL drift was small and concentrated in late layers. It is also consistent with finding 13: SFT's rotation grows from about layer 10.

**E2 [E], visual.** Apply the logit lens (final norm plus unembedding at each layer) to the three alkali-family strings, for layers ≥ 18, per model. This gives a per-layer path across the ternary simplex.

**Outputs:** `results/phase17/patching.json`, `results/phase17/figdata_lens.json`.

---

## Block F — dimension and dose (GPU, forward)

### F1. How many directions does a step need?

At layer $L^*=L_{80}$ of each step (the last layer if there is no $L_{80}$):
1. Collect the differences $\delta=h_{\text{after}}-h_{\text{before}}$ at all prefix positions on half the probe targets, and compute the PCA.
2. On the other half, intervene with $h_{\text{before}}+P_k\delta$ at every prefix position, where $P_k$ projects onto the top $k$ components.
3. Measure the recovery $R(k)$ for $k\in\{1,2,4,8,16,32,64\}$.

- **Null:** 20 random $k$-dimensional subspaces.
- **Global steering vector:** add the single mean $\delta$, the same for every target.

**Predictions [C]:**
- $k_{80}$ (the smallest $k$ with $R\ge0.8$) is smaller for S6 and S7 than for S3 and S4.

**[E]:**
- The global vector's recovery per step. "Is this training step a steering vector?"

### F2. Weight dose-response

Build $W(\alpha)=W_{\text{before}}+\alpha\,\Delta W_s$ for $\alpha\in\{0,0.25,0.5,0.75,1,1.25,1.5,2\}$ with PEFT `add_weighted_adapter(..., combination_type="cat")`. That combines $\Delta W$ exactly:
- for practice steps, use weights $[\alpha,\,1-\alpha]$ on (after, before) adapters;
- for fine-tuning steps, use weight $[\alpha]$ on the single adapter.

On the probe set, answer-only, report as functions of $\alpha$:
- $N_{\text{eff}}$;
- $\varepsilon$-support;
- alkali-carbonate share;
- the probability of the ASTRAL rule-picked set.

**Predictions [C]:**
- For S6 and S7, $N_{\text{eff}}$ decreases monotonically over $\alpha\in[1,2]$: extrapolating practice keeps narrowing.
- For S3 and S4, $N_{\text{eff}}(1)>N_{\text{eff}}(0)$.

**[E]:** whether $\alpha=2$ raises rule-picked probability further. This is a first-order stand-in for "train longer", not a substitute for it, and the write-up says so.

**Outputs:** `results/phase17/steering_dim.json`, `results/phase17/dose.json`.

---

## Block G — can a pairwise judge be a reward? Hodge decomposition (CPU)

**Why.** GDPO needs one number per sample.
- With a scalar reward, the pairwise differences $r(a)-r(b)$ form a gradient flow on the comparison graph, which has zero curl by construction.
- A pairwise judge need not have zero curl.
- `core/comparator.py::compare` sums only the channels gradeable for *both* routes in a pair. The channel set therefore changes from pair to pair, and that alone can create cycles ($a\succ b\succ c\succ a$) that no scalar reward can express.

HodgeRank (Jiang, Lim, Yao & Ye, Math. Programming 2011) splits any edge flow orthogonally:

$$Y=\operatorname{grad}s\;+\;\operatorname{curl}^{*}\Phi\;+\;h .$$

The gradient part is the best scalar ranking, the curl part is local cycles, and $h$ is global cycles. $h$ vanishes on a complete graph.

**G1. Method.**
- **Graph:** per target (ASTRAL 35 + held-out 40), the top-30 recipes of $U(x)$. Each recipe is represented by its most frequent full parsed route.
- **Edges:** all pairs with a defined margin. $Y_{ab}$ is the comparator margin (current `COMPARATOR_VERSION`, existing scales).
- **Decomposition:** solve weighted least squares for $s$, then project the residual onto the triangle (curl) space and the harmonic remainder.
- **Report:** energy fractions $\|\cdot\|^2/\|Y\|^2$ per target and pooled; cyclic-triad counts.
- **Controls:**
  1. Checker-score and ranker-v2-score differences. Curl + harmonic must be 0 to numerical precision. This is the instrument check.
  2. An intervention: a fixed-channel comparator wrapper in `research/`, in which every pair uses the same channel set (a missing channel enters at percentile 0.5). **Do not edit `core/comparator.py`.**

**Predictions [C]:**
- The comparator's non-gradient fraction (curl + harmonic) is ≥ 10% pooled.
- The fixed-channel variant cuts it by at least half.
- Mean per-triangle curl is at least 1.5× higher on triangles whose three comparisons used different channel sets.

**G2 [E]. Real lab preferences (ARROWS³ YBCO).**
- Flow: $Y_{ab}$ = the mean over shared temperatures of $\operatorname{sign}(\text{yield}_{a,T}-\text{yield}_{b,T})$, weighted by the number of shared temperatures. Load data with `research/distributional/arrows_gate_scoring.py`.
- Compare the non-gradient fraction against an additive null, $\text{yield}=\text{recipe}+\text{temperature}+\text{noise}$, fitted to the data and simulated 1,000 times with the same coverage.
- No direction is predicted.
- If lab preferences carry curl beyond the null, no scalar recipe-level reward can represent them, which is a structural limit on scalar RLVR in this domain.

**Outputs:** `results/phase17/hodge_comparator.json`, `results/phase17/hodge_arrows.json`, `results/phase17/figdata_hodge.json` (3 example graphs with their curl triangles).

---

## Block R — the reward audit: properties a reward must have before it trains anything (CPU)

**Why.** The project trained first and diagnosed afterwards. Reward capacity was measured after run 1; the temperature hack was found by reading outputs; the carbonate shift was found after Phase 12. Every one of these is a property of the reward and the starting policy's own samples, so it was computable **before training**. Block R defines eight properties, each tied to a piece of mathematics, computes them for every reward the project built, and checks them against what training later showed.

This is a **back-test**: the outcomes are already known. To keep it honest, every definition and threshold below is fixed in this file before anything is computed, and the write-up calls it retrospective.

### Rewards audited

| Id | Reward | Policy whose existing samples are scored | Outcome to back-test against |
|---|---|---|---|
| R1 | Checker, 10 channels (run 1) | Corpus SFT | run 1 capacity, dead channels, temperature hack (findings 1, 6) |
| R2 | Checker, 5-channel RUN3 aggregate | RS-SFT | Phase 12 shifts (findings 22, 24–26), Phase 16 checkpoints |
| R3 | Low-temperature objective (`temperature_economy` added) | Corpus SFT (hardening-probe samples) | finding 15 |
| R4 | Ranker v2 | RS-SFT | E1 (findings 30, 32) |
| R5 | Comparator (`core/comparator.py`) | RS-SFT | never trained; ARROWS³ below chance (finding 27) |
| R6 | Novelty reward (Phase 16) | E3 | E4 never ran; audit only |

Group size $G=8$ throughout, matching training.

### R-a. Symmetry: the axioms a chemistry reward must satisfy

A reward should not change when the chemistry does not. Each axiom becomes a metamorphic test: transform a route in a way that leaves the chemistry unchanged, and require the score to stay the same.

| Axiom | Transformation | Bug it would have caught |
|---|---|---|
| Order invariance | permute precursors; permute independent operations | — |
| Scale invariance | multiply all amounts by a constant | amounts 1 : 1 vs 0.5 : 0.5 scored as maximally wrong |
| Notation invariance | equivalent formula spellings (hydrate dot vs no dot, parentheses), atmosphere synonyms | hydrate middle dot broke a whole chemical system |
| Token identity | atmosphere strings that contain, but are not, a reducing gas (`h2o`, `co2`) | `h2` inside `h2o` counted as reducing |
| Monotonicity | append an unrecognised step | unknown steps scored as a perfect order |
| Abstention | a check that cannot be computed must return None, not a number | the 0.5 sentinel paid as reward |
| Boundary | a stability value of exactly 0.0 | 0.0 treated as missing |

**Method.**
- Build a fixed suite: 200 routes drawn from existing samples, with every transformation applied.
- Run it on the current validator, and on each historical `VALIDATOR_VERSION` reachable in git. Use a read-only `git worktree` for each, never a checkout of the working tree.
- Report violations per axiom per version.

**Prediction [C, back-test].**
- The suite flags at least 4 of the 6 bugs in the table, at the versions where they existed.
- It flags none on the current version.

### R-b. Signal: Fisher's fundamental theorem

Under natural-gradient (replicator) dynamics, expected reward rises at a rate equal to the reward's variance under the policy:

$$\frac{d}{dt}\,\mathbb E_\pi[r] = \operatorname{Var}_\pi(r).$$

This is Fisher's fundamental theorem of natural selection (1930); replicator dynamics are the gradient flow of the Fisher metric (Shahshahani 1979). So "reward capacity" is not a heuristic. It is the speed at which the reward can move the policy.

GRPO and GDPO divide each group by its standard deviation. Every group that is not a tie therefore contributes about equally, and the speed reduces to the **share of non-tied groups**. With $G=8$ and the project's ddof convention, a z-normalised non-tied group has variance exactly $7/8$, so a channel's capacity is exactly $0.875\times$ its share of live groups.

For a pass/fail check with per-target pass rate $p_x$, a group of $G$ samples is a tie with probability

$$P_{\text{tie}}(x)=p_x^{\,G}+(1-p_x)^{\,G},$$

so the dead-group rate is predictable from pass rates alone.

**Method.**
- Per reward and channel, compute: capacity (as `probe_hardening.capacity_metrics`), share of live groups, predicted tie rate from per-target pass rates, and observed tie rate.
- Write out the design rule: a channel is informative only where $p_x$ is far from 0 and 1. Report the share of targets with $p_x\in[0.1,0.9]$.

**Prediction [C, back-test].** For binary channels, the predicted and observed tie rates agree within 5 percentage points.

### R-c. Direction: the Price equation

Selection changes the mean of any feature $\phi$ of a recipe in proportion to its covariance with fitness (Price 1970):

$$\Delta\,\mathbb E[\phi]\;\approx\;\eta\,\operatorname{Cov}_\pi\big(A,\phi\big).$$

Here $A$ is the within-group advantage. A reward's **selection differential** on a feature therefore predicts, before training, which way practice will push that feature. Hacks included.

**Features:**
- alkali carbonate used;
- bare alkali oxide used;
- ammonium precursor used;
- number of precursors;
- maximum reported temperature;
- recipe absent from the policy's 200-sample reference (novelty);
- matches the ASTRAL rule-picked set.

**Method.**
- Per reward, compute the standardised differential $S_\phi=\operatorname{Cov}(A,\phi)/\operatorname{sd}(\phi)$, pooled over groups, with a cluster bootstrap by target.
- Back-test against the observed shift in the trained model: R1 → GDPO-300, R2 → Phase 12, R4 → E1.

**Prediction [C, back-test].**
- Among (reward, feature) pairs with $|S_\phi|\ge0.1$, the sign of $S_\phi$ matches the sign of the observed shift in at least 75%.
- Two cases named in advance:
  - R2 predicts the carbonate → bare-oxide shift and a temperature rise.
  - R4 predicts a temperature *fall* (its `temperature_economy` channel), opposite to R2.

### R-d. Consistency: zero curl

This is Block G, restated as a property. A scalar reward is curl-free by construction. A pairwise judge must be checked.

- R1, R2, R4 and R6 are scalar and serve as the zero-curl control.
- R5 is the test case.

No new computation here; Block G's numbers enter the report card.

### R-e. Aggregation: how the channels are combined

GDPO z-normalises each channel within its group and sums them. The aggregate behaves like a voting rule over channels, and its failures are the familiar ones of social choice and robust statistics.

**Measured, per reward:**
- **Effective number of channels:** the participation ratio of the within-group channel correlation matrix,
  $$\mathrm{PR}=\Big(\sum_k\lambda_k\Big)^2\Big/\sum_k\lambda_k^2 .$$
- **Conflict:** the mean pairwise cosine between channels' within-group advantage vectors (negative means the channels pull against each other).
- **Dominance:** the share of live groups in which one channel supplies more than half of $\sum_k|A_k|$.
- **Noise amplification:** the share of live groups in which a channel's within-group range is below its resolution, yet z-normalisation scales it to unit variance. Resolution: $10^{-3}$ for continuous channels, as stored.
- **Scale-estimator breakdown:** for any MAD- or percentile-scaled channel, the share of calibration values tied at the median. Above 50%, MAD collapses toward 0; this was finding 23's `C7_gas_evolution`.
- **Gates:** for rewards that multiply gates by objectives (R4, R5), Shapley attribution of each pairwise decision to gates vs channels. This replaces Phase 15's "gate-decided vs channel-decided" split with an exact decomposition.

**Prediction [C, back-test].**
- R5's `C7` calibration set shows more than 50% median ties.
- At least one channel of R1 has noise-amplified live groups above 10%.

### R-f. Validity: how much the reward knows about the lab

By the data-processing inequality, an update driven by the reward can carry no more information about true quality $Q$ than the reward itself does:

$$I(\text{update};Q)\;\le\;I(R;Q).$$

What matters is the information beyond what a trivial rule already gives: $I(R;Q\mid\text{carbonate-free})$.

**Method.**
- Data: ARROWS³ YBCO pairs (the Phase 15 endpoint), with $Q$ = which recipe of the pair had higher yield, and $R$ = which one the reward prefers.
- Compute $I(R;Q)$ and $I(R;Q\mid\text{carbonate-free})$ in bits, with a bootstrap CI, for R2, R4, R5 and the carbonate-free baseline.
- Do the same on ASTRAL's 35 pairs, flagged as small.

**Prediction [C, back-test].** For R2, R4 and R5, the CI of $I(R;Q\mid\text{carbonate-free})$ includes 0. This is finding 27 restated in information-theoretic units.

### R-g. Goodhart curve: best-of-n as a free simulation of practice

Picking the best of $n$ samples by a reward is a cheap stand-in for optimising against it. Its KL from the policy is at most $\log n-\frac{n-1}{n}$ (Beirami et al. 2024).

**Method.**
- Per ARROWS³ target, draw $n\in\{1,2,4,8,16\}$ recipes at random from the measured recipe pool (2,000 draws per $n$).
- Select the best by each reward, and record the measured yield of the pick.
- Plot mean yield against the KL bound, for R2, R4, R5, the carbonate-free rule, random selection, and the oracle.

**Prediction [C, back-test].** At every $n$, the CIs of R2 and R4 overlap the carbonate-free rule's.

**[E]:** whether any reward's curve turns down at large $n$, the over-optimisation shape of Gao, Schulman & Hilton (2023).

### R-h. Reachability: how much of the reward lives where practice cannot go

On-policy practice tilts; it cannot reach recipes the policy does not sample (thesis, Block B3).

**Method.**
- Per target, score every recipe in $U(x)$ with the reward, and take $\pi_{\text{init}}$ from Block A.
- Compute the **unreachable reward share**: the share of recipes in the reward's top decile that the init assigns $\pi<\varepsilon$.
- Compute the **tilt ceiling**: the largest gain in log-probability a tilt of the fitted strength $b_s$ (B1) can give such recipes.
- Back-test on Phase 12's four gained targets (ledger `L6.gains`): was the gained rule-picked set above or below $\varepsilon$ under RS-SFT's teacher-forced probability? RS-SFT never sampled it in 32 draws.

**[E]:** no direction is predicted for the four-target check. If any of the four was below $\varepsilon$, practice reached a recipe the thesis says it should not, through shared weights, and that is reported prominently.

### R-i [E]. Learnability and landscape

- **Kernel–reward alignment** (Cristianini et al. 2002), using Block D's kernel $K$ at the init: $\mathrm{KTA}=\langle K,\tilde r\tilde r^\top\rangle_F/(\|K\|_F\,\|\tilde r\tilde r^\top\|_F)$.
  - Also report the share of $\tilde r$'s variance in the top-$k$ eigenvectors of $K$.
  - Rewards concentrated in large-eigenvalue directions are learned first (spectral bias; Canatar, Bordelon & Pehlevan 2021).
- **Landscape on the recipe graph:** recipes are nodes, and edges join recipes one precursor substitution apart.
  - Count the reward's local maxima and the persistence of its superlevel-set components (0-dimensional persistent homology).
  - This is meaningful only if Block C finds that mass moves locally.

### The report card (the reward-design figure)

Rows are R1–R6; columns are R-a to R-h. Each cell is the measured value, coloured by pass or fail against the thresholds below. They are fixed now and are the design rules the paper proposes for future verifiers.

| Property | Pass if |
|---|---|
| R-a symmetry | 0 axiom violations |
| R-b signal | ≥ 50% of targets have $p_x\in[0.1,0.9]$ on at least one channel, and capacity ≥ 40% |
| R-c direction | no differential with $|S|\ge0.1$ on a feature the designer did not intend to reward (max temperature, number of precursors) |
| R-d consistency | curl + harmonic < 5% of flow energy |
| R-e aggregation | PR ≥ 2, noise-amplified groups < 5%, no scale estimator above 50% ties |
| R-f validity | $I(R;Q\mid\text{carbonate-free})$ CI excludes 0 |
| R-g Goodhart | the BoN curve beats the carbonate-free rule at some $n$ (CI) |
| R-h reachability | unreachable reward share < 50% |

**Outputs:**
- `results/phase17/audit_symmetry.json`, `audit_signal.json`, `audit_price.json`, `audit_aggregation.json`, `audit_validity.json`, `audit_bon.json`, `audit_reach.json`, `audit_landscape.json`
- `results/phase17/report_card.json`
- `research/phase17_reward_audit.py`: one entry point, which future rewards also run through.

## Block H — optional (each needs Gowtham's explicit OK)

- **H1. Weight geometry (CPU).** Extend `research/lora_spectral_analysis.py` to measure, per module:
  - the alignment of $\Delta W_s$ with the top-$k$ singular subspaces of $W_{\text{base}}$ ($k=16,64,256$);
  - the principal angles.

  *The Path Not Taken* (arXiv 2511.08567) predicts lower alignment for practice. It studied full fine-tuning; under LoRA this is open. [E]
- **H2. Causal abstraction** (distributed alignment search, Geiger et al.). Learn a low-dimensional subspace at $L^*$ that mediates alkali-source choice. Test whether practice changes the variable's value or the mechanism that reads it. [E]
- **H4. Local learning coefficient** (singular learning theory; `devinterp`'s SGLD estimator on the LoRA parameters). Is the effective complexity of the solution changed more by fine-tuning steps than by practice steps? [E]
- **H5. Fisher-metric steering dimension.** Repeat F1 with directions ranked by the readout's pullback Fisher $J_h^	op F J_h$ instead of Euclidean PCA. This asks whether practice moves only the activation directions the output is most sensitive to. [E]
- **H3. Counterfactual data ablation.** Retrain E3 without the 10% of training recipes with the highest kernel influence on one held-out target group. Does spillover to that group drop? This is a small SFT run (about an hour). **Training: only with approval.**

---

## Rules restated

- No training, except H3 with approval.
- Do not edit `validator.py` or `core/comparator.py`; wrappers go in `research/phase17_*.py`.
- Every bootstrap clusters by target, with 10,000 resamples and seed 20261016.
- ASTRAL claims quote the clean 22 first.
- **New files only.** Never write over an existing tracked results file (the 2026-10-15 `astral_gen_n32_rs_sft.json` incident).
- Long jobs run in tmux with `tee` and the ntfy pattern, in user-space environments. No `sudo`, no `apt`.
- Every results JSON carries `prereg_commit`.
- A failed prediction is reported as failed, in the words it was pre-registered in.
- Do not commit or push. Gowtham commits.

## What each outcome means for the paper

- **B1, C, E and F hold.** Practice is a tilt: written late, in few directions, and support-preserving. Fine-tuning is a mixture: written deeper, in more directions, and it moves mass to new recipes. This becomes the mechanism section.
- **$d_s>0$ for practice.** Practice follows Euclidean, not Fisher, geometry, so rich-get-richer is the operative force (as in Differential Smoothing). Natural-gradient or probability-corrected practice becomes the named fix.
- **D4 resolves PSR against NSR.** The domain answers a live disagreement in the RLVR literature.
- **G holds.** The comparator was never a reward, structurally, which is independent of its below-chance score on ARROWS³.
- **R's back-tests hold.** The paper's reward-design section becomes eight properties with their mathematics, a report card for six rewards, and the claim that the audit, run before training, predicts what training did. If the back-tests fail, the section reports which properties did not predict the outcome.
- **Any [C] prediction fails.** That result is reported in the paper as a failed prediction.

## Figure map (built by regular Claude from the `figdata_*` files)

| Figure | Block | One-sentence claim it carries |
|---|---|---|
| Signature | B1 | Practice shifts recipes by reward; fine-tuning lifts the rare ones. |
| Ternary + Fisher sphere | B4 | Practice arrows run to a corner; fine-tuning arrows move inward. |
| Transport map | C | Fine-tuning moves mass to neighbouring recipes; practice grows it in place. |
| Kernel heatmap + spillover | D | New recipes for unseen compounds come from shared gradient directions. |
| Patching curve | E | Practice is written in the last quarter of the network. |
| Dimension curve + dose | F | Practice needs few directions, and scaling it narrows the model further. |
| Hodge graph | G | The pairwise judge contains cycles no reward can represent. |
| Reward report card | R | Six rewards against eight properties; every failure was visible before training. |
| Price predictions vs outcomes | R-c | A reward's covariance with a feature predicted which way practice moved it. |
| Goodhart curves | R-g | Selecting harder by the reward does not select better lab outcomes than a two-line rule. |

## References

- Amari, *Information Geometry and Its Applications* (2016): dual coordinates, e- and m-geodesics, natural gradient.
- Ren & Sutherland, *Learning Dynamics of LLM Finetuning*, ICLR 2025, arXiv 2407.10490.
- Wu et al., *On the Generalization of SFT: An RL Perspective with Reward Rectification*, ICLR 2026, arXiv 2508.05629.
- Zhu et al., *The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning*, NeurIPS 2025, arXiv 2506.01347.
- *Differential Smoothing Mitigates Sharpening and Improves LLM Reasoning*, arXiv 2511.19942.
- *RL Squeezes, SFT Expands*, arXiv 2509.21128.
- Rath et al., *What pass@k Cannot Measure*, NeurIPS 2026.
- *What Does an LLM Learn from Reinforcement Learning? (Fixed-SAE Track)*, arXiv 2609.15064.
- Venhoff et al., *Base Models Know How to Reason, Thinking Models Learn When*, arXiv 2510.07364.
- *The Path Not Taken: RLVR Provably Learns Off the Principals*, arXiv 2511.08567.
- Malladi et al., *A Kernel-Based View of Language Model Fine-Tuning*, ICML 2023.
- Choe et al., *What is Your Data Worth to GPT? (LoGra)*, 2024.
- Chizat, Peyré, Schmitzer & Vialard, *Unbalanced optimal transport: dynamic and Kantorovich formulations*, 2018; Liero, Mielke & Savaré, *Optimal entropy-transport problems*, 2018.
- Jiang, Lim, Yao & Ye, *Statistical ranking and combinatorial Hodge theory*, Math. Programming 2011.
- Geiger et al., *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*, 2023.
- Fisher, *The Genetical Theory of Natural Selection* (1930); Shahshahani, *A new mathematical framework for the study of linkage and selection* (1979); Price, *Selection and covariance*, Nature 1970.
- Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization*, ICML 2023; Beirami et al., *Theoretical guarantees on the best-of-n alignment policy*, 2024.
- Cristianini, Shawe-Taylor, Elisseeff & Kandola, *On kernel-target alignment*, NeurIPS 2002; Canatar, Bordelon & Pehlevan, *Spectral bias and task-model alignment*, Nat. Commun. 2021.
- Ng, Harada & Russell, *Policy invariance under reward transformations*, ICML 1999 (potential-based shaping: the multi-step analogue of zero curl).
- Sundararajan, Taly & Yan, *Axiomatic Attribution for Deep Networks*, ICML 2017 (the axiomatic style of R-a); Shapley (1953).
- Olah, *Neural Networks, Manifolds, and Topology* (2014) and *Visual Information Theory* (2015): the writing models for the support and KL figures.
