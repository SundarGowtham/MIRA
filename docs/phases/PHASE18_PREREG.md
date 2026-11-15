## Spec: Phase 18 pre-registration

### Why this phase exists

Phase 17 left three gaps that papers from Anima Anandkumar's group speak to directly:

- **Block C (unbalanced transport) converged on only 6 of 385 target-steps.** It also read teacher-forced log-probabilities, which Phase 17 found miscalibrated in the tail.
- **H1 (weight geometry, *The Path Not Taken*) was never run.**
- **The reward audit tested symmetry (R-a) but not representation.** It never asked whether a judge scores the physical recipe or the way it is written down.

Five further blocks reuse existing data to put the reward audit into the language of that group:
- whether judge inconsistency (curl) locates judge error;
- distributional evaluation (posterior matching);
- channel-level validity;
- tensor latent-variable structure;
- conformal calibration.

### Rules (frozen)

- **No training and no new sampling.** Use existing raw completions, adapters, Phase 17 outputs, ASTRAL and ARROWS³ data.
- **Recipe** means the canonical precursor set from `precset_of` in `research/support_growth_by_checkpoint.py`. Every block uses this canonicaliser.
- **Step table and target sets** are as in `docs/phases/PHASE17_PREREG.md` (S1–S7; ASTRAL clean 22 primary, full 35 secondary; held-out 40).
- **Bootstraps** cluster by target unless a block says otherwise: 10,000 resamples, seed 20261115.
- **Code** goes in `research/phase18_*.py`. Do not edit `validator.py`, `core/comparator.py` or ranker code; wrap them.
- **New files only,** under `results/phase18/`. Never overwrite a tracked results file.
- **Jobs** are long-running only in tmux with `tee` and ntfy, in user-space environments. No `sudo`, no `apt`.
- **Reporting.** A failed prediction is reported as failed. Anything decided after the commit is labelled **[E]**.
- **Missing inputs.** If an input a block needs is missing, skip that block and say so. Do not improvise a substitute.

---

### Block 0 — inventory (CPU)

Record, for every model in S1–S7 plus Base:
- the raw-completion files per target set, with the number of samples per target;
- the adapter paths;
- intermediate checkpoints;
- the LoRA rank and alpha.

Also record:
- **Phase 17 outputs.** Which `results/phase17/*` files exist, and which blocks ran.
- **Base weights.** Whether bf16 base weights for Qwen3-8B are in the local HF cache. If they are not, ask before downloading.
- **ASTRAL lab data.** The per-target lab outcomes, using the success definition Phase 16 used for hits.
- **ARROWS³ YBCO data.** The recipes, temperatures and yields, loaded with `research/distributional/arrows_gate_scoring.py`.
- **Judge versions.** The current `VALIDATOR_VERSION`, `COMPARATOR_VERSION` and ranker v2 path.

**Output:** `results/phase18/inventory.json`.

---

### 18A — transport or reweighting, solved exactly (CPU)

**Link.** This is the Wasserstein–Fisher–Rao (unbalanced optimal transport) split of Phase 17 Block C, which failed to converge. It is the area of Chuwei Wang (optimal transport and sampling, Anandkumar and Stuart groups).

**What changes from Phase 17:**
- **Measures.** The measures are empirical recipe frequencies from existing samples, not teacher-forced probabilities. This avoids the tail-calibration problem.
- **Equal sample sizes.** Where the two models of a step have different sample counts, subsample both to the smaller n. Repeat 200 times and average.
- **Solver.** The problem is solved exactly as a linear program (HiGHS via `scipy.optimize.linprog`), with a total-variation penalty on created or destroyed mass:
  $$\min_{P\ge0}\ \langle C,P\rangle+\lambda\big(\|\mu-P\mathbf 1\|_1+\|\nu-P^\top\mathbf 1\|_1\big)\quad\text{s.t. } P\mathbf 1\le\mu,\ P^\top\mathbf 1\le\nu .$$
- **Cost.** $C$ is the Jaccard distance between precursor sets, with $\lambda\in\{0.35,0.5\}$. Moving mass is only cheaper than creating it when $C<2\lambda$. A one-precursor swap in a three-precursor set costs 0.5.
- **Secondary solver.** Log-domain stabilised entropic unbalanced OT (POT `sinkhorn_unbalanced`, `method="sinkhorn_stabilized"`), for comparison only.
- **Convergence.** Report solver status for every target-step. The LP must solve 100% of them.

**Quantities, per target and step:**
- **Mass that stays:** $m_{\text{stay}}=\sum_i P_{ii}$.
- **Mass that moves:** $m_{\text{move}}=\sum_{i\ne j}P_{ij}$.
- **Created mass** $m_{\text{new}}$ and **destroyed mass** $m_{\text{gone}}$.
- **Transport share:**
  $$T=\frac{\langle C,P\rangle}{\langle C,P\rangle+\lambda(m_{\text{new}}+m_{\text{gone}})} .$$
- **New-support mass:** the after-model's mass on recipes absent from the before-model's largest available reference.
- **Noise floor:** each quantity computed between two independent same-size subsamples of the before-model.

**Predictions [C]:**
- **P-18A-1.** At both values of $\lambda$, the median $T$ is higher for the own-sample fine-tuning steps (S2–S4) than for the practice steps (S5–S7). The bootstrap CI on the difference of medians excludes 0.
- **P-18A-2.** New-support mass minus its noise floor has a CI that includes 0 for each of S5–S7, and excludes 0 for S3 and S4.

**Does not support the thesis:** P-18A-1 reversed, or any practice step's new-support mass CI lying above its noise floor.

**[E] analyses:**
- the distance histogram of moved mass (one-precursor swaps versus unrelated sets);
- six example transport plans with a 2-D MDS layout of each target's recipes.

**Outputs:** `results/phase18/transport_exact.json`, `results/phase18/figdata_transport_exact.json`.

---

### 18B — does judge inconsistency sit where the judge is wrong? (CPU)

**Link.** Anandkumar's thesis is that physical experiments are the bottleneck and verification should guide where they are spent. If the comparator's cycles mark where it disagrees with the lab, curl becomes a free guide for where to run experiments.

**Data:**
- **Primary:** ARROWS³ YBCO.
  - Nodes are recipes.
  - The lab flow on edge $(a,b)$ is the mean, over shared temperatures, of the sign of the yield difference (as in Phase 17 G2).
  - The comparator flow $Y_{ab}$ is the margin under the current `COMPARATOR_VERSION`, on every pair where both are defined.
- **[E]:** ASTRAL, pooled over targets with lab outcomes. Edges join a lab-successful and a lab-failed set.

**Method:**
1. Hodge-decompose the comparator flow (weighted least squares for the potential $s$).
2. Per edge, compute the non-gradient load $\rho_e=|Y_e-(s_a-s_b)|$.
3. Compute disagreement $d_e=\mathbf 1[\operatorname{sign}Y_e\ne\operatorname{sign}Y^{\text{lab}}_e]$, on edges with a nonzero lab sign.
4. Report the AUC of $\rho_e$ for predicting $d_e$.
5. Bootstrap by resampling recipes (nodes). Use a permutation null of 1,000 with $\rho$ shuffled across edges.

**Prediction [C]:**
- **P-18B-1.** The AUC is above 0.5, with the CI excluding 0.5, and it beats the permutation null at p < 0.05.

**Does not support** curl as a locator of error: the CI includes 0.5. That is still reported as a finding, because curl is a structural defect whether or not it locates errors.

**[E]:** the same AUC using a channel-conflict score for the scalar rewards R2 and R4. The score is the share of channels gradeable on both recipes whose pairwise sign opposes the aggregate's.

**Output:** `results/phase18/curl_vs_lab.json`.

---

### 18C — low-rank structure of the updates, and the *Path Not Taken* check (CPU; GPU optional for SVD)

**Links:**
- GaLore and InRank (Jiawei Zhao, Anandkumar): gradients and cumulative updates are low-rank, and rank grows incrementally during training.
- *The Path Not Taken* (arXiv 2511.08567): RLVR updates avoid the base weights' principal directions. This was Phase 17 H1, never run.

**Caveats in the write-up.** Weight-space Euclidean angles do not respect the Fisher metric, which is why the project deprecated this diagnostic in September. They are reported only for comparability with *The Path Not Taken*. That paper used full fine-tuning; MIRA uses QLoRA, so every rank is capped at $2r$.

**Method.** Per targeted module (all layers):
- **Update:** $\Delta W=\tfrac{\alpha}{r}BA$ for fine-tuning steps, and $\tfrac{\alpha}{r}(B_1A_1-B_0A_0)$ for practice steps, which continue their init adapter.
- **Base weights:** $W_{\text{base}}$ from bf16 weights, loaded per layer. Never use the 4-bit weights.
- **Metrics:**
  - effective rank $\exp H(\sigma^2/\textstyle\sum\sigma^2)$;
  - stable rank $\|\Delta W\|_F^2/\sigma_1^2$;
  - principal-subspace energy $E_k=\|U_k^\top\Delta W\|_F^2/\|\Delta W\|_F^2$ for $W_{\text{base}}$'s top-$k$ left singular vectors (and the right-side analogue), with $k\in\{16,64,256\}$, against the random-subspace expectation $k/d_{\text{out}}$.
- **Trajectory:** the same metrics on the cumulative $\Delta W_t$ at checkpoints 100, 200 and 300 for S6 and S7, and for any intermediate fine-tuning checkpoints that exist.

**Predictions [C]:**
- **P-18C-1 (*The Path Not Taken*).** The pooled median of $E_{64}$ across modules is lower for practice steps (S5–S7) than for own-sample fine-tuning steps (S2–S4). The module-bootstrap CI on the difference excludes 0.
- **P-18C-2 (InRank).** For S6 and S7, the effective rank of the cumulative $\Delta W_t$ is non-decreasing over 100 → 200 → 300 in at least 75% of modules.

**[E]:**
- the per-layer profile of $E_{64}$ and effective rank;
- whether practice updates are lower-rank than fine-tuning updates.

**Output:** `results/phase18/weight_geometry.json`.

---

### 18D — does a judge score the physics or the text? (CPU)

**Link.** A neural operator gives the same answer at any discretisation of its input function. A heating schedule is a function, so a reward should not change when the same schedule is written differently. This extends Phase 17 R-a.

**Suite.** Start from the 200-route R-a suite. Apply each transformation that leaves the physical recipe unchanged:

| Id | Transformation |
|---|---|
| T1 | °C ↔ K in every temperature |
| T2 | hours ↔ minutes in every duration |
| T3 | split one isothermal hold of duration $t$ into two consecutive identical holds of $t/2$ |
| T4 | insert a zero-duration hold at the current temperature |
| T5 | state a ramp's rate explicitly where it was implied, only if the schema allows |
| T6 | permute precursors (control, already in R-a) |

**Judges and what counts as a violation:**
- **Validator:** the 5-channel aggregate and each channel. A violation is $|\Delta\text{score}|>10^{-3}$.
- **Ranker v2:** same threshold as the validator.
- **Comparator:** each route is compared against its own transform. A violation is a margin $|m|>10^{-3}$.
- **Parse failures** are reported as their own category.

**[C-measure], no direction.** Report violation rates per judge and transformation. This adds a report-card column, **R-a′ representation**, which passes only with 0 violations.

**Output:** `results/phase18/audit_representation.json`.

---

### 18E — posterior matching instead of best sample (CPU)

**Link.** FunDPS and PosteriorBench (Jiachen Yao) evaluate whether a generator recovers the whole set of valid answers, not one good answer.

**Setup:**
- **Models:** Base, Corpus SFT, RS-SFT, E3, E3c, GDPO-300, Phase 12 @300, E1 @300.
- **Distributions:** each model's empirical recipe distribution $p_m$ from existing ASTRAL samples.
- **Lab-valid set:** $V(x)$, the experimentally successful sets under Phase 16's definition.
- **Targets:** clean 22 primary, full 35 secondary.

**Metrics:**
- **Lab-valid mass:** $p_m(V)$.
- **Coverage:** $|V\cap\operatorname{supp}p_m|/|V|$.
- **Distance to the valid set:** MMD between $p_m$ and the uniform distribution on $V$, with kernel $k(y,y')=1-J(y,y')$.
- **Near-miss distance:** $\mathbb E_{y\sim p_m}\min_{v\in V}J(y,v)$.

**Prediction [C]:**
- **P-18E-1.** E3's MMD is lower than Phase 12 @300's, with the cluster-bootstrap CI on the difference excluding 0.

**[E]:**
- the rank agreement between the MMD ordering and Phase 16's hit ordering;
- the share of practice samples within one precursor swap of a valid set ("the needle is nearby, but a tilt cannot move there").

**Output:** `results/phase18/posterior_matching.json`.

---

### 18F — do specific channels know the lab better than the aggregate? (CPU)

**Link.** *The Personality Illusion* (ICML 2026) and its June 2026 follow-up (Kocielnik, Han et al., with Anandkumar):
- LLM self-reports on broad traits do not predict behaviour;
- measures specific to a behaviour do better.

The analogue here: does a specific validator channel carry lab information that the aggregate washes out?

**Data.** Use the Phase 17 R-f setup:
- ARROWS³ pairs are primary;
- ASTRAL pairs are flagged small.

**Method:**
- Per channel $c$ (on pairs where it is gradeable for both recipes), compute $I(R_c;Q\mid\text{carbonate-free})$ in bits, with a bootstrap CI.
- Put it next to the aggregate R2, R4 and R5 values from R-f.
- Apply a Holm correction across channels.

**[C-measure], no direction:**
- the number of channels whose corrected CI excludes 0;
- the difference between the best channel and the aggregate, with its CI.

**Interpretation, fixed now:**
- If some channel beats the aggregate, aggregation destroys validity.
- If none does, no channel knows the lab.

**Output:** `results/phase18/channel_validity.json`.

---

### 18G — what hidden variable do the channels measure? (CPU)

**Link.** Anandkumar, Ge, Hsu, Kakade & Telgarsky, *Tensor decompositions for learning latent variable models*, JMLR 2014. The model treats each channel as a view of a hidden class $h$, with the channels conditionally independent given $h$.

**Method:**
1. **Data.** Use all existing scored samples, pooled, with each checker channel binarised at its pass threshold.
2. **Method of moments.** Form the whitened third-order moment and run the robust tensor power method for $k\in\{2,3,4\}$.
3. **Check.** Fit EM latent-class analysis with the same $k$.
4. **Report** the component weights and the per-channel conditional pass rates.
5. **Assignment.** Assign each sample to its posterior component.
6. **Association.** On samples whose recipe has an ASTRAL lab outcome, compute the odds ratio between the "all channels pass" component and lab success.

**Prediction [C]:**
- **P-18G-1.** That odds ratio's CI includes 1. This is consistent with R-f: the channels' shared latent is validity, not lab quality.

**[E]:** residual within-component pairwise correlations, as a diagnostic of the conditional-independence assumption.

**Output:** `results/phase18/tensor_latent.json`.

---

### 18H — a calibrated judge that abstains (CPU)

**Link.** The UQNO paper (Ma, Pitt, Azizzadenesheli & Anandkumar, TMLR 2024) gives conformal, distribution-free calibration with a finite-sample guarantee.

**Method:**
- **Data:** ARROWS³ pairs labelled by lab sign.
- **Judges:** R2, R4, R5 and the carbonate-free rule.
- **Calibration:** split conformal on $\hat p(a\succ b)=\operatorname{sigmoid}(\text{margin}/\text{scale})$, with the scale fit on the calibration half.
- **Splits:** grouped by recipe so no recipe appears on both sides; 100 random splits.
- **Report at target coverage 80% and 90%:**
  - achieved coverage;
  - the decisive share (prediction set of size 1);
  - the abstention share (size 2).

**Prediction [C]:**
- **P-18H-1.** For R2, R4 and R5, the decisive share at 90% coverage is ≤ 20%. A calibrated version of each judge would abstain on at least 80% of lab decisions, consistent with chance-level validity.

**[E]:**
- the carbonate-free rule;
- ASTRAL.

**Output:** `results/phase18/conformal.json`.

---

## What each outcome means for the paper

- **18A holds.** Block C's claim stands, from data that avoids the miscalibrated instrument. This is the transport figure.
- **18B holds.** Curl is not only a defect; it is a free guide for where to spend lab experiments. This is the strongest new sentence for the reward-design section.
- **18C holds.** The *Path Not Taken* and InRank findings carry over to QLoRA in this domain. If 18C fails, the paper reports that the published weight-space signature did not appear under LoRA.
- **18D, 18F, 18G and 18H.** These become columns and rows of the report card: representation invariance, channel-level validity, latent structure, and calibrated abstention.
- **18E.** This gives the paper a distributional evaluation in the language FunDPS and PosteriorBench use.