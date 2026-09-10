# MIRA — Complete Project Timeline & Decision Log
*For the paper. Compiled 2026-09-01. Companion to RESEARCH_BIBLIOGRAPHY.md (papers) and the doc family in this folder. Everything below is verified against repo artifacts, logs, and commit hashes. Sections: master timeline → decision tree (forks + why) → bug/fix ledger → experiment/results ledger → current state + run 3 spec → open threads.*

---

## 1. Master timeline

### Phase 0 — System understanding & forensics (Jul 20)
- Full parallel audit of the codebase (training core, data pipeline, eval/validator, tooling + git history). Established what MIRA actually is: Qwen3-8B QLoRA, SFT→GRPO, deterministic physics oracle (`validator.py`) instead of reference-recipe matching; factored 9-check reward; 19,861-shard Materials Project phase-diagram cache.
- **July PD-cache saga reconstructed**: 8.5% of shards corrupt (non-atomic writes) → 1,691 refetched; LiFePO4-family 82% ungradeable → root-caused to pymatgen's inability to parse middle-dot hydrates (`FeC2O4·2H2O`); hydrate fix written but *uncommitted*; MP API schema drift (`entry_id` as dict) broke refetch mid-saga.
- Verified hydrate fix live (`inspect_lfp_chemsys.py`): all LFP records resolve exact chemsys; `_best_entry_for_formula` finds on-hull LiFePO4. **Also found**: corpus typo class (`FeC2O4.2H20` — period + zero-for-O) parses silently as garbage (`H20` = 20 H, `O4.2` fractional); `expand_hydrate_notation` deliberately doesn't touch it (indistinguishable from fractional stoichiometry like `Li0.5CoO2`) → flagged as data-audit issue, not parser issue.
- Ran `post_patch_gradeability_audit.py` (full SFT corpus): **96.9% ΔG-gradeable** (50.6% discrete + 46.3% interpolated); interpolation fallback is *not* a rubber stamp (interpolated scores mean 0.768 vs discrete 0.856, only 56% at 1.0); `target_stability` still 49% `sentinel_no_entry`.

### Phase 1 — Repo hygiene (Jul 20–21)
- Committed + pushed the pending work: `c3d6547` (hydrate fix, `get_target()` tolerant lookup, pymatgen pin), `b193030` (13 July forensic scripts + stratified_difficulty_eval).
- Deleted stale `data/cache/phase_diagram.json` (Jun 1 v1-era artifact, unreferenced).
- Documented recurring stale `.git/index.lock` (editor plugin) and the local-machine rsync procedure (`--checksum` for changed/missing shards).

### Phase 2 — Crash fixes + validator hardening (Jul 21)
- **Critical fix #1**: `load_validator(pd_cache_path=...)` stale kwarg crashed BOTH `experiments/grpo.py` and `evaluate_batched.py` — and even when "working", `pd_cache_path=None` meant **the entire thermodynamic reward had never influenced a single training gradient**. Rewired both to `pd_index_path` + `project_root` (`8398978`).
- **Validator fixes** (all regression-tested, `32c9e45`): atmosphere classification rewritten token-based (was substring: steam→"reducing", CO₂→"oxidizing", "cold"→"reducing", "carbon"→"inert"); `_resolve_pd` made deterministic (smallest covering superset; was dict-order-dependent → same target could score against different PDs across runs); **gas-uptake rule** (pymatgen's null-space balance silently consumed O₂/NH₃ the route never declared — now a consuming balance is only accepted if the route declares a supplying atmosphere); `operation_order` unknown op types → neutral 0.5 (was free 1.0).
- Parser hardening: `//`-comment strip became fallback-only (was mangling `//` inside JSON string values); `reward_fn` catches broad exceptions (a malformed op could kill a GRPO run mid-step); `mixing_media` str-coerced.
- Data pipeline: `chemistry_class` case bug fixed (niobates were labeled "borate"); null-`thinking` crash hardened.
- **`tests/test_validator.py` created: 41 assertions, all green** — the project's first test suite.
- **The truncation discovery** (the single most consequential finding): v3 corpus sequences are p50 ≈ 4,092, p99 ≈ 8,389 tokens, but `max_seq_len=1024` — **every prior v3 SFT run trained on 100%-truncated sequences**; the model never saw a complete JSON route in training. This retroactively explains the "six checkpoints, identical 0.2846, empty-route" eval artifacts.
- Built the real v3 split from the July corpus (`data/sft_v3/`: 1,217/157/157, zero target leakage, fixed chemistry labels). User retrained SFT → `runs/sft-qlora-sft-v3-2nd-rank16` (checkpoint-117 + final).

### Phase 3 — Stratified eval & the memory war (Jul 21–22)
- `stratified_difficulty_eval.py` on the retrained checkpoint: CUDA memory ratcheted 28.4GB and climbing. Diagnosis: KV cache (~13GB at batch 8 × 11k tokens) + allocator fragmentation + never-freed reserved memory. Fix: `del` + `torch.cuda.empty_cache()` in `generate_batch` + `expandable_segments` + batch sizing. (Eval results: pass@0.65 = 1.0 all tiers; `frac_ge_0.9` 0.2 hard → 0.7 ambiguous.)
- Claude's review of that eval: tiers stale (assigned by old validator), sentinel 0.5s paid as reward, tier-4 threshold (50 meV) degenerate (`1_easy` had 1 record, `2_medium` zero), ungradeable tier contaminated with now-gradeable records.

### Phase 4 — Theory consolidation & the sentinel decision (Jul 22–23)
- Read the theory corpus (HANDOFF_2, chat snippets); researched Ringstrom (STOKs — scalar reward as lossy compression of event structure), GDPO (per-reward decoupled normalization), JEPA line incl. Crys-JEPA (confirmed: adjacent task, not a scoop; and "surrogate embedding as RL reward under optimization pressure" identified as the open experiment).
- **Sentinel exclusion implemented** (`923f311`): can't-compute checks (`ungradeable`/`sentinel_no_entry`/`no_balance_found`/`no_thermo_checker`) dropped from `active_weights`, rest renormalized. Sentinel payout was up to ~0.225 free reward in thermo mode and constant advantage-zero mass in GRPO groups. Tests → 43 assertions.
- p̂ formalized for the team: per-target success probability; GRPO signal ∝ informative-group rate (87% at p=0.5, 4% at p=0.99, G=4) → RL datasets should be built from the intermediate band.

### Phase 5 — Re-triage + the p̂ probe (Jul 22–30)
- Full re-triage of the 17,616-record corpus with the fixed validator: **gradeability 82.65% → 85.4%** (discrete 46.9%, interpolated 38.5%, ungradeable 14.6%); literature-route median reward 0.866 → 0.893; fractional targets = 35% of corpus, 0% discrete (interpolation is load-bearing for a third of all data).
- `probe_effective_support.py` built (stratified sampling, full-thermo grading, incremental/resumable). **Pilot: 125 targets × 24 samples @ T=1.0** → at bar 0.65 saturated (p̂ 0.83–0.98); at bar 0.9 mean 0.46–0.67 with ~half the targets mid-band; strata discriminate (frac hardest). Verdict: **SFT solved "adequate," not "excellent" — RL has a real job, and the buckets define the RL dataset.**
- Infra pattern established after lost background tasks: tmux + auto-restart runners (`run_probe.sh`) + tee logs + ntfy phone pings.
- `KIMI_SESSION_WRITEUP.md` produced for Claude. Claude's response (the four pre-launch requirements): None-not-0.0 for can't-compute checks under GDPO; sweep remaining constant-0.5 branches; drop the `max(r−0.30,0)` clip under GDPO; G=4→8 minimum (per-check z-std at G=4 has ~40% relative error); dump every generation. Plus the apparatus: 4-run grid, frozen axes, pre-registered metric, validator version stamps, generation archive for RS-SFT/DPO/JEPA-monitor reuse.
- Verified in installed TRL 1.8.0 source: GDPO path (`normalize_then_sum`) is fully NaN-aware; None from a reward func is the official no-signal channel.

### Phase 6 — RL dataset + GDPO run 1 (Jul 30–Aug 1)
- **Decision**: skip the 2,000-target scaled probe (~28 GPU-days for full-universe p̂) — build `data/rl/` from the non-SFT universe instead (8,099 unique triage targets − 1,531 SFT = 6,568 → 6,368 train + 200 val; prompts regenerated with the fixed validator; manifest with composition + caveats). Rationale: continuous rewards soften p=1; first run doubles as on-policy probe via the generation dump.
- **Per-check reward bank** (`core/reward.make_check_reward_fns`): one cached `validate()` per completion; 9 check funcs returning None→NaN for sentinels; `format_ok` (parse discipline without poisoning chemistry channels); every generation archived to `runs/<run>/generations.jsonl` (exactly-once — a dump-duplication bug was caught and fixed pre-launch).
- `experiments/gdpo.py` registered (`normalize_then_sum`); `run_gdpo.sh` auto-restart runner with latest-checkpoint resume.
- **OOM #1**: `convert_to_fp32` tried 13.91 GiB (fp32 logits for batch 4 × ~8k × 152k vocab) → `batch_size 4→2` (`673d71d`).
- **OOM #2**: 3.48 GiB inside `loss.backward()` at batch 2 (fp32 logits gradient chain ~20GB) → `batch_size 2→1, accum 8→16` (`e7af963`). Effective batch 16 held constant throughout — memory knobs only, never statistical ones.
- **Speed**: 889 s/step → 66-day ETA → `use_transformers_continuous_batching` + `max_completion_len 6144→8192` + scope 1 epoch × 2,000 prompts (`3041bf9`). Outcome: step_time rose to ~1,400 s (longer generations + CB memory cap) but `clipped_ratio` 0.20 → 0.0175 — slower steps, ~10× less wasted compute per step.
- **Run 1 (300 steps, user-stopped)**: mean reward 0.96–1.0 flat-ish, entropy 0.59→0.63 (no collapse), `frac_reward_zero_std=0` throughout, KL ~0.001 (policy barely moved), clipping fixed, parse-fail 4.7%.

### Phase 7 — On-policy diagnostics (Aug 5–10)
- Dump analysis (5,040 completions, 622 targets): strata coverage representative (shuffled sampler OK); on-policy score by stratum matches the difficulty map (int×discrete 0.932 > … > frac×ungradeable 0.852); **on-policy p̂: 53.5% mid-band, 19.1% p̂=0, 27.3% p̂≥0.95** → ~46% of run-1 compute went to low/no-signal targets; learning curve flat (KL tiny).
- **β-ablation probe (run 2)** from ckpt-300 with β 0.04→0.001, lr 5e-6→1e-5, 150 steps: KL rose 3.7× with growing deltas, entropy rose faster than run 1, `clip_ratio` identically 0, no reward collapse → **run 1 was over-constrained, not at ceiling. β=0.001 and lr=1e-5 confirmed.**
- Claude's `reward_geometry.py` diagnostics (the D1–D5 pack): **reward capacity = 1.43/10 = 14%** — only `amount_accuracy` (z-var 0.60) and `thermodynamic_favorable` (0.36) alive; two channels prompt-determined (`target_stability`, `target_match`), two saturated (`precursors_exist`, `temperature_plausible`), four diversity-limited (mean **2.0 distinct precursor sets per group**; 45% of groups share one set); within-target trends *inestimable* (0/612 targets repeated across steps → per-check curves are target-sampling noise); effective rank 4.36 explicitly flagged as misread-bait.
- **§4a resolved as false alarm**: identical NaN rates across channels were an artifact of `reward_geometry.py` reading raw dump breakdowns without masking `*_gradeability` tags; the training-side None-propagation was verified clean (unit test + TRL source).

### Phase 8 — Closed-book verdict + the analysis chain (Aug 13–16)
- **Closed-book probe** (125×24 @ T=1.0, no PD context in prompts): strip verified four ways (MODE log; side-by-side render 826 vs 1,383 tokens; code path; completions reason from first principles, never cite prompt-provided decomposition/hull values). Result: **Δp̂ ≤ 0.04 vs SE ~0.08 on every stratum — closed-book ≡ open-book. It is a cost lever (~2× throughput), not a difficulty lever.** Consequence: the model never used the PD context; thermodynamics lives entirely in the reward.
- **Gradeability flap measured**: up to 60% of frac-stratum targets get graded by different channel sets across samples → run-3 dataset must filter to stable-gradeability targets; tag vectors logged for variance decomposition.
- **Temperature sweep** (20 targets × T{1.0,1.2,1.5} × G=8): distinct precursor sets/group 1.90 / 2.30 / 2.05 → **policy structurally peaked; temperature is not a diversity lever.** Diversity intervention must be clip-higher / entropy bonus / seeded support.
- **PD attention check** (eager loader, 5 open-book prompts): PD block = 43% of tokens; attention mass 1.07× proportional all-layers but **0.69× in late layers — "read early, discounted late."** Mechanistic confirmation (second instrument) that prompt thermodynamics isn't driving decisions.
- **pass@k baseline** (15 held-out targets × 48 samples, bar 0.9, closed-book, full-thermo grading): base 0.64/0.72/0.73 → SFT 0.76/0.87/0.87 → GDPO-300 0.78/0.88/**0.93** (pass@1/8/48). SFT expanded the boundary (distillation installs support); **GDPO expanded it further (+6.6 pts pass@48) with modest pass@1 gain — first positive boundary-expansion signal in the project** (n=15, directional; needs more targets + equal-compute SFT control).

### Phase 9 — Documentation (Aug 17 → Sep 1)
- `KIMI_SESSION_WRITEUP.md`, `GDPO_RUN1_LEARNINGS.md`, `ANALYSIS_CHAIN_RESULTS.md`, `KIMI_NEW_CHAT_PRIMER.md`, `RESEARCH_BIBLIOGRAPHY.md`, this timeline. Docs in this folder are the canonical handoff chain.

---

## 2. Decision tree (the forks, with why)

1. **Fix the validator before more training** (over "train more on a broken instrument"). Every downstream number depended on it; the month of bugs was three systems in one trench coat (parser / cache / reward function) plus a self-reference loop (validator filters data that trains the model it evaluates).
2. **Thermo reward must be wired or the PD cache is dead weight** → the `pd_index_path` rewiring. Discovered the whole thermo infrastructure had never touched a gradient.
3. **Sentinel exclusion (None-propagation)** over sentinel 0.5 payout — Route C from HANDOFF_2 §2b. Chosen over partial-reward substitution: honest for eval, and TRL's NaN channel makes it exact for GDPO.
4. **Bump `max_seq_len` instead of prompt surgery** for the retrain (option 1 over option 2) — pragmatic; prompt-bloat fix deferred to a corpus iteration.
5. **p̂ probe before RL spend** (measure where gradient exists) — the pilot (125×24) instead of the full corpus (28 GPU-days).
6. **Skip the scaled probe; build RL set from the non-SFT universe** (continuous rewards soften p≈1; the run itself becomes the probe via the dump). Cost: ~46% compute to dead bands (measured after the fact); benefit: ~3 days saved + on-policy p̂ for free.
7. **GDPO over GRPO** for the factored reward (Ringstrom's diagnosis; GRPO's sum-then-normalize collapses distinct check combinations into identical advantages). TRL-native implementation chosen over NVIDIA's fork.
8. **Per-check funcs return None, not 0.0** (Claude's #1) — 0.0 reinstalls the sentinel bug one level up and injects spurious z-scores.
9. **batch_size 1 + accum 16** — memory geometry: batch is the only free dimension in batch × seq × 152k × dtype; effective batch (16) and G (8) are statistical knobs and were never touched.
10. **Continuous batching + cap 8192 + scope 1000 steps** — lockstep was letting clipped ramblers gate every batch; 6144 clipped 20% (→ parse-fail-only signal); 1000 steps is the honest single-GPU RLVR budget with checkpoints every 100.
11. **β=0.001, lr=1e-5** (run-2 ablation) — run 1 was over-constrained; KL needs to grow for the policy to move.
12. **Closed-book as default** — proven equivalent (Δ≤0.04) at ~2× cheaper. The "closed-book creates headroom" hypothesis died; it became a cost optimization.
13. **Diversity via clip-higher, not temperature** — sweep proved the policy is structurally peaked (1.9–2.3 distinct sets regardless of T).
14. **Run-3 reward vector = 5 live channels, uniform weights** — capacity 14% means the 10-channel vector was mostly noise-free-but-useless slots; validator's scalar weights were tuned for [0,1] sums and mean something different after z-scaling.
15. **Fixed ~30-target repeated probe set** — the only way to make trends identifiable (612/612 targets appeared exactly once in runs 1–2).

## 3. Bug/fix ledger (paper-relevant subset)

| # | Bug | Root cause | Fix | Commit |
|---|---|---|---|---|
| 1 | GRPO + eval crashed / light-only thermo | stale `pd_cache_path` kwarg after cache sharding | `pd_index_path` + `project_root` | `8398978` |
| 2 | **Thermo reward never trained anything** | `pd_cache_path=None` → light mode silently | same as #1 | `8398978` |
| 3 | **100% of v3 training sequences truncated** | `max_seq_len=1024` vs p50 4,092 tokens | 9,216 for SFT retrain; 8,192 for RL | manual / `3041bf9` |
| 4 | Middle-dot hydrates unparseable (LFP 82% ungradeable) | pymatgen can't parse `·` | `expand_hydrate_notation` `A·nB→A(B)n` | `c3d6547` |
| 5 | Sentinel 0.5 paid as reward (≤0.225 free) | can't-compute returned 0.5 at full weight | None-propagation in `validate()` | `923f311` |
| 6 | Gas-uptake hole (phantom O₂ consumption) | null-space balance puts volatiles on either side | supplier-atmosphere required for consuming balances | `32c9e45` |
| 7 | Atmosphere misclassification (steam→reducing, CO₂→oxidizing) | substring matching with 2-letter tokens | token-based matching | `32c9e45` |
| 8 | Non-deterministic scoring across runs | `_resolve_pd` dict-order superset choice | smallest covering superset + tiebreak | `32c9e45` |
| 9 | Unknown op types scored 1.0 | rank-99 never drops | skip pairs; neutral 0.5 if unassessable | `32c9e45` |
| 10 | `//` inside JSON strings mangled | eager comment-strip regex | fallback-only stripping | `32c9e45` |
| 11 | Niobates labeled "borate" (stratification corrupted) | uppercasing formula before element regex | proper-case extraction | `8398978` |
| 12 | Eval `aggregate()` crash | `statistics.mean` over string gradeability tags | numeric-only filter | `8398978` |
| 13 | Six checkpoints scored identically (0.2846) | parse failures silently fell back to empty route | ParseFailure visibility + regrade script | (prior) + eval wiring |
| 14 | 8.5% PD shards corrupt | non-atomic shard writes | refetch all 1,691 (+ atomic index) | (Jul) |
| 15 | MP refetch crashed | `entry_id` returned as dict (API schema drift) | defensive handling | (Jul) |
| 16 | CUDA memory ratchet on eval | KV cache + fragmentation + no `empty_cache` | `del` + `empty_cache` + expandable segments | `8398978` |
| 17 | OOM in logps forward (13.91 GiB) | fp32 copy of full-vocab logits, batch 4 | batch_size 4→2 | `673d71d` |
| 18 | OOM in backward (3.48 GiB) | fp32 logits gradient chain ~20GB | batch_size 2→1, accum 16 | `e7af963` |
| 19 | 66-day ETA | lockstep generation + 20% clipping + 6,368×2 scope | continuous batching + cap 8192 + limit 2000 | `3041bf9` |
| 20 | Generation dump wrote 7× per completion | dump inside per-func `bank()` calls | dump only on cache miss (exactly-once) | `07e6758` |
| 21 | Corpus typo formulas (`FeC2O4.2H20`) parse as garbage | OCR-style `0`→`O` typos | flagged: data audit, not parser (fractional-stoich collision) | open |
| 22 | `data_pull_3` polymorph dedup keeps wrong polymorph | `0.0 or 1e9` → 1e9 | latent (cached from older correct pull) | open |
| 23 | Trace generation skips 89% of corpus | "completed" keyed on target_formula | credit-capped; future regen decision | open |

## 4. Experiment & results ledger

| Experiment | Config | Headline result |
|---|---|---|
| Gradeability audit (Jul 21) | full SFT corpus, patched validator | 96.9% ΔG-gradeable; interpolation discriminates (not rubber stamp); stability 49% sentinel |
| Re-triage (Jul 30) | 17,616 records, fixed validator | gradeable 82.65%→85.4%; median reward 0.866→0.893 |
| SFT retrain (Jul 21) | QLoRA r16, data/sft_v3, seq 9,216 | `runs/sft-qlora-sft-v3-2nd-rank16` |
| Stratified eval (Jul 22) | retrained SFT, full PD context | pass@0.65 = 1.0 all tiers; ge_0.9: 0.2–0.7 by tier |
| p̂ probe pilot (Jul 30) | 125×24 @ T=1.0 | saturated at bar 0.65; informative band at 0.9 (0.46–0.67); strata discriminate |
| GDPO run 1 (Aug 1–5) | 10-ch reward, β=0.04, lr 5e-6, G=8, 300 steps | machinery validated (0 dead groups, entropy stable, clipping fixed) but KL ~0.001, curve flat → over-constrained |
| β-ablation probe (Aug 8–10) | β=0.001, lr 1e-5, 150 steps | KL ×3.7 rising, no collapse → keep β=0.001/lr=1e-5 |
| Reward geometry (Aug 10) | run-2 dump | capacity 14%; 2/10 channels live; diversity ~2 sets/group; trends inestimable |
| Closed-book probe (Aug 14) | 125×24, no PD context | ≡ open-book within noise → cost lever, not difficulty lever |
| Temperature sweep (Aug 15) | 20 targets × 3 T × G=8 | 1.90/2.30/2.05 sets → structurally peaked; temperature dead |
| PD attention (Aug 15) | 5 open-book prompts | 1.07× all-layers, 0.69× late — "read early, discounted late" |
| pass@k (Aug 15–16) | 15 targets × 48, bar 0.9 | base 0.64 → SFT 0.76 → GDPO-300 0.78 (pass@1); pass@48: 0.73 → 0.87 → **0.93** |

## 5. Current state + run 3 spec (as of 2026-09-01)

**State**: SFT-v3-2nd is the production checkpoint. GDPO run 1 stopped at 300 (checkpoints 100/200/300). All validator fixes live + tested (43 assertions). `data/rl/` (6,368+200) exists. Run 2 confirmed β=0.001/lr=1e-5. Closed-book is the default prompt mode. All diagnostics (capacity, diversity, flap, pass@k) complete.

**Run 3 spec** (ready to implement, not yet launched):
- Reward: 5 channels (`amount_accuracy`, `thermodynamic_favorable`, `stoichiometry`, `chempot_atmosphere`, `operation_order`), uniform `reward_weights`; drop the two prompt-determined, two saturated, and `charge_neutrality` (insensitive to the policy's actual variation).
- β=0.001, lr=1e-5; closed-book prompts; G=8; batch 1 × accum 16; `epsilon_high` (clip-higher) for diversity.
- Dataset: mid-band p̂ + gradeability-stable targets only; p̂ re-estimated from run-2 generations.
- Fixed ~30-target held-out probe evaluated every N steps; per-check within-group std logged individually.
- Watch metric: reward capacity (z-variance sum) — must clear ~50% before the GRPO-vs-GDPO grid is a meaningful contrast.

## 6. Open threads

1. **Run 3 launch** (prep edits then tmux launch; runner pattern established).
2. **pass@k sharpening**: more targets + GDPO-300 vs equal-compute SFT continuation — the boundary-expansion claim's control.
3. **Interpretability programme**: do live channels (`amount_accuracy` vs `thermodynamic_favorable`) move different representational subspaces? (activations over fixed probe set; weight-space Euclidean measures deprecated as non-Fisher-respecting per Path Not Taken.)
4. **Curriculum**: filter/weight by p̂(1−p̂), re-estimate on-policy (VADE-style Beta posterior); no easy→hard ordering (literature: filtering > ordering).
5. **Crys-JEPA comparison** (experiment two, downstream): exact-reward GDPO vs surrogate-embedding-reward GDPO; MACE embeddings + Qdrant already on disk.
6. **Adapter arms**: LoRA-no-quant (cleaner geometry) as cheap third arm; full-FT off this card.
7. **Corpus hygiene backlog**: typo-formula audit (`H20` class), re-grade corpus with final validator, prompt-bloat fix at the generator, `data_pull_3` polymorph bug, 89% unattempted trace coverage.
