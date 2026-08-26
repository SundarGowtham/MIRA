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

## Repo map

- `validator.py` — the oracle. `VALIDATOR_VERSION` bumps on scoring changes. **The user owns this file; check before touching it.**
- `core/` — data/model/reward/observability. `core/reward.py::make_check_reward_fns` builds the RL reward vector (currently the 5-channel `RUN3_CHECKS`, uniform weights).
- `experiments/` — sft.py, grpo.py (base), gdpo.py (thin override), sft_grpo.py. `train.py` is the CLI dispatcher.
- `data/` — `sft_v3/` (1,217/157/157), `rl/` (6,568 prompts, SFT-disjoint), `rl_run3/` (run-3's filtered set), `cache/` (pd shards + formula set — **never delete anything here**), `raw/synthesis_clean.json` (17.6k Kononova literature routes).
- `runs/` — checkpoints. `sft-qlora-sft-v3-2nd-rank16/` (SFT, the RL-init), `gdpo-qlora-gdpo-v3/checkpoint-{100,200,300}` (run 1), `gdpo-qlora-beta-ablation-probe/` (run 2). `gdpo-qlora-gdpo-v4/` = killed run 3, dead artifact.
- `misc/` — **gitignored scratch**: probes, results, the doc family.
- `run_logs/`, `run_*.sh` — tmux runners with ntfy pings.
- `tests/test_validator.py` — 43 assertions; run after ANY validator change: `uv run python tests/test_validator.py`.

## Verified findings (the load-bearing numbers)

1. **Reward capacity = 14%** (z-variance 1.43/10): only `amount_accuracy` (0.60) and `thermodynamic_favorable` (0.36) are alive within-group. 8/10 channels contribute no gradient (saturated, prompt-determined, or diversity-limited).
2. **Diversity collapse is structural**: ~2.0 distinct precursor sets/group of 8, flat across T=1.0–1.5 and across runs. Temperature is not a lever; it was an SFT property, not RL-caused.
3. **Closed-book ≡ open-book** (|Δp̂| ≤ 0.04): the model never used the PD context (attention "read early, discounted late"). Closed-book is default: 2× cheaper.
4. **pass@k n=200, k=16 (2026-08-23, THE decision result)**: SFT 0.870 vs GDPO-300 0.880 → paired diff **+1.0 pt** (median 0; 188/200 ties, 7 vs 5 decisive). The pre-set bar was ≥5 pts. **300 steps of GDPO did not expand the boundary.** The n=15 +6.6pt signal was small-n noise. RL-premise verdict: negative for runs 1–2; the difficulty must be manufactured (see below) before RL can show anything.
5. **Activation geometry (2026-08-21)**: channel-aligned activation directions (z-weighted) do NOT separate from a within-group permutation null at any layer in base/SFT/GDPO-300 (median p 0.2–0.4, 0/37 layers p<0.05). CKA base→SFT 0.92, SFT→GDPO 0.96. No linear channel-subspace signal at pooled-completion resolution.
6. **Validity read (2026-08-18)**: 20 validator-≥0.9 completions vs literature (`misc/validity_read_20.md`) — mostly plausible route families, BUT confirmed the temperature hack: AgCa₂Mg₂V₃O₁₂ scored 1.000 with a 1300 °C sinter (Ag volatilizes); `temperature_plausible` only checks a 100–2000 °C window. Two independent reads converged on this. Fix is queued as run-4 prep (do NOT fix before the hardening probe — comparability).
7. β=0.001, lr=1e-5 confirmed (run-2 ablation: KL rose 3.7×, no collapse, clip_ratio=0).
8. Gradeability flap: up to 60% of fractional targets get graded by different channel sets across samples — run-3 dataset filters to stable-tag targets (strict = constant tag vector).

## Journey (why things are the way they are)

1. Six checkpoints once "scored identically" — root cause: silent parse-failure fallback; then the sentinel-payout bug. Validator now has None-propagation end to end + 43 tests.
2. SFT v3 (real corpus) → p̂ probes: saturated at bar 0.65, headroom at 0.9 → RL's job is top-decimal precision.
3. Run 1 (GDPO β=0.04) + run 2 (β-ablation): produced the capacity/diversity diagnostics above, and taught us every per-check trend curve was target-sampling noise (0/612 targets repeated).
4. Run-3 prep (5-channel uniform reward, closed-book data filtered to mid-band p̂ + gradeability-stable, fixed 30-target probe eval, `epsilon_high=5.0`, EvalModeGuard for a TRL eval bug) — all implemented and smoke-verified.
5. **Run 3 was launched, then killed** after a strategy review: pass@k n=200 came first and returned null (finding 4), so run 3's capacity verdict answers a question nobody needs answered *yet*. `run_gdpo_run3.sh` remains ready.
6. Now: the task-difficulty question is the bottleneck — **hardening probe is live** (below).

## Active work / next steps (ordered)

1. **Hardening probe (RUNNING in tmux `hardening`)**: `probe_hardening.py --n-targets 40 --samples 8 --out misc/hardening.json` — 5 conditions (baseline / temp_ceiling / inventory / atmosphere / combined) on the SFT policy. Pre-registered: capacity >40% → that condition is run-4's task; all <25% → factored-reward RL is unsupported on this task. Kimi fixed 7 bugs in it before launch (model default, parse crash, channel set, adherence schema, literature last-write-wins, resume, seeds) — see git diff.
2. **RS-SFT control arm** (scripted, unlaunched): `data_curation/build_rs_sft_dataset.py` — isolates "RL objective" from "verifier-filtered data". Runs after the hardening verdict.
3. **Validator temperature fix** (run-4 prep, after hardening for comparability): decouple ΔG evaluation T from the model's reported T; replace `temperature_plausible` with a target-conditioned check. User owns validator edits.
4. **Run 4** = hardening-winning condition + run-3 reward/config stack (`run_gdpo_run3.sh` pattern, new data).
5. Article drafting (user + Claude chat): the instrument-paper arc (capacity 14%, diversity mechanism, temperature hack, pass@k null) needs no new results.

## Operational playbook (each rule was learned the hard way)

- Run everything: `uv run python ...`; scripts outside repo root need `PYTHONPATH=.`; GPU work: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Secrets: `.env` (`set -a; source .env; set +a`) — never print it.
- **Long jobs ALWAYS in tmux** with tee + auto-restart (`run_gdpo.sh`, `run_gdpo_run3.sh`, `run_passk.sh` patterns; ntfy topic `mira-g5x7k2-status`). Agent-launched raw background processes die when the user logs off — the user has called this out repeatedly.
- **GPU memory rule** (32 GB card): full-vocab logits = batch × seq × 152k × bytes dominate. Per-device batch 1 at ~8–10k tokens; tune accum, never effective batch (16) or G (8). Generation ≈65% of RL step time.
- **git**: never commit/push/reset/rebase without the user explicitly asking. Stale 0-byte `.git/index.lock` reappears (editor plugin) — `rm` it if no git process runs. Committer identity is auto-guessed (fine).
- **Validator**: user is rewriting it on the side — check before big changes; small fixes + tests OK; bump `VALIDATOR_VERSION` on scoring-behavior changes.
- **Known traps**: `load_validator(formula_set, pd_index_path, project_root)` signature (old `pd_cache_path` kwarg dead); pd_index paths are repo-root-relative (pass `project_root=Path(".")`); `data/sft/` doesn't exist (`data/sft_old/`); `--model` is the BASE model — always pass `Qwen/Qwen3-8B`, never a checkpoint path (`--checkpoint` carries adapters); data/generation files are GBs — sample with `head -c`, never read fully.
- **TRL 1.8.0 + continuous batching + eval bug**: generation calls `unwrapped_model.train()` unconditionally (grpo_trainer.py:1685), misrouting later eval batches into the train branch. `EvalModeGuard` (core/observability.py) works around it — do not remove on TRL 1.8.0; re-test on any TRL upgrade.
- User preferences: explicit permission before edits in new areas; verify by execution, no fabrication; be candid about null/bad results; minimal diffs; dense replies.

## Getting up to speed (token-efficient read order)

1. This file.
2. `misc/some_claude_files/KIMI_NEW_CHAT_PRIMER.md` — the deep primer (58 lines): validator contract, probe inventory, run-3 spec detail, doc map.
3. On demand only: `gdpo_v4_next_steps_claude_recommendation.md` (capacity diagnostics D1–D5), `CLAUDE_CLOSED_BOOK_ANALYSIS.md` (closed-book + gradeability flap), `CLAUDE_RESPONSE_TO_RUN3_JUSTIFICATION.md` + `KIMI_ITEMS_2_3_5_PLAN.md` (the resequencing logic), `ANALYSIS_CHAIN_RESULTS.md` (temp sweep/attention/pass@k-15).
4. Data/results files (`runs/*/generations.jsonl`, `misc/*.json`) are large — inspect with `head -c`, `jq`-style one-liners, or the existing analyzers (`reward_geometry.py`, `run_debug_and_analysis/`).
5. Before proposing anything, grep the repo for prior art — most good ideas have a script already (`probe_*.py`, `reward_geometry.py`, `run_debug_and_analysis/*`).
