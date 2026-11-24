# Phase 20 results

Pre-registration: `docs/phases/PHASE20_PREREG.md`, committed in `2ac94de` (blob `15d89d8a`). Every analysis script asserts that blob is committed before it computes a P-20 number, and every results JSON records `prereg_commit`. Anything decided after that commit is labeled **[E]**. RL = on-policy GDPO; SFT = supervised training on a fixed set.

---

## 2026-11-17: Stage 0 (fixes, audits, gates)

Code: `experiments/sft.py --sft-full` (via `train.py sft --sft-full`), `core/data.py::build_sft_full_dataset`, `research/phase20_gate_t.py`, `research/phase20_gate_m.py`, `research/phase20_gate_r.py`, `tests/test_sft_masking.py`, `data_curation/phase20_sampling.py`, `research/phase20_verify_samples.py`, `scripts/phase20_sample_standalone.sh`, `scripts/run_phase20_chain.sh` + `scripts/phase20_stages/`. Audit: `docs/phases/PHASE20_FINDINGS_AUDIT.md`. RunPod procedure: `docs/phases/PHASE20_RUNPOD.md`.

### Pre-training handling decision: two capped completions (decided 2026-11-17, before any Phase 20 training)

Gate T's first run failed on 4 of 2,987 examples. They are the same two completions, each present in both `data/e3/e3_train.jsonl` and `data/e3c/e3c_train.jsonl`:

- **CaFe3O(PO4)3**
- **CsLuP2O7**

Each is a Base sample that hit the 8,192-token sampling cap without closing `</think>`. Its JSON answer sits inside the unclosed reasoning, and the text ends mid-sentence. For an assistant turn with no `</think>`, the Qwen3 chat template inserts an empty `<think>\n\n</think>\n\n` before the real `<think>` and appends `<|im_end|>`.

Decision (Gowtham): **render as sampled.** Those two rows train on the rendered generation prompt plus the raw sampled text, tokenized as-is. There is no template-inserted empty think block, no `</think>` and no `<|im_end|>` (the model never stopped). The loss covers those completion tokens only.

The rows are not dropped and the spec is not edited. Gate T check 4 is adjusted for exactly these rows: the loss text must equal the sampled text exactly, with nothing added. Every other row is checked unchanged.

Implementation: `core/data.py::build_sft_full_dataset` (`is_capped_completion`). The text is re-tokenized because the original sampled token ids were not stored.

### Gates

**Gate T (masking): PASS** after the handling decision above (`results/phase20/gate_t.json`, every row of every train and val file; `tests/test_sft_masking.py` passes). The first run, before the decision, failed only on the 4 capped rows.

| dataset | n (train / val) | max full length | median full length | median loss tokens | capped rows |
|---|---|---|---|---|---|
| RS-SFT | 283 / 12 | 8,886 | 5,468 | 4,621 | 0 |
| E3 | 505 / 22 | 9,043 | 5,843 | 5,002 | 2 |
| E3c | 505 / 22 | 9,031 | 5,843 | 5,002 | 2 |
| Corpus SFT | 1,217 / 157 | 14,648 | 4,825 | 4,328 | 0 |

**Gate M (memory): PASS, no fallback needed.** One optimizer step on each dataset's longest example through the real `--sft-full` path: peak allocated 12.1 GiB (RS-SFT, E3, E3c) and 14.3 GiB (Corpus SFT, 14,648 tokens) on the 32 GiB 5090. Trainer sequence length equals the rendered length in every case. TRL 1.8.0's default loss is already the chunked cross-entropy (`chunked_nll`), which the old runs used too.

Gate R results are added after each Stage 1 run.

| model | R1: trained tokens per example (incl. prompt) | dataset mean full length | rel. diff | R2: parse rate (64) | median completion tokens | clipped | |
|---|---|---|---|---|---|---|---|
| `sft-rssft-full` | 5,587.8 (logged step 25, 790 examples seen) | 5,582.3 | +0.10% | 100% | 4,815 | 0% | PASS |
| `sft-e3-full` | 5,862.2 (logged step 25, 793 examples seen) | 5,888.2 | −0.44% | 98.4% | 4,039 | 0% | PASS |
| `sft-e3c-full` | 5,843.8 (logged step 25, 793 examples seen) | 5,841.5 | +0.04% | 100% | 3,773 | 0% | PASS |

### Temperature mixing [E] (Gowtham, 2026-11-17, after the commit): Base and old RS-SFT on their T 0.9 vLLM files

`results/phase20/temp_vllm_E.json`. Base and old RS-SFT already have T 0.9 vLLM ASTRAL draws (`base_reference_astral`, `rssft_reference_astral`). Their 32-sample `resampling_rate` group gives n = 32, like every other row. No new sampling.

ASTRAL rule-picked hits (exact match, finding 18/22 metric):

| model | old value (HF T 1.0 file) | T 0.9 vLLM, n = 32 | T 0.9 vLLM, any of 200 |
|---|---|---|---|
| Base | 10/35 (clean 8/22) | **7/35 (6/22)** | 11/35 (8/22) |
| old RS-SFT | 10/35 (9/22) | 10/35 (8/22) | — |
| E3 (already vLLM 0.9) | 13/35 (10/22) | unchanged | — |
| E3c (already vLLM 0.9) | 11/35 (9/22) | unchanged | — |

- **Finding 31:** at matched temperature and backend, E3 13/35 and E3c 11/35 compare with Base 7/35, not 10/35. One 32-sample draw is noisy: Base's 200-sample draw reaches 11/35.
- **Finding 33's RS-SFT row:** the reported 48.2% carbonate share and 10/35 hits equal the vLLM `resampling_rate` group exactly: carbonate 48.2% per alkali choice, against 45.9% in the HF file. So that row was already computed on vLLM data, and RS-SFT's ckpt100 and ckpt200 comparisons in it are like for like. Only the step-300 points (HF files) are mixed; Stage 5 resamples those.
- **18A / 18E:** these already read Base and RS-SFT from their vLLM reference files. Their mixing comes only from the four HF models, which Stage 5 resamples. The rerun is in `tempfix_*`.

---

## 2026-11-22: Stage 3 done; all remaining sampling moved to this machine

RL `gdpo-e1-ranker-rssft-full` finished at 04:06 UTC (checkpoints 100, 200 and 300 kept). The RunPod reservations for Stages 2 and 5 were cancelled (Gowtham): every sample set is now drawn locally with the same script and settings (vLLM 0.23.0, T 0.9, top-p 0.95, 8,192 tokens, seed 42), so the sampling hardware is the same for every Phase 20 set.

---

## 2026-11-23: Stage 2b (SFT side) — Report 2

Stage 2 sampled the three new SFT models locally (accepted by `research/phase20_verify_samples.py`). Stage 2b ran Blocks G, E and F1 on the 2b step table (`results/phase20/p19_2b/`) and the SFT-side CPU analyses.

**Two code bugs, fixed before this report.**
- (1) `load_recipes20` recursed into itself once Phase 18's `load_recipes` was patched, so transport and posterior failed. 
- (2) `coverage.json` and `astral_hits.json` were Stage 0 smoke outputs committed in `534b117`, so the write-guard refused them. These outputs are now stage-suffixed (`*_2b.json`, `*_final.json`); the committed smoke files are untouched.

**Data-handling error, affects every `sft-rssft-full` row in this section.** The 2b analyses ran while Stage 4 was still writing `sft_rssft_full_reference_astral.jsonl`, and the recipe loader took the half-written file (16 of 35 ASTRAL targets) as the RS-SFT reference. Every `sft-rssft-full` number below that comes from that loader is therefore **not valid**: P-20-1's S2 row, its Block G gate value and its coverage share. The loader now ignores the reference until `STAGE_S4_sample_ref.done` exists. Those rows are recomputed at the final analysis on the complete 232-sample reference. Rows for E3, E3c and Base never read that file and are unaffected. Hits and alkali shares use the complete 32-sample `sft_rssft_full_*_n32` files and are unaffected.

**Block G (gate, median Spearman ≥ 0.4): passes** — Base 0.462, `sft-e3-full` 0.415, `sft-e3c-full` 0.482 (`sft-rssft-full` 0.414, invalid as above).

**P-20-1 (SFT adds support), E3 and E3c rows: FAILS.** P-18A-2 measure, same code, λ, floor and targets (62).

| step | new-support mass − Base floor, full-sequence SFT | 95% CI | old (Phase 18, 1,024-token SFT) |
|---|---|---|---|
| Base → `sft-e3-full` | −0.0036 | [−0.0106, +0.0040] | +0.0111 [+0.0011, +0.0224] |
| Base → `sft-e3c-full` | +0.0030 | [−0.0052, +0.0118] | +0.0204 [+0.0073, +0.0344] |

Both CIs include 0. With full-sequence training, E3 and E3c no longer add new-support mass above Base's resampling floor on this measure; the old truncated runs did.

**P-20-3 (exact reach), SFT steps: FAILS for both** (CI must lie entirely above 0): Base → `sft-e3-full` median +0.685, CI [−0.0002, +1.015], 31 usable targets; Base → `sft-e3c-full` +0.535, CI [−0.091, +0.678], 31 usable. Both point estimates are positive; both CIs touch or cross 0.

**Block E and F1 on the SFT steps (inputs to P-20-4/5, which need Stage 4):**

| step | L50 [CI] | old L50 | R(4) | old R(4) | R(4) random-subspace null max |
|---|---|---|---|---|---|
| Base → `sft-e3-full` | 10 [6, 14] | 6 | 0.284 | 0.733 | 0.041 |
| Base → `sft-e3c-full` | 14 [10, 24] | 6 | 0.011 | 0.764 | 0.041 |
| RS-SFT → Phase 12 @300 (old, reused) | 26 [26, 28] | 26 | 0.385 | 0.385 | 0.039 |
| RS-SFT → E1 @300 (old, reused) | 16 [14, 16] | 16 | 0.552 | 0.552 | 0.039 |

Full-sequence SFT is written deeper (L50 10–14 vs 6) and is less concentrated in 4 directions (R(4) 0.28 and 0.01 vs 0.73–0.76) than the truncated runs. `sft-e3c-full`'s R(4) is at the random-subspace null.

**Measurements.**

- **ASTRAL rule-picked hits (n = 32, T 0.9 vLLM):**

  | | new (full sequence) | old |
  |---|---|---|
  | RS-SFT | 6/35 (5/22 clean) | 10/35 (8/22) |
  | E3 | 10/35 (7/22) | 13/35 (10/22) |
  | E3c | 9/35 (7/22) | 11/35 (9/22) |
  | Base | — | 7/35 (6/22) |
  | E1-full @100 / @200 / @300 | 10/35 (8/22), 10/35 (9/22), 9/35 (7/22) | E1 @300: 14/35 |

  Every full-sequence SFT model is below its truncated predecessor; `sft-rssft-full` (6/35) is below Base (7/35).

- **Alkali-source share (per alkali choice), carbonate / bare oxide:**

  | | carbonate | bare oxide |
  |---|---|---|
  | `sft-e3-full` | 78.6% | 9.6% |
  | `sft-e3c-full` | 81.8% | 8.5% |
  | `sft-rssft-full` (n32) | 79.2% | 10.2% |
  | E1-full @300 | 69.8% | 17.5% |
  | old RS-SFT (vLLM `resampling_rate`) | 48.2% | 24.2% |
  | old E1 @300 (finding 30) | 18.3% | — |
  | Base (ledger `L8.alkali_share`) | 72.9% | 9.1% |

  Source: `results/phase20/alkali_source_share.json`; the run started before the suffix fix, so it is the 2b file.

  The full-sequence SFT models sit at or slightly above Base's carbonate share, not below it as old RS-SFT did. RL from the new RS-SFT still moves toward bare oxides (79.2 → 69.8% carbonate), but from a much higher starting point than the old E1 (45.9 → 18.3%).

- **Probe-set coverage:**

  | new model | share in probe set | old model it replaces | share |
  |---|---|---|---|
  | `sft-e3-full` | 99.0% | E3 | 99.5% |
  | `sft-e3c-full` | 98.7% | E3c | 99.8% |
  | E1-full @300 | 98.7% | E1 @300 | 99.5% |

  The 671 Phase 19 probe strings cover the new models nearly as well as the old ones.

**Old findings touched so far (provisional until the final analysis):** finding 31 ("self-distillation adds recipes") — **changed**: on the P-18A-2 measure the full-sequence E3/E3c do not clear Base's floor, and their hits fall (13→10, 11→9). Finding 18 (RS-SFT retains 10/35) — **changed, provisional**: full-sequence RS-SFT reads 6/35 at n = 32.

---

## 2026-11-24: Stages 4–7 — final analysis (Report 3)

The chain finished at 21:30 UTC. One final step (`weights`, the 18C rerun) first crashed: the Phase 20 wrapper dropped step S6, which the Phase 18 code's P-18C-2 test hard-codes. The fix keeps S6 (old RS-SFT → Phase 12 @300) as an unchanged reference step on its own old init. It needed one additive line in `research/phase18_c_weight_geometry.py` (load any trajectory init); Phase 18's own behavior is unchanged. Check: S5 and S6 reproduce Phase 18's P-18C-2 values exactly.

`sft-rssft-full` rows below use the complete 232-sample reference, replacing the invalid 2b rows. Every new model passes Block G (≥ 0.4): Base 0.462, `sft-rssft-full` 0.526, `sft-e3-full` 0.415, `sft-e3c-full` 0.482, E1-full @300 0.439. No P-20 result is labeled exploratory.

Stage 6 ran: `sft-corpus-full` passed Gate R. R1 +0.58% against the 5% bar; R2 parse 100%, median 4,073 completion tokens, 0% clipped.

### Predictions

| | prediction | result |
|---|---|---|
| P-20-1 | each full-sequence SFT clears Base's floor (P-18A-2) | **FAILS** for all three |
| P-20-2 | E1-full @100/200/300 does not exceed `sft-rssft-full`'s floor | **HOLDS** |
| P-20-3 | reach: RL CI < 0, SFT CIs > 0 | **FAILS** for all three steps |
| P-20-4 | L50(RL) > L50 of both SFT steps | **HOLDS** |
| P-20-5 | R(4) of both SFT steps > R(4)(RL) | **FAILS** (E3 holds, E3c does not) |

**P-20-1 (P-18A-2 measure, 62 targets; new-support mass − Base floor):**

| step | full-sequence | 95% CI | old (1,024-token SFT) |
|---|---|---|---|
| Base → `sft-rssft-full` | −0.0012 | [−0.0066, +0.0045] | +0.0315 [+0.0180, +0.0447] |
| Base → `sft-e3-full` | −0.0036 | [−0.0107, +0.0039] | +0.0111 [+0.0011, +0.0224] |
| Base → `sft-e3c-full` | +0.0030 | [−0.0051, +0.0120] | +0.0204 [+0.0073, +0.0344] |
| Base → `sft-corpus-full` (S1, not predicted) | **+0.0676** | [+0.0400, +0.0992] | +0.0579 [+0.0303, +0.0903] |
| `sft-rssft-full` → E1-full @300 (S7, not predicted) | +0.0029 | [−0.0025, +0.0089] | −0.0040 [−0.0096, +0.0021] |

Training on the model's own samples, with the full sequence, no longer adds support above Base's resampling floor. Training on the external corpus still does, by about the same amount as before.

**P-20-2 (finding 33's procedure; new-recipe rate vs `sft-rssft-full`'s floor):**

| | ckpt 100 | ckpt 200 | ckpt 300 | floor |
|---|---|---|---|---|
| ASTRAL | 9.8% [5.3, 14.0] | 9.9% [5.7, 14.1] | 10.8% [6.8, 14.8] | 8.6% [4.3, 13.1] |
| held-out | 6.0% [2.4, 9.6] | 9.3% [4.3, 13.9] | 14.7% [7.6, 20.5] | 16.1% [9.8, 22.1] |

No checkpoint's interval lies above the floor's interval. Held-out rises 6.0 → 14.7% across checkpoints, still under the floor.

**P-20-3 (exact reach, median centered Δ unsampled − sampled):**

| step | median | 95% CI | usable targets |
|---|---|---|---|
| Base → `sft-e3-full` | +0.685 | [−0.0002, +1.015] | 31 |
| Base → `sft-e3c-full` | +0.535 | [−0.091, +0.678] | 31 |
| `sft-rssft-full` → E1-full @300 | −0.084 | [−0.236, +0.268] | 34 |

SFT points are positive and the RL point is negative, as predicted, but no CI clears 0.

**P-20-4 / P-20-5 (Block E, F1):**

| step | L50 [CI] | R(4) | L* |
|---|---|---|---|
| Base → `sft-e3-full` | 10 [6, 14] | 0.284 | 26 |
| Base → `sft-e3c-full` | 14 [10, 24] | 0.011 | 28 |
| `sft-rssft-full` → E1-full @300 | 24 [22, 26] | 0.134 | 30 |

### Measurements

**ASTRAL rule-picked hits, n = 32, T 0.9 vLLM unless noted:**

| model | new (full sequence) | old |
|---|---|---|
| Base | — | 7/35 (6/22) |
| Corpus SFT | 2/35 (2/22) | 0/35 (0/22) |
| RS-SFT | 6/35 (5/22) | 10/35 (8/22) |
| E3 | 10/35 (7/22) | 13/35 (10/22) |
| E3c | 9/35 (7/22) | 11/35 (9/22) |
| E1 @100 / @200 / @300 | 10 (8), 10 (9), 9 (7) | @300: 13/35 (10/22), Stage 5 |

- **Temperature fix (Stage 5, old models at T 0.9 vLLM vs their HF T 1.0 files):**

  | model | HF T 1.0 | vLLM T 0.9 |
  |---|---|---|
  | Corpus SFT | 0/35 | 0/35 |
  | GDPO-v3 @300 | 2/35 | 0/35 |
  | Phase 12 @300 | 14/35 (12/22) | **12/35 (10/22)** |
  | E1 @300 | 14/35 (11/22) | **13/35 (10/22)** |

  The 18A rerun at matched temperature (`tempfix_transport_exact.json`) moves no step's verdict: S5 −0.0047, S6 −0.0106, S7 −0.0042, each within 0.0015 of before.

**Alkali source, per alkali choice (carbonate / bare oxide):**

| model | carbonate | bare oxide |
|---|---|---|
| `sft-rssft-full` | 79.2% | 10.2% |
| E1-full @300 | 69.8% | 17.5% |
| `sft-e3-full` | 78.6% | 9.6% |
| `sft-e3c-full` | 81.8% | 8.5% |
| old RS-SFT (vLLM) | 48.2% | 24.2% |
| old E1 @300, T 0.9 | 20.1% | 50.5% |
| Phase 12 @300, T 0.9 | 16.3% | 57.6% |

**Old vs new RS-SFT (18E kernel MMD):** ASTRAL 0.334 [0.290, 0.380] against a resampling floor of 0.128; held-out 0.178 [0.137, 0.220] against a floor of 0.084. The two RS-SFTs are different distributions.

**Probe coverage (share of samples in the 671 probe strings):** `sft-rssft-full` 98.6% (old 99.6%), `sft-e3-full` 99.0% (99.5%), `sft-e3c-full` 98.7% (99.8%), E1-full @300 98.7% (99.5%), `sft-corpus-full` 95.1% (97.3%).

**Uniform shift, RL step (Δ mean / Δ RMS on answer strings):** 0.27 / 0.93 for the new step, against 0.48 / 1.72 for the old one.

**18C weights, P-18C-2 (share of modules with non-decreasing effective rank over 100 → 200 → 300):** E1-full 0.52, against old E1 0.968. S5 0.302 and S6 0.452 are unchanged.

### Old findings touched

| finding | verdict | why |
|---|---|---|
| 16 (Corpus SFT destroys Base's hits) | **confirmed** | full-sequence Corpus SFT 2/35 against Base 7/35 |
| 18 (RS-SFT retains Base's 10/35) | **changed** | full-sequence RS-SFT 6/35 against Base 7/35 at matched temperature; the retention was a property of 1,024-token training |
| 22 (Phase 12 amplified 10 → 14/35) | **changed (weaker)** | at matched temperature Phase 12 @300 reads 12/35 against old RS-SFT 10/35 |
| 24 / 26 (RL shifts alkali sourcing to bare oxide) | **confirmed, smaller** | E1-full moves carbonate 79.2 → 69.8%, against 48.2 → 20.1% for the old pair |
| 30 (E1 reaches 14/35) | **changed** | 13/35 at matched temperature; from full-sequence RS-SFT E1 reaches 9–10/35 |
| 31 (self-distillation adds recipes) | **reversed** | full-sequence E3/E3c no longer clear Base's floor on P-18A-2; the rarity filter still adds nothing |
| 33 (RL's new-recipe rate stays under the floor) | **confirmed** | P-20-2 holds from the new init |
| Phase 18 P-18A-2 (SFT S2–S4 positive) | **reversed** | only Corpus SFT (S1) remains positive |
| Phase 19 E/F1 (SFT written shallow, low-dimensional) | **changed** | full-sequence SFT L50 10–14 (was 6), R(4) 0.01–0.28 (was 0.73–0.76); RL is still written deeper (P-20-4) |
| Phase 18 P-18C-2 for E1 | **changed** | 0.968 → 0.52 non-decreasing |

**Net:** the one robust SFT-vs-RL contrast that survives is *where* the change is written (P-20-4). The "SFT on own samples adds recipes, RL does not" contrast does not survive full-sequence training: neither adds support above floor. External-corpus SFT still does.
