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
