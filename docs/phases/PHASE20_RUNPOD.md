# Phase 20: running a sampling stage on RunPod

This moves one or more sampling stages to a rented GPU while manifold trains. The chain accepts the returned files only if every row's recorded settings match the spec (`research/phase20_verify_samples.py`). When they do, it marks the stage done and skips it.

## Which stages can go, and when

| stage | what | needs these adapter folders | earliest start | approx. 5090-hours |
|---|---|---|---|---|
| `stage5` | old corpus SFT, GDPO-300, Phase 12 @300, E1 @300; ASTRAL only; 32/target | `runs/sft-qlora-sft-v3-2nd-rank16/final`, `runs/gdpo-qlora-gdpo-v3/checkpoint-300`, `runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300`, `runs/gdpo-qlora-gdpo-e1-ranker-rssft/checkpoint-300` | **now** (old models) | ~6 |
| `stage2` | `sft-rssft-full`, `sft-e3-full`, `sft-e3c-full`; ASTRAL + held-out; 32/target | `runs/sft-qlora-sft-rssft-full/final`, `runs/sft-qlora-sft-e3-full/final`, `runs/sft-qlora-sft-e3c-full/final` | after Stage 1 | ~9.5 |
| `stage4_ref` | `sft-rssft-full` reference, 200 + 32 per target | `runs/sft-qlora-sft-rssft-full/final` | after the first Stage 1 model | ~23 |
| `stage4_e1` | E1-full @100/@200/@300; 32/target | `runs/gdpo-qlora-gdpo-e1-ranker-rssft-full/checkpoint-{100,200,300}` | after Stage 3 | ~9.5 |

The hours are measured on manifold's 5090 from the Phase 16 logs. The old RS-SFT reference took ~23 h (11 h ASTRAL + 12 h held-out), not ~9.

`stage5` and `stage4_ref` are the best candidates because they don't block anything on manifold. `stage4_ref` can start as soon as `sft-rssft-full` exists, which is the first model Stage 1 trains.

## 1. Pod

- GPU: an RTX 5090 (32 GB) matches manifold. Any ≥ 32 GB GPU works, because the file records the GPU name. A different GPU gives draws from the same distribution, but not byte-identical draws.
- Template: a CUDA 13 / PyTorch image. Disk: ≥ 60 GB (the Qwen3-8B weights are 16 GB, plus the env and outputs).

## 2. Environment (on the pod)

vLLM is **not** in the repo's `uv.lock`. On manifold it lives in a separate env, `~/envs/mira-vllm`, so `uv sync` will not install it. Recreate that env from its frozen package list instead:

```bash
git clone git@github.com:SundarGowtham/MIRA.git && cd MIRA      # or https + token
git checkout <the commit manifold is on>                           # same code as the chain
python3.12 -m venv ~/envs/mira-vllm
~/envs/mira-vllm/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r scripts/phase20_vllm_requirements.txt
~/envs/mira-vllm/bin/python -c "import vllm; print(vllm.__version__)"   # must print 0.23.0
export HF_TOKEN=...            # only if the HF download is rate-limited; Qwen3-8B is public
```

If pip can't resolve a pinned line, install `vllm==0.23.0` with `torch==2.11.0` on its own, then `transformers==4.57.6`. The acceptance check only requires vLLM 0.23.0.

## 3. Send the adapter folders (manifold → pod)

Each adapter folder is ~100 MB. On manifold, from the repo root, send each folder the stage needs. For example, for `stage5`:

```bash
runpodctl send runs/sft-qlora-sft-v3-2nd-rank16/final
runpodctl send runs/gdpo-qlora-gdpo-v3/checkpoint-300
runpodctl send runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300
runpodctl send runs/gdpo-qlora-gdpo-e1-ranker-rssft/checkpoint-300
```

On the pod, receive each one into **the same relative path**:

```bash
mkdir -p runs/gdpo-qlora-gdpo-v3 && cd runs/gdpo-qlora-gdpo-v3 && runpodctl receive <code> && cd -
# ...repeat per folder; the result must be e.g. MIRA/runs/gdpo-qlora-gdpo-v3/checkpoint-300/adapter_model.safetensors
```

To check the paths, `~/envs/mira-vllm/bin/python data_curation/phase20_sampling.py --job stage5 --list-outputs` prints each output file, its adapter path and its expected row count.

## 4. Run (on the pod, in tmux)

```bash
tmux new -s p20
bash scripts/phase20_sample_standalone.sh stage5      # or stage2 / stage4_ref / stage4_e1
```

It is resume-safe, so rerun the same command after any crash. It ends by running the acceptance check and printing `ACCEPTED` or `REJECTED`.

## 5. Send the results back (pod → manifold)

```bash
# pod
cd MIRA && tar czf p20_stage5.tgz data/novelty/phase20/*_t09vllm.jsonl run_logs/phase20_sample_stage5.log
runpodctl send p20_stage5.tgz
# manifold, repo root
runpodctl receive <code> && tar xzf p20_stage5.tgz
PYTHONPATH=. uv run python research/phase20_verify_samples.py stage5
```

The chain runs the same check before each sampling stage. If the files are present and accepted, it writes `runs/phase20/STAGE_<stage>.done` and moves on. A rejected or partial file is not overwritten. The chain then samples locally and resumes from the targets already present, so partial pod output is not wasted, as long as it was drawn with the same settings.

Do not drop a pod file into place while the chain is sampling that same stage locally, because both would append to one file. Check `tmux attach -t phase20_chain` first.

## 6. Stop the pod

Stop the pod as soon as the files are back and accepted. A pod that runs out of balance stops mid-stage, and the partial file is still usable (see step 5).
