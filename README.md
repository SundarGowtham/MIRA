# MIRA — Materials Intelligence Retrieval Architecture

**Fine-tuning a language model to propose inorganic synthesis routes, and measuring
whether reinforcement learning actually taught it anything.**

*This repository is the working record of a five-month investigation. The headline
result is a well-characterised null, plus the diagnostics that explain it and an
external validation against a robotic laboratory's experimental data.*

📄 Paper: *(link)* · 🌐 Interactive writeup: *(link)* · 📓
[Full timeline](docs/TIMELINE.md)

---

## The question

Given a target compound, propose a solid-state synthesis route: which precursors, in
what amounts, heated to what temperature, under what atmosphere.

The route can be **checked by physics** rather than against an answer key — does the
reaction balance, do the precursors exist, is the target thermodynamically reachable at
that temperature. That property is what makes reinforcement learning from verifiable
rewards plausible here, the same way it works for mathematics and code.

The specific research question: the verifier emits **ten separate physical checks**.
Standard RL sums them into one number before computing the gradient, which destroys
information about *which* sub-skill was good. Does keeping the reward factored (GDPO,
per-channel normalisation) let RL learn which sub-skill to fix?

## The answer

**No — because the reward was never factored in practice, and we can measure exactly
why.**

| finding | number |
|---|---|
| RL vs SFT, pass@k on 200 held-out targets | gap **shrinks** with k: +2.4 @ k=1 → +1.0 @ k=16 (McNemar p = 0.77) — **sharpening, not expansion** |
| Reward capacity (within-group z-variance / channels) | **14%** — 8 of 10 channels constant within group |
| Verifier agreement with robot-measured phase purity | **17/34 = 50.0%** — chance |
| Base model proposes the experimentally-superior route | **10/35** targets |
| After supervised fine-tuning | **1/35** (paired exact McNemar p = 0.004–0.012) |
| After RS-SFT from base instead | **10/35** — capability retained, and conventional-route rate *drops* below base's own |
| Novel precursors ASTRAL's robot found superior, vs. their training-corpus frequency | 3 of 9 appear **zero** times in 17,600 corpus routes; mean 2.6 vs. **674.8** for the conventional precursors the model defaults to |

The short version: **the training corpus, the verifier and the model formed a closed
loop, and the loop excluded the right answers.** Fine-tuning on a corpus of published
successes collapsed the model onto conventional practice; RL, being on-policy, could
only reweight what the policy already sampled.

See [`docs/DECISIONS.md`](docs/DECISIONS.md) for every pre-registered prediction in
this project and what actually happened — several went against the hypothesis that
motivated them.

## Reproducing the headline numbers

Every number in the table above is computed from a tracked artifact in
[`results/`](results/) — reading them needs no GPU. Re-scoring a route from scratch
(not just reading these numbers) needs the Materials Project phase-diagram cache,
which is not tracked here — see [`results/README.md`](results/README.md#honest-limit-on-reproducibility).

```bash
# pass@k, 200 targets, paired bootstrap + exact McNemar
PYTHONPATH=. uv run python research/analyze_passk.py --results results/passk_n200.json

# reward capacity + per-channel diagnostics (needs the full generation archive,
# not tracked here — hardening.json in results/ has the already-computed numbers)
PYTHONPATH=. uv run python reward_geometry.py --gens runs/<run>/generations.jsonl --strata data/rl --all

# external validation: five models' proposed precursor sets vs. the ASTRAL robot data
PYTHONPATH=. uv run python research/build_astral_5model_summary.py
```

## What's here

```
core/              validator, reward construction, model loading, observability
experiments/       sft.py · grpo.py · gdpo.py · sft_grpo.py
train.py           CLI entrypoint for all training
validator.py       the physics oracle — 10 checks, no reference-recipe comparison
tests/             43-assertion conformance suite for the validator
research/          diagnostics, probes, one-off analyses (~50 scripts)
scripts/           tmux launchers for long runs
data_curation/     corpus triage, split construction, RS-SFT dataset builder
results/           analysis outputs — the evidence for every claim above
docs/              phase-by-phase narrative, timeline, decision record
```

**Start here:** [`docs/TIMELINE.md`](docs/TIMELINE.md) is the chronological record of
what was tried, in what order, and why — including the interventions that failed and
the measurements that turned out to be measuring nothing.
[`docs/DECISIONS.md`](docs/DECISIONS.md) is the shorter version: one row per
pre-registered prediction and its outcome.

## The verifier

`validator.py` scores a route on ten physical checks — stoichiometric balance, amount
accuracy, charge neutrality, precursor existence, operation ordering, temperature
plausibility, thermodynamic favourability (ΔG at the route's own maximum temperature),
target stability, chemical-potential compatibility with the declared atmosphere, and
output format.

**No ground-truth recipe comparison anywhere.** Every check is physics or constraint
based, so the oracle can grade routes nobody has written. Backed by a cached
point-in-time snapshot of Materials Project phase diagrams (485 shards, 19,861 diagrams).

Run the conformance suite:

```bash
PYTHONPATH=. uv run python tests/test_validator.py
```

## Setup

```bash
uv sync                              # or: pip install -r requirements.txt
cp .env.example .env                 # MP_API_KEY, WANDB_API_KEY, etc.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Trained on a single 32 GB GPU. Qwen3-8B with QLoRA (r=16, α=32). Long runs go through
`scripts/` in tmux — see the launchers for the auto-restart pattern.

## Training

```bash
# supervised fine-tuning
uv run python train.py sft --adapter qlora --model Qwen/Qwen3-8B \
    --data-dir data/sft_v3 --lora-r 16 --lora-alpha 32

# GDPO — factored reward, per-channel normalisation
uv run python train.py gdpo --adapter qlora --model Qwen/Qwen3-8B \
    --init-from runs/<sft-checkpoint>/final --scorer validator
```

Hyperparameters for every reported run are recorded inline in
[`docs/phases/`](docs/phases/) alongside the results they produced.

## A note on negative results

Most of this repository documents things that did not work: five interventions that
failed to move reward capacity, an interpretability experiment that was blocked by a
degeneracy rather than answered, a headline result retracted when re-run at proper
sample size, and a verifier that turned out to agree with experiment at chance.

Those are kept deliberately, with the reasoning that produced them. The methodology is
the contribution.

## Citation

```bibtex
@misc{mira2026,
  title  = {...},
  author = {...},
  year   = {2026},
  eprint = {...},
}
```

## License

[AGPLv3](LICENSE)
