# Results — evidence for every headline number

One line per file: what produced it, on what n, which claim it supports.
Regenerate any file from its source script/data rather than hand-editing.

| file | produced by | n | supports |
|---|---|---|---|
| `passk_n200.json` | `probe_passk.py` | 200 held-out targets, k up to 16, base/SFT/GDPO-300 | pass@k sharpening result (gap shrinks +2.4→+1.0 with k) |
| `hardening.json` | `probe_hardening.py`, 5 conditions | 40 targets × 8 samples × 5 conditions | reward capacity 14–24%, the five failed interventions |
| `ranker_capacity_probe.json` | `probe_hardening.py --scorer ranker` | 40 targets × 8 samples | ranker (Arm B) capacity probe |
| `astral_validator_correlation.json` | `run_debug_and_analysis/astral_validation.py` | 35 ASTRAL targets × 2 routes | validator vs. robot-measured phase purity, Spearman ρ=0.228, 17/34=50.0% agreement |
| `ranker_v2_astral_gate.json` | `run_debug_and_analysis/ranker_v2_astral_gate.py` | 35 ASTRAL targets × 2 routes | ranker v2 external gate result: 21/35=60.0%, FAILED the pre-registered 24/35 bar |
| `corpus_coverage.json` | `run_debug_and_analysis/astral_corpus_coverage.py` | 17.6k Kononova corpus routes | ASTRAL-winning precursors average 2.6 corpus occurrences vs. 674.8 for conventional ones |
| `astral_gen_n32_base.json` | `run_debug_and_analysis/astral_model_generations.py --checkpoint base --n-samples 32` | 35 targets × 32 samples | base: 10/35 predicted-set hits, 24/35 conventional |
| `astral_gen_n32_sft.json` | same script, SFT checkpoint | 35 × 32 | full SFT: 1/35 predicted, 32/35 conventional — the collapse |
| `astral_gen_n32_gdpo300.json` | same script, GDPO-300 checkpoint | 35 × 32 | GDPO-300: 3/35 predicted, 31/35 conventional — partial, non-significant recovery |
| `astral_gen_n32_format_only.json` | same script, format-only-SFT checkpoint | 35 × 32 | format-only SFT: 11/35 predicted — schema installation alone doesn't collapse the capability |
| `astral_gen_n32_rs_sft.json` | same script, RS-SFT-from-base checkpoint | 35 × 32 | RS-SFT: 10/35 predicted, 17/35 conventional (lowest of all models) |
| `astral_5model_n32.json` | `run_debug_and_analysis/build_astral_5model_summary.py`, aggregates the five files above | — | the headline comparison table in the README, computed not hand-typed |
| `pass3_interp_probes.json` | interpretability probes A/B/C (`manifold_visualization/act_geo/`) | 200 targets, 5-fold CV | binary hit/miss is linearly probeable (AUC ~0.73–0.77 mid-late layers); continuous pass@1 is not |
| `astral_validation_set.json` | Chen/Cross/Sun, *Nature Synthesis* 2024 (arXiv 2304.00743) | 35 targets, traditional + predicted precursor sets, robot-measured phase purity | input data (not an output) for every ASTRAL-derived number above |

## Not tracked (too large or not evidence)

- `runs/*/generations.jsonl` — full per-completion generation archives, hundreds of MB per run. Regenerable from the checkpoint + `train.py`/eval scripts; not needed to verify the aggregate numbers above.
- `misc/kononova_triage_results{2,3}.json` — raw 6–7 MB corpus-triage data blobs, an input to data curation, not a result.
- `misc/*chat_snippet*.txt` — raw planning-chat transcripts, working notes, not evidence.
- `runs/` (checkpoints) — multi-GB adapter weights; not part of this repo's evidence trail.

If a number in the README or `docs/` cannot be traced to a file in this
directory, that is a bug in the docs, not a missing file — flag it.

## Honest limit on reproducibility

The JSON files here let you verify every headline number by reading data,
with no GPU. They do not let you *re-score a route from scratch*: that
needs the Materials Project phase-diagram cache (`data/cache/`, 19,861
shards) and the corpus (`data/raw/synthesis_clean.json`, 17.6k routes),
neither of which is tracked here — both are large, and the phase-diagram
cache is a point-in-time MP snapshot, not something to vendor into a git
repo. Re-scoring or re-generating requires building that cache locally
(see the main README's setup section) or requesting it directly.
