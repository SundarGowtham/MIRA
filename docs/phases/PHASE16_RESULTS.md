# Phase 16 results

Per `misc/PHASE16_INSTRUCTIONS.md`. Dated entries as they land.

## 2026-10-03 — §1 rule 1 check FAILED: pre-existing ASTRAL contamination in active training data

**Check that failed**: the mandated unit test (`tests/test_no_astral_circularity.py`,
§1 rule 1 — "no ASTRAL target formula appears in any training or sampling
prompt file") fails. Per the instructions' own protocol ("if a check
fails, stop... and wait for Gowtham"), stopping here rather than
proceeding to E3 sampling or anything else that depends on this data.

**This is a pre-existing condition, not something Phase 16 introduced.**
The contaminated files were all built well before this phase (SFT v3,
RL run3, and the general RL pool all predate Phase 16's own changes).
The rule-1 test is simply the first time anyone has checked for it.

**Exact scope, by file** (contaminated records / total, which ASTRAL
target(s)):

| file | contaminated / total | targets |
|---|---|---|
| `data/rs_sft/rs_sft_train.jsonl` | **0 / 283** | — |
| `data/rs_sft/rs_sft_val.jsonl` | **0 / 12** | — |
| `data/rl_run3/rl3_train.jsonl` | 0 / 1200 | — |
| `data/rl_run3/rl3_val.jsonl` | 1 / 200 | `Na2Al2B2O7` |
| `data/rl_run3/rl3_probe.jsonl` | **1 / 30** | `Na2Al2B2O7` |
| `data/sft_v3/{train,sft_train}.jsonl` | 3 / 1217 each | `KTiNbO5`, `LiMnPO4`, `Li3V2(PO4)3` |
| `data/sft_v3/format_only_train.jsonl` | 2 / 300 | `KTiNbO5`, `LiMnPO4` |
| `data/sft_v3/{val,test,sft_val,sft_test,format_only_val}.jsonl` | 0 each | — |
| `data/rl/{train,rl_train}.jsonl` | 9 / 6368 each | `Li3Fe2(PO4)3`, `NaSrBO3`, `KBaPO4`, `LiMgPO4`, `LiZnBO3`, `Li3Sc2(PO4)3`, `KNbWO6`, `Li2CuP2O7`, `KMgPO4` |
| `data/rl/{val,rl_val}.jsonl` | 1 / 200 | `Na2Al2B2O7` |

**The single most consequential dataset for Phase 16 — `data/rs_sft/` itself,
RS-SFT's actual training set, which every Phase 16 arm (E1/E2/E3/E4) either
inits from or distills further from — is clean, 0/295 contaminated.** The
Phase 11/12/16 RS-SFT lineage was not trained on any ASTRAL target.

**What IS affected**: full SFT (`data/sft_v3`, 3/1217 ≈ 0.25%, used in the
Phase 10/11 "full SFT" comparison arm) and, more notably, the **fixed
30-target probe set** (`data/rl_run3/rl3_probe.jsonl`, 1/30 = **3.3%** —
`Na2Al2B2O7`). This probe set is re-evaluated every N steps throughout
every GDPO run (Phase 12 and the planned Phase 16 E1/E2/E4 runs) to
produce the per-check training-health curves `CLAUDE.md` and
`docs/phases/PHASE12_RESULTS.md` report from. One target out of 30 being
an ASTRAL target is a real data-hygiene gap, not previously flagged
anywhere in this project, though its effect on any already-reported
32-target-averaged probe metric is necessarily small (1/30 weight).

**Not fixed or filtered here.** Removing records from datasets that
already produced completed, reported Phase 11/12 results is a decision
with real retroactive implications (does this change how those numbers
should be read, does the probe set need to be rebuilt and old curves
re-labeled, etc.) that is Gowtham's call, not something to decide
unilaterally mid-Phase-16. `tests/test_no_astral_circularity.py` is left
failing honestly rather than narrowed in scope to pass — scoping it to
only the directories real experiment code reads (not literally every
`.jsonl` in the repo, which also turned up one confirmed-dead legacy file,
`data/processed/reasoning_traces.jsonl`, excluded on that verified basis)
already reflects the real training/sampling surface; the remaining
failures are real.

ntfy sent to `mira-g5x7k2-status` at the time of this finding.

**Everything else in pass 2 items 1-2 (mira-vllm rebuild, standalone
smoke test, `core/novelty_reward.py` + its own tests) passed and is
unaffected by this** — it is the project-wide circularity guardrail
specifically that is failing, not novelty_reward.py's own tests (all of
which pass). Pending Gowtham's decision on this before any pre-registration
drafting or data-touching step proceeds.
