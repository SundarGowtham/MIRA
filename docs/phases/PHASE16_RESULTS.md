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

## 2026-10-03 — resolved: this was test-set contamination, not rule-1 circularity. Erasure and carbonate findings survive on the clean subset.

**Gowtham's correction, confirmed by direct re-check, not assumed**: rule 1
forbids training on ASTRAL's *answers* (its precursor sets or its
principles), not a target formula merely appearing as a prompt. The
schemas of the contaminated files settle this directly —
`data/rl_run3/*.jsonl` and `data/rl/*.jsonl` have no `completion`/
`response` field at all (prompt-only; the model generates on-policy, there
is nothing fixed to memorize). Only `data/sft_v3` stores a real answer.

**The test was split, per instruction**:
- `tests/test_no_astral_answer_leakage.py` — the strict gate. **PASSES.**
  No `core/`/`data_curation/` module references an ASTRAL path, and zero
  training records anywhere have a stored answer equal to ASTRAL's
  rule-picked ("predicted") set for their target.
- `scripts/astral_overlap_report.py` → `results/phase16/astral_overlap.json`
  — informational, not gated. Also asserts zero overlap for any dataset
  Phase 16 itself creates (vacuously true today; nothing created yet).
- The old combined test moved to `research/test_no_astral_circularity_superseded.py`
  (never deleted, per project convention), with a header explaining the split.

**Full overlap classification** (dataset, target, role, match-against-ASTRAL):

| dataset | target | role | matches |
|---|---|---|---|
| `sft_v3_train`/`sft_v3_sft_train` | `KTiNbO5` | prompt+answer | **traditional** (exact) |
| `sft_v3_train`/`sft_v3_sft_train` | `LiMnPO4` | prompt+answer | neither |
| `sft_v3_train`/`sft_v3_sft_train` | `Li3V2(PO4)3` | prompt+answer | neither (close: `(NH4)2HPO4` vs traditional's `NH4H2PO4`) |
| `sft_v3_format_only_train` | `KTiNbO5`, `LiMnPO4` | prompt+answer | same as above (shared records) |
| `rl_run3_val` | `Na2Al2B2O7` | prompt-only | — |
| `rl_run3_probe` | `Na2Al2B2O7` | prompt-only | — |
| `rl_run3_train` | — | — | **0 overlap, confirmed explicitly, per instruction item 2** |
| `rl_train`/`rl_rl_train` | 9 targets (`Li3Fe2(PO4)3`, `NaSrBO3`, `KBaPO4`, `LiMgPO4`, `LiZnBO3`, `Li3Sc2(PO4)3`, `KNbWO6`, `Li2CuP2O7`, `KMgPO4`) | prompt-only | — |
| `rs_sft_train`/`rs_sft_val` | — | — | **0 overlap** |

**Zero records anywhere match ASTRAL's rule-picked set** — the only
non-trivial match is `KTiNbO5`'s exact match to the *conventional*
(traditional) literature recipe, a genuine, literal memorization case.
`LiMnPO4` and `Li3V2(PO4)3` do not exactly match either ASTRAL set.

**Union of all overlapping targets across every dataset: 13/35** —
`KBaPO4, KMgPO4, KNbWO6, KTiNbO5, Li2CuP2O7, Li3Fe2(PO4)3, Li3Sc2(PO4)3,
Li3V2(PO4)3, LiMgPO4, LiMnPO4, LiZnBO3, Na2Al2B2O7, NaSrBO3`. Clean
subset: **22/35**.

### Clean-subset sensitivity analysis (`research/phase16_astral_clean_subset.py` → `results/phase16/astral_clean_subset.json`)

**No headline conclusion changes.** Every number moves by the expected
small amount from removing ~37% of targets; none flips sign, none loses
significance in a way that changes what's claimed.

**1. Rule-picked hits, full-35 vs clean-22:**

| model | full-35 | clean-22 |
|---|---|---|
| base | 10/35 (28.6%) | 8/22 (36.4%) |
| full SFT | 0/35 | 0/22 |
| RS-SFT | 10/35 (28.6%) | 9/22 (40.9%) |
| GDPO-phase12-β0 | 14/35 (40.0%) | 12/22 (54.5%) |

**2. base → full SFT erasure**: full-35 base-only=10, sft-only=0,
McNemar p=0.0020 (finding 16/18's own number). **Clean-22: base-only=8,
sft-only=0, McNemar p=0.0078.** Still fully unidirectional, still
significant. **This directly answers the memorization concern**: the
erasure is not reliant on the 13 overlapping targets (including
`KTiNbO5`, the one literal-memorization case) — removing them, the same
base→SFT collapse pattern survives at essentially the same strength.

**3. RS-SFT → Phase 12 GDPO amplification**: full-35 gained=4
(`BaLiBO3`, `KLi(PO3)2`, `Li2TiSiO5`, `NaSrBO3`), one-sided p=0.0625
(finding 22's own number). **Clean-22: `NaSrBO3` is itself one of the 13
overlapping targets (prompt-only, in `data/rl` and `rl_run3`'s val/probe
— not an answer), so it drops out: gained=3 (`BaLiBO3`, `KLi(PO3)2`,
`Li2TiSiO5`), one-sided p=0.125.** The *direction* (RS-SFT 9/22 → GDPO
12/22, zero losses) is unchanged and, proportionally, slightly stronger
on the clean subset than on the full 35 — but note `NaSrBO3` was also
separately flagged (Task 1 finding 1.3) as "the one robust,
noise-exceeding GDPO-specific gain" (0/32→6/32). That claim rests on a
target that happens to be prompt-contaminated (not answer-contaminated)
in the general RL pool — worth keeping in mind when citing it, though
prompt-only overlap does not itself undermine a behavioral claim about
sampled generations. No pre-registered bar exists for the 22-target
subset (the `>12/35` amplification bar was set against the full 35), so
no new pass/fail claim is made here — only the side-by-side numbers.

**4. Carbonate/bare-oxide/other alkali-source shares** — virtually
unchanged full-35 vs clean-22 for every model:

| model | bare_oxide (full→clean) | carbonate (full→clean) |
|---|---|---|
| base | 0.091 → 0.097 | 0.729 → 0.734 |
| full SFT | 0.005 → 0.008 | 0.982 → 0.983 |
| RS-SFT | 0.258 → 0.282 | 0.459 → 0.440 |
| GDPO-phase12-β0 | 0.569 → 0.628 | 0.181 → 0.151 |

The base→SFT→RS-SFT→GDPO carbonate-to-bare-oxide shift (finding 24) is
not an artifact of the overlapping targets.

**Decision, per instruction**: `data/rl_run3` stays unchanged (new arms
stay comparable with Phase 12). For every arm, Phase 12 included, the
**clean-22-subset ASTRAL metrics are now primary**; full-35 metrics are
reported alongside as secondary. The E3/E3c target pool (from
`data/rs_sft`, 0/295 overlap) already excludes all ASTRAL targets per
§6.1, independent of this analysis.

### Environment guard (the mis-install correction)

Added to `train.py`: `_assert_main_environment()`, called at the top of
`main()`, raises immediately if `sys.prefix` isn't the main project
`.venv` — verified it fires correctly both ways (passes under the main
env, raises under `mira-vllm`'s python). Standing rule for all future
Phase 16 code: any script touching `mira-vllm` invokes
`~/envs/mira-vllm/bin/python` by absolute path, never bare `python` or
`uv pip` without an explicit environment target. End-of-pass check run
and confirmed: `uv sync --frozen` reports zero changes in the main
environment ("Checked 165 packages"), and all 6 test suites
(`test_validator`, `test_ranker`, `test_comparator`, `test_reward`,
`test_novelty_reward`, `test_no_astral_answer_leakage`) pass.

**Next**: draft `docs/phases/PHASE16_PREREG.md` with the clean-subset
metrics as primary, and stop for Gowtham's approval — per instruction.
