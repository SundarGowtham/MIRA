# Distributional analysis of ASTRAL n=32 generations (2026-09-20)

Inputs (from repo `results/`): astral_gen_n32_{base,sft,rs_sft,gdpo_phase12_beta0}.json
Run from a directory containing those four files: `python3 analyze.py && python3 analyze2.py && python3 analyze3.py`

All findings here are EXPLORATORY [E] — found by looking, not pre-registered.

Key results:
- Base and RS-SFT both hit 10/35 predicted sets but share only 7 targets.
- 16 targets ever hit across all models; GDPO covers 14.
- Most hits are 1–3 of 32 samples; P(0/32 | p=1/32) = 0.36 → per-target churn is mostly noise.
  Robust GDPO gain: NaSrBO3 (0 → 6/32).
- Precursor-type shift: bare alkali oxide 8.6% → 0.5% → 25.1% → 56.3% (base, SFT, RS-SFT, GDPO);
  carbonate 74.7% → 98.5% → 48.0% → 19.1%. ASTRAL-predicted share stays 2–4%.
- Base-model validator pass@0.9: bare-oxide-only 98.9% vs carbonate-only 78.2%.
- Ammonium phosphate on phosphate targets: 23.8% → 82.6% → 4.9% → 2.1%.
- Hit definition: full SFT is 0/35 exact; 1/35 if superset matches count (LiZnBO3).
