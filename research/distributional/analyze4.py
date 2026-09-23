"""
analyze4.py — split-half noise baseline for the Jaccard overlap numbers in
analyze.py's item 5 (Phase 15 Task 1 correction from regular Claude,
2026-09-23).

analyze.py reports full-32-vs-full-32 Jaccard overlap of precursor-set
solutions between model pairs (e.g. rs_sft vs gdpo: 0.59). That number
alone can't say whether RS-SFT and GDPO's solution sets are MORE
different than two independent samples from the SAME model would be —
i.e. whether the cross-model difference exceeds sampling noise. This
script answers that by resampling at matched sample size (16 vs 16,
matching each cross-model draw) 200 times per target and averaging:

  SELF(M):      split M's own 32 samples into two disjoint random halves
                of 16, Jaccard of the two halves' distinct precursor-set
                keys. This is the noise floor for "how different can two
                samples of size 16 from the same distribution look."
  CROSS(M1,M2): draw an independent random half of 16 from M1's 32 and a
                separate independent random half of 16 from M2's 32,
                Jaccard of the two. Comparable in sample size to SELF.

If CROSS(A,B) is indistinguishable from SELF(A) and SELF(B), the two
models' solution sets are indistinguishable from noise -- i.e. B did not
add or remove routes relative to A beyond what resampling A would show.

Same 4 input files, same target list (35), same `fam`/`key` conventions
as analyze.py — run from the same directory.

Usage: python3 analyze4.py
"""
import json
import random
from collections import Counter

M = {'base': 'astral_gen_n32_base.json', 'sft': 'astral_gen_n32_sft.json',
     'rs_sft': 'astral_gen_n32_rs_sft.json', 'gdpo': 'astral_gen_n32_gdpo_phase12_beta0.json'}
D = {k: {r['target']: r for r in json.load(open(v))['results']} for k, v in M.items()}
targets = list(D['base'].keys())

SEED = 20260923
N_SPLITS = 200
HALF = 16


def key(ps):
    return tuple(sorted(ps))


def precursor_keys(model, target):
    return [key(x['precursors']) for x in D[model][target]['samples'] if x.get('precursors')]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def self_jaccard(model, rng):
    """Mean Jaccard over N_SPLITS random 16/16 disjoint splits of the
    same model's own 32 samples, averaged over all 35 targets."""
    per_target = []
    for t in targets:
        keys = precursor_keys(model, t)
        n = len(keys)
        if n < 2:
            continue
        vals = []
        for _ in range(N_SPLITS):
            idx = list(range(n))
            rng.shuffle(idx)
            half = min(HALF, n // 2)
            a_idx, b_idx = idx[:half], idx[half:2 * half]
            a = [keys[i] for i in a_idx]
            b = [keys[i] for i in b_idx]
            vals.append(jaccard(a, b))
        per_target.append(sum(vals) / len(vals))
    return sum(per_target) / len(per_target)


def cross_jaccard(model_a, model_b, rng):
    """Mean Jaccard over N_SPLITS draws of an independent random 16-sample
    half from each of two DIFFERENT models' 32 samples, per target,
    averaged over all 35 targets. Sample-size-matched to self_jaccard."""
    per_target = []
    for t in targets:
        keys_a = precursor_keys(model_a, t)
        keys_b = precursor_keys(model_b, t)
        na, nb = len(keys_a), len(keys_b)
        if na < 1 or nb < 1:
            continue
        vals = []
        for _ in range(N_SPLITS):
            half_a = min(HALF, na)
            half_b = min(HALF, nb)
            a = rng.sample(keys_a, half_a)
            b = rng.sample(keys_b, half_b)
            vals.append(jaccard(a, b))
        per_target.append(sum(vals) / len(vals))
    return sum(per_target) / len(per_target)


def main():
    rng = random.Random(SEED)
    print(f"== split-half self-Jaccard noise floor (16 vs 16, {N_SPLITS} splits, mean over 35 targets) ==")
    self_vals = {}
    for m in M:
        v = self_jaccard(m, rng)
        self_vals[m] = v
        print(f"  self({m:7s}) = {v:.3f}")
    print()
    print(f"== cross-model Jaccard, matched sample size (16 vs 16, {N_SPLITS} draws, mean over 35 targets) ==")
    pairs = [('base', 'rs_sft'), ('rs_sft', 'gdpo'), ('base', 'gdpo'), ('base', 'sft')]
    cross_vals = {}
    for a, b in pairs:
        v = cross_jaccard(a, b, rng)
        cross_vals[(a, b)] = v
        print(f"  cross({a:7s},{b:7s}) = {v:.3f}   vs self({a})={self_vals[a]:.3f} self({b})={self_vals[b]:.3f}")
    print()
    print("== conclusion check: is cross(A,B) within noise of self(A) and self(B)? ==")
    for a, b in pairs:
        c = cross_vals[(a, b)]
        lo, hi = sorted([self_vals[a], self_vals[b]])
        within = lo - 0.05 <= c <= hi + 0.05  # descriptive band, not a formal test
        print(f"  {a} vs {b}: cross={c:.3f}, self range=[{lo:.3f},{hi:.3f}]  "
              f"{'~within noise' if within else 'DIFFERENT from noise floor'}")

    json.dump({
        "seed": SEED, "n_splits": N_SPLITS, "half": HALF,
        "self": self_vals, "cross": {f"{a}-{b}": v for (a, b), v in cross_vals.items()},
    }, open("splithalf_jaccard.json", "w"), indent=1)
    print("\n-> splithalf_jaccard.json")


if __name__ == "__main__":
    main()
