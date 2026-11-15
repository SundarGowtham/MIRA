#!/usr/bin/env python
"""
research/phase18_g_tensor_latent.py — Phase 18 Block 18G: what hidden variable
do the channels measure? PHASE18_PREREG.md "18G" (Anandkumar, Ge, Hsu, Kakade
& Telgarsky, JMLR 2014).

Data. Every existing raw completion (all models, ASTRAL + held-out; the four
structured-only ASTRAL files have no operations/amounts and cannot be
scored), parsed and scored per channel with the current validator. No new
sampling. Cached to results/phase18/channel_scores.jsonl.
Binarisation (decided before any score was looked at; validator.py defines
no per-channel pass threshold): pass = channel score >= 0.9, the project's
standard validator bar (RS-SFT survival bar, p-hat@0.9). [E] sensitivity:
pass = score == 1.0. Channels that pass on > 99.5% or < 0.5% of samples are
degenerate for a latent-class model and are dropped (rule fixed here);
samples where any retained channel is ungradeable are excluded from the
moment method (complete cases); EM handles them as missing.

Method of moments. Model: h ~ w, x_c | h ~ Bernoulli(mu_{c,h}) independent.
Multi-view estimator of the JMLR paper (Sec. 3.3): retained channels split
in fixed order into 3 contiguous views, each one-hot over its joint channel
configuration; views 1 and 2 symmetrised onto view 3; whitened M2,
symmetric whitened M3, robust tensor power method (L = 25, 50 iterations,
deflation). Identifiable only if every view has 2^{|view|} >= k; otherwise
that k is reported as not identifiable. (An earlier single-view variant
with iterative diagonal correction was tried in development and failed to
recover synthetic parameters; it was replaced before touching real data.)
EM latent-class analysis: same k, 20 random starts, missing entries skipped.
k in {2, 3, 4}.

Association. Samples whose canonical recipe equals one of its ASTRAL target's
two measured sets; lab success = that set's best purity > 0.5 (Block 0).
Each sample is assigned to its posterior-argmax component under the MoM
parameters (primary) and the EM parameters (check). "All channels pass"
component = the component with the highest mean conditional pass rate.
Odds ratio (in that component vs not) x (lab success), with a 0.5 Haldane
correction; 95% CI by cluster bootstrap over targets (10,000, seed 20261115).

Prediction [C] P-18G-1: the odds ratio's CI includes 1 (read strictly: at
every k in {2,3,4} under the MoM assignment).
[E]: residual within-component pairwise correlations (posterior-weighted).

Usage (tmux, CPU): scripts/run_phase18_block.sh 18G research/phase18_g_tensor_latent.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

from research.phase18_common import (  # noqa: E402
    SOURCES, SEED, N_BOOT, OUT_DIR, provenance, write_json, notify, astral_lab, target_sets,
)

ALL_10 = ("stoichiometry", "amount_accuracy", "charge_neutrality", "precursors_exist", "operation_order",
          "temperature_plausible", "thermodynamic_favorable", "target_stability", "chempot_atmosphere",
          "target_match")
SCORES = OUT_DIR / "channel_scores.jsonl"
PASS = 0.9
KS = (2, 3, 4)
V = None


def _init():
    global V
    from core.reward import load_validator
    V = load_validator(REPO_ROOT / "data/cache/mp_formula_set.pkl", REPO_ROOT / "data/cache/pd_index.json", REPO_ROOT)


def _score(args):
    model, ts, target, completion = args
    from core.reward import parse_completion
    from support_growth_by_checkpoint import precset_of
    from validator import SynthesisValidator
    try:
        route = parse_completion(completion, target)
        _r, bd = V.validate(route, target)
    except Exception:
        return None
    ch = {}
    for c in ALL_10:
        v = bd.get(c)
        ch[c] = None if (v is None or bd.get(f"{c}_gradeability") in SynthesisValidator.SENTINEL_TAGS) else float(v)
    return {"model": model, "target_set": ts, "target": target, "recipe": list(precset_of(route)), "ch": ch}


def build_scores():
    jobs = []
    for (model, ts), srcs in SOURCES.items():
        kind, path, _ = srcs[0]
        if kind != "raw":
            continue
        with path.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    jobs.append((model, ts, r["target"], r["completion"]))
    print(f"scoring {len(jobs)} raw completions", flush=True)
    out = []
    with Pool(30, initializer=_init) as pool:
        for i, r in enumerate(pool.imap(_score, jobs, chunksize=64)):
            if r is not None:
                out.append(r)
            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    with SCORES.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


def load_scores():
    if SCORES.exists():
        return [json.loads(l) for l in SCORES.open()]
    return build_scores()


# ---------------------------------------------------------------------------
# Method of moments with iterative diagonal correction + robust tensor power
# ---------------------------------------------------------------------------
def model_moments(w, mu):
    M2 = np.einsum("h,ah,bh->ab", w, mu, mu)
    M3 = np.einsum("h,ah,bh,ch->abc", w, mu, mu, mu)
    return M2, M3


def repeated_mask(d):
    i, j, k = np.indices((d, d, d))
    return (i == j) | (j == k) | (i == k)


def rtpm(T, k, rng, L=25, n_iter=50):
    T = T.copy()
    lams, vs = [], []
    for _ in range(k):
        best = None
        for _l in range(L):
            v = rng.normal(size=T.shape[0])
            v /= np.linalg.norm(v)
            for _t in range(n_iter):
                v2 = np.einsum("abc,b,c->a", T, v, v)
                nv = np.linalg.norm(v2)
                if nv < 1e-12:
                    break
                v = v2 / nv
            lam = float(np.einsum("abc,a,b,c->", T, v, v, v))
            if best is None or lam > best[0]:
                best = (lam, v)
        lam, v = best
        for _t in range(n_iter):
            v2 = np.einsum("abc,b,c->a", T, v, v)
            v = v2 / max(np.linalg.norm(v2), 1e-12)
        lam = float(np.einsum("abc,a,b,c->", T, v, v, v))
        lams.append(lam)
        vs.append(v)
        T = T - lam * np.einsum("a,b,c->abc", v, v, v)
    return np.array(lams), np.array(vs).T


def _onehot_view(X, cols):
    code = np.zeros(len(X), dtype=int)
    for j, c in enumerate(cols):
        code += (X[:, c].astype(int) << j)
    D = 2 ** len(cols)
    V = np.zeros((len(X), D))
    V[np.arange(len(X)), code] = 1.0
    return V


class NotIdentifiable(Exception):
    pass


def _pinv_k(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    if len(S) < k or S[k - 1] <= 1e-8 * max(S[0], 1e-300):
        raise NotIdentifiable(f"cross-moment rank < k (singular values {np.round(S[:k+1], 8).tolist()})")
    return Vt[:k].T @ np.diag(1 / S[:k]) @ U[:, :k].T


def mom_fit(X, k, rng, n_starts=None, n_outer=None):
    try:
        return _mom_fit(X, k, rng)
    except (NotIdentifiable, np.linalg.LinAlgError) as e:
        return {"not_identifiable": repr(e)[:300]}


def _mom_fit(X, k, rng):
    """Multi-view method of moments (Anandkumar et al. JMLR 2014, Sec. 3.3 +
    Sec. 4 robust tensor power method). Channels are split, in fixed order,
    into 3 contiguous views; each view is one-hot over its joint channel
    configuration (dim 2^{|view|}). Symmetrise views 1, 2 onto view 3,
    whiten M2 = E[x1~ x2~^T], decompose M3 = E[x1~ x2~ x3], recover view-3
    conditional means and weights, then views 1/2 via E[x_v x3^T] =
    mu_v diag(w) mu_3^T. Requires every view dim >= k (else returns None)."""
    N, d = X.shape
    groups = np.array_split(np.arange(d), 3)
    if any(2 ** len(g) < k for g in groups) or any(len(g) == 0 for g in groups):
        return None
    X1, X2, X3 = (_onehot_view(X, g) for g in groups)
    E = lambda A, B: A.T @ B / N  # noqa: E731
    X1t = X1 @ (E(X3, X2) @ _pinv_k(E(X1, X2), k)).T
    X2t = X2 @ (E(X3, X1) @ _pinv_k(E(X2, X1), k)).T
    M2 = E(X1t, X2t)
    M2 = (M2 + M2.T) / 2
    evals, evecs = np.linalg.eigh(M2)
    idx = np.argsort(evals)[::-1][:k]
    if not np.all(np.isfinite(M2)) or np.any(evals[idx] <= 1e-10):
        raise NotIdentifiable(f"whitening matrix not positive definite at rank k (eigenvalues {evals[idx].tolist()})")
    W = evecs[:, idx] / np.sqrt(evals[idx])
    a, b, c = X1t @ W, X2t @ W, X3 @ W
    T = np.einsum("ni,nj,nk->ijk", a, b, c) / N
    T = (T + T.transpose(0, 2, 1) + T.transpose(1, 0, 2) + T.transpose(1, 2, 0) + T.transpose(2, 0, 1)
         + T.transpose(2, 1, 0)) / 6
    lams, Vv = rtpm(T, k, rng)
    sg = np.where(lams < 0, -1.0, 1.0)
    lams, Vv = np.maximum(np.abs(lams), 1e-8), Vv * sg
    mu3 = np.linalg.pinv(W.T) @ (Vv * lams)
    w = 1 / lams ** 2
    w = w / w.sum()
    mus = []
    for Xv in (X1, X2):
        mus.append(E(Xv, X3) @ np.linalg.pinv(np.diag(w) @ mu3.T))
    mus.append(mu3)
    # per-channel conditional pass rates from each view's configuration distribution
    mu = np.zeros((d, k))
    for g, mv in zip(groups, mus):
        mv = np.clip(mv, 0, None)
        mv = mv / np.maximum(mv.sum(0, keepdims=True), 1e-12)
        D = 2 ** len(g)
        for j, ch in enumerate(g):
            bit = np.array([(code >> j) & 1 for code in range(D)])
            mu[ch] = bit @ mv
    mu = np.clip(mu, 0.0, 1.0)
    order = np.argsort(-mu.mean(0))
    return {"w": w[order], "mu": mu[:, order], "views": [list(map(int, g)) for g in groups],
            "tensor_eigenvalues": lams[order].tolist()}


def em_fit(X, M, k, rng, n_starts=20, n_iter=500):
    """X binary with missing mask M (True = observed)."""
    N, d = X.shape
    best = None
    for _s in range(n_starts):
        w = rng.dirichlet(np.ones(k))
        mu = rng.uniform(0.2, 0.95, (d, k))
        prev = -np.inf
        for _it in range(n_iter):
            lp = np.log(w)[None, :] + (M[:, :, None] * (X[:, :, None] * np.log(mu + 1e-12)[None]
                                                           + (1 - X[:, :, None]) * np.log(1 - mu + 1e-12)[None])).sum(1)
            mx = lp.max(1, keepdims=True)
            ll = float((mx[:, 0] + np.log(np.exp(lp - mx).sum(1))).sum())
            R = np.exp(lp - mx)
            R /= R.sum(1, keepdims=True)
            w = R.mean(0)
            num = (M * X).T @ R
            den = M.T.astype(float) @ R
            mu = np.clip(num / np.maximum(den, 1e-12), 1e-6, 1 - 1e-6)
            if ll - prev < 1e-6:
                break
            prev = ll
        if best is None or ll > best[0]:
            best = (ll, w, mu)
    ll, w, mu = best
    order = np.argsort(-mu.mean(0))
    n_par = (k - 1) + k * d
    return {"w": w[order], "mu": mu[:, order], "loglik": ll, "bic": -2 * ll + n_par * np.log(N)}


def posterior(X, M, w, mu):
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    lp = np.log(np.maximum(w, 1e-12))[None, :] + (M[:, :, None] * (X[:, :, None] * np.log(mu)[None]
                                                                    + (1 - X[:, :, None]) * np.log(1 - mu)[None])).sum(1)
    lp -= lp.max(1, keepdims=True)
    R = np.exp(lp)
    return R / R.sum(1, keepdims=True)


def odds_ratio(in_c, succ, targets, rng):
    def orr(ic, sc):
        a = np.sum(ic & sc) + 0.5
        b = np.sum(ic & ~sc) + 0.5
        c = np.sum(~ic & sc) + 0.5
        dd = np.sum(~ic & ~sc) + 0.5
        return float(a * dd / (b * c))
    point = orr(in_c, succ)
    uts = sorted(set(targets))
    by_t = defaultdict(list)
    for i, t in enumerate(targets):
        by_t[t].append(i)
    bs = []
    for _ in range(N_BOOT):
        draw = rng.integers(0, len(uts), len(uts))
        idx = np.concatenate([by_t[uts[j]] for j in draw])
        bs.append(orr(in_c[idx], succ[idx]))
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    degenerate = bool(in_c.all() or (~in_c).all())
    return {"or": point, "ci95": ci, "ci_includes_1": ci[0] <= 1 <= ci[1], "degenerate_all_one_side": degenerate,
            "n_samples": int(len(in_c)), "n_in_component": int(in_c.sum()), "n_success": int(succ.sum()),
            "n_targets": len(uts),
            "success_rate_in": float(succ[in_c].mean()) if in_c.any() else None,
            "success_rate_out": float(succ[~in_c].mean()) if (~in_c).any() else None}


def residual_corr(X, R, h):
    wts = R[:, h]
    if wts.sum() < 2:
        return None
    m = (wts[:, None] * X).sum(0) / wts.sum()
    Xc = X - m
    C = (wts[:, None] * Xc).T @ Xc / wts.sum()
    sd = np.sqrt(np.diag(C))
    with np.errstate(divide="ignore", invalid="ignore"):
        Cr = C / np.outer(sd, sd)
    off = Cr[~np.eye(len(sd), dtype=bool)]
    off = off[np.isfinite(off)]
    return {"mean_abs_offdiag_corr": float(np.abs(off).mean()) if len(off) else None,
            "max_abs_offdiag_corr": float(np.abs(off).max()) if len(off) else None}


def analyse(rows, thr_fn, label, lab, astral_set, rng):
    vals = np.array([[np.nan if r["ch"][c] is None else r["ch"][c] for c in ALL_10] for r in rows])
    obs = ~np.isnan(vals)
    Xall = np.where(obs, thr_fn(np.nan_to_num(vals)), 0).astype(float)
    rates = {c: float(Xall[obs[:, i], i].mean()) if obs[:, i].any() else None for i, c in enumerate(ALL_10)}
    keep = [i for i, c in enumerate(ALL_10) if rates[c] is not None and 0.005 < rates[c] < 0.995]
    chans = [ALL_10[i] for i in keep]
    X, Mobs = Xall[:, keep], obs[:, keep]
    complete = Mobs.all(1)
    Xc = X[complete]
    print(f"[{label}] retained channels {chans}; complete cases {complete.sum()}/{len(rows)}", flush=True)
    # lab-outcome samples
    lab_idx, succ, tgts = [], [], []
    for i, r in enumerate(rows):
        t = r["target"]
        if t not in astral_set:
            continue
        rec = tuple(r["recipe"])
        L = lab[t]
        for s, pur in ((tuple(L["traditional"]), L["trad_best_purity"]), (tuple(L["predicted"]), L["pred_best_purity"])):
            if rec == s and pur is not None:
                lab_idx.append(i)
                succ.append(pur > 0.5)
                tgts.append(t)
                break
    lab_idx = np.array(lab_idx, dtype=int)
    succ = np.array(succ, dtype=bool)
    out = {"label": label, "pass_rates_all_10": rates, "retained_channels": chans,
           "n_samples": len(rows), "n_complete_cases": int(complete.sum()),
           "n_lab_outcome_samples": int(len(lab_idx)), "n_lab_outcome_targets": len(set(tgts)), "by_k": {}}
    for k in KS:
        res = {}
        mom = mom_fit(Xc, k, rng)
        em = em_fit(X, Mobs, k, rng)
        for name, fit in (("MoM", mom), ("EM", em)):
            if fit is None or "not_identifiable" in fit:
                res[name] = {"not_identifiable": True, "reason": (fit or {}).get("not_identifiable",
                             "a view has fewer than k configurations")}
                print(f"  [{label}] k={k} {name}: not identifiable ({res[name]['reason'][:120]})", flush=True)
                continue
            R = posterior(X, Mobs, fit["w"], fit["mu"])
            assign = R.argmax(1)
            ap = int(np.argmax(fit["mu"].mean(0)))
            in_c = assign[lab_idx] == ap
            entry = {"weights": fit["w"].tolist(),
                     "conditional_pass_rates": {c: fit["mu"][i].tolist() for i, c in enumerate(chans)},
                     "all_pass_component": ap,
                     "all_pass_component_min_rate": float(fit["mu"][:, ap].min()),
                     "assigned_share": np.bincount(assign, minlength=k).tolist(),
                     "odds_ratio_all_pass_vs_lab_success": odds_ratio(in_c, succ, tgts, rng) if len(lab_idx) else None,
                     "E_residual_within_component_corr": {h: residual_corr(Xc, posterior(Xc, np.ones_like(Xc, bool),
                                                                                       fit["w"], fit["mu"]), h)
                                                          for h in range(k)}}
            if name == "MoM":
                entry["views"] = fit["views"]
                entry["tensor_eigenvalues"] = fit["tensor_eigenvalues"]
            else:
                entry["loglik"] = fit["loglik"]
                entry["bic"] = fit["bic"]
            res[name] = entry
            o = entry["odds_ratio_all_pass_vs_lab_success"]
            print(f"  [{label}] k={k} {name}: w={np.round(fit['w'],3).tolist()} OR={o['or']:.3f} CI={np.round(o['ci95'],3).tolist()} "
                  f"in={o['n_in_component']}/{o['n_samples']}", flush=True)
        out["by_k"][str(k)] = res
    return out


def main():
    prov = provenance()
    rng = np.random.default_rng(SEED)
    rows = load_scores()
    lab = astral_lab()
    astral = set(target_sets()["astral_35"])
    primary = analyse(rows, lambda v: (v >= PASS).astype(float), "pass>=0.9", lab, astral, rng)
    sens = analyse(rows, lambda v: (v >= 1.0 - 1e-12).astype(float), "[E] pass==1.0", lab, astral, rng)
    mom_rows = {k: primary["by_k"][k]["MoM"].get("odds_ratio_all_pass_vs_lab_success") for k in primary["by_k"]}
    evaluable = {k: v for k, v in mom_rows.items() if v is not None and not v["degenerate_all_one_side"]}
    passes = bool(evaluable) and all(v["ci_includes_1"] for v in evaluable.values())
    doc = {**prov, "block": "18G", "pass_threshold": PASS,
           "n_scored_samples": len(rows),
           "models_scored": sorted({r["model"] for r in rows}),
           "excluded_sources": "structured-only ASTRAL files (corpus_sft, gdpo_v3_ckpt300, phase12_ckpt300, "
                               "e1_ckpt300): no operations/amounts, cannot be scored",
           "P_18G_1": {"statement": "Odds ratio between the all-channels-pass component and ASTRAL lab success has a "
                                    "CI including 1 (every k in {2,3,4}, MoM assignment).",
                       "by_k_MoM": mom_rows,
                       "by_k_EM": {k: primary["by_k"][k]["EM"].get("odds_ratio_all_pass_vs_lab_success")
                                   for k in primary["by_k"]},
                       "passes": passes,
                       "evaluation_rule_note": "k values where the MoM fit is not identifiable, or where every "
                                               "lab-outcome sample falls in one component (OR undefined up to the "
                                               "Haldane correction), are reported but cannot count for or against."},
           "primary": primary, "E_sensitivity_pass_eq_1": sens}
    p = write_json("tensor_latent.json", doc)
    print("P-18G-1 passes:", passes)
    print(f"-> {p}")
    notify(f"MIRA 18G done: P-18G-1 passes={passes}")


if __name__ == "__main__":
    main()
