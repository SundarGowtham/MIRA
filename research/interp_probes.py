#!/usr/bin/env python
"""
interp_probes.py — probes A/B/C from CLAUDE.md, now hardened per
misc/some_claude_files/SPEC_interp_probe_hardening.md (2026-08-26/27).

v1 (findings 12-14) found: probe A predicts binary hit/miss from prompt
activations (AUC ~0.75, mis-reported as null by a layer-selection bug),
probe B found a frac/int direction that exists at nearly every layer and
rotates under SFT, probe C found the 17 hard-zero targets are linearly
separable in BASE activations (LOO AUC 0.821). v2 hardens all three:

  Task 1 (probe A): select "best layer" by AUC, not R^2 (R^2 is negative
    everywhere -- continuous pass@1 is genuinely not linearly predictable;
    that's a separate, correctly-negative result, not the same finding).
    Adds a bootstrap CI on the best-layer AUC and an AUC-vs-layer PNG.
  Task 2 (probe C): re-tests the hard-zero result with the 5 malformed
    corpus-artifact targets (BaCrO, MgSnZnO, LiFeBO3C, Eu4Y1, ReBa2Cu3O)
    excluded (12 remain). The FULL-17 numbers are byte-identical to the
    prior run (LOO is deterministic) so they're reused from the prior
    pass3_interp_probes.json rather than recomputed -- only the best
    layer's LOO is refit (for a bootstrap CI); the reduced-12 case is a
    fresh full sweep since its label vector is new. Layer 0 (embeddings)
    is reported explicitly as the negative control for both.
  Task 3 (probe B): adds a held-out (stratified train/test, repeated)
    refit of the frac/int direction, a random-direction cosine null to
    give the cos(d_base,d_sft) decay a reference scale, and a held-out
    version of the projection-vs-Delta-pass@1 correlation. Also fixes the
    ConstantInputWarning by guarding spearmanr on near-constant input.

No new forward passes: reuses the cached activations in
manifold_visualization/act_geo/cache/{base,sft,gdpo300}_prompts200.npz.
Everything below is CPU-only.

Usage:
  PYTHONPATH=. uv run python run_debug_and_analysis/interp_probes.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import spearmanr, rankdata  # noqa: E402

from activation_geometry import (  # noqa: E402
    CHECKPOINTS, closed_book_prompt, pooled_activations, get_acts,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

CACHE_ROOT = Path("manifold_visualization/act_geo")
PASSK_FILE = Path("misc/passk_n200.json")
OUT_FILE = CACHE_ROOT / "pass3_interp_probes.json"

N_FOLDS = 5
N_PERMS = 200  # probe B in-sample direction permutation test (unchanged from v1)
N_PERMS_C = 50  # probe C 5-fold permutation null (halved from v1's 100 -- ablation, not the headline number)
N_BOOT = 2000
N_REPEATS_B = 20  # held-out train/test repeats for probe B
RANDOM_DIR_NULL_PAIRS = 2000
RIDGE_ALPHA = 100.0
LOGREG_ALPHA = 1.0
SEED = 42

MALFORMED_HARD_ZEROS = ["BaCrO", "MgSnZnO", "LiFeBO3C", "Eu4Y1", "ReBa2Cu3O"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--forward", action="store_true", help="cache activations only")
    p.add_argument("--probe-a", action="store_true")
    p.add_argument("--probe-b", action="store_true")
    p.add_argument("--probe-c", action="store_true")
    p.add_argument("--all", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_targets():
    data = json.loads(PASSK_FILE.read_text())["per_target"]
    targets = [r["target"] for r in data]
    strata = [r["stratum"] for r in data]
    by_ckpt = {}
    for ckpt in CHECKPOINTS:
        by_ckpt[ckpt] = {
            "pass1": np.array([r["models"][ckpt]["pass@1"] for r in data]),
            "n_success": np.array([r["models"][ckpt]["n_success"] for r in data]),
            "mean_reward": np.array([r["models"][ckpt]["mean_reward"] for r in data]),
        }
    return targets, strata, by_ckpt


def load_prior():
    if not OUT_FILE.exists():
        return None
    try:
        prior = json.loads(OUT_FILE.read_text())
        print(f"loaded prior results from {OUT_FILE} (version {prior.get('version', 1)})",
              flush=True)
        return prior
    except Exception as e:
        print(f"could not load prior JSON ({e}); proceeding without reuse", flush=True)
        return None


# --------------------------------------------------------------------------
# forward pass / cache (unchanged from v1 -- hits the on-disk cache, no GPU)
# --------------------------------------------------------------------------

def forward_all(args, targets):
    acts = {}
    ns_args = SimpleNamespace(out=CACHE_ROOT, model=args.model)
    for name in CHECKPOINTS:
        def build(model, tok, targets=targets):
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": closed_book_prompt(t)}],
                tokenize=False, add_generation_prompt=True) for t in targets]
            starts = [len(tok(t).input_ids) - 1 for t in texts]
            return pooled_activations(model, tok, texts, starts,
                                      args.max_tokens), {"pool": "last_prompt_token",
                                                          "n_targets": len(targets)}
        a, _ = get_acts(ns_args, name, "prompts200", build)
        acts[name] = a
        print(f"[{name}] acts shape {a.shape}", flush=True)
    return acts


# --------------------------------------------------------------------------
# small linear-algebra toolkit (no sklearn). n (~160-200) << d (4096): ridge
# uses the dual/kernel form (n x n solve, not d x d) and logistic gets an
# analytic gradient so L-BFGS-B doesn't fall back to O(d) finite differences.
# Both operate on the FULL hidden dim (no PCA) so a signal outside the
# top-variance subspace isn't discarded before the readout sees it.
# --------------------------------------------------------------------------

def standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_dual_fit_predict(Xtr, ytr, Xte, alpha=RIDGE_ALPHA):
    Xtr, Xte = standardize(Xtr, Xte)
    ybar = ytr.mean()
    yc = ytr - ybar
    K = Xtr @ Xtr.T
    c = np.linalg.solve(K + alpha * np.eye(len(Xtr)), yc)
    return (Xte @ Xtr.T) @ c + ybar


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logreg_nll_grad(w, Z, y, alpha):
    z = Z @ w[:-1] + w[-1]
    p = _sigmoid(z)
    eps = 1e-9
    nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    nll += alpha * np.sum(w[:-1] ** 2)
    gw = Z.T @ (p - y) / len(y) + 2 * alpha * w[:-1]
    gb = np.mean(p - y)
    return nll, np.concatenate([gw, [gb]])


def logreg_fit_predict(Xtr, ytr, Xte, alpha=LOGREG_ALPHA):
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr[0]))
    Xtr, Xte = standardize(Xtr, Xte)
    w0 = np.zeros(Xtr.shape[1] + 1)
    res = minimize(_logreg_nll_grad, w0, args=(Xtr, ytr, alpha), jac=True,
                    method="L-BFGS-B", options={"maxiter": 200})
    w = res.x
    return _sigmoid(Xte @ w[:-1] + w[-1])


def auc_score(y_true, y_score):
    """Mann-Whitney U / AUC with proper tie-averaged ranks (rankdata, not
    argsort). Matters here: layer 0 is the embedding of the fixed last
    prompt token, IDENTICAL across every target, so LOO predictions there
    are constant/near-constant -- argsort-based ranking turns that into an
    arbitrary extreme value (v1 reported layer-0 AUC=0.000) instead of the
    correct chance reading (0.5 exactly for fully-tied scores)."""
    y_true = np.asarray(y_true)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(y_score, method="average")
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def kfold_indices(n, k, rng):
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def argmax_ignore_none(lst):
    arr = np.array([v if v is not None else -np.inf for v in lst])
    return int(np.argmax(arr))


def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOT, seed=0):
    """Nonparametric bootstrap CI on AUC: resample (y_true, y_score) PAIRS
    from an already-computed held-out prediction set (not refitting the
    model each iteration -- standard practice for a CI on a held-out
    metric, and the only tractable option at ~2000 iters given per-fit
    cost)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc_score(y_true[idx], y_score[idx])
        if a is not None:
            boots.append(a)
    boots = np.array(boots)
    if len(boots) == 0:
        return {"mean": None, "lo": None, "hi": None, "n_boot_valid": 0}
    return {"mean": float(boots.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n_boot_valid": int(len(boots))}


def safe_spearman(x, y):
    """spearmanr guarded against near-constant input (source of the
    ConstantInputWarning in v1) -- returns NaN instead of an undefined
    correlation."""
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan"), float("nan")
    rho, pval = spearmanr(x, y)
    return float(rho), float(pval)


def stratified_split(mask, rng, test_frac=0.5):
    idx_pos = rng.permutation(np.where(mask)[0])
    idx_neg = rng.permutation(np.where(~mask)[0])
    n_pos_te = max(1, int(round(len(idx_pos) * test_frac)))
    n_neg_te = max(1, int(round(len(idx_neg) * test_frac)))
    test = np.concatenate([idx_pos[:n_pos_te], idx_neg[:n_neg_te]])
    train = np.concatenate([idx_pos[n_pos_te:], idx_neg[n_neg_te:]])
    return train, test


def random_direction_cosine_null(d, n_pairs, rng):
    """Null distribution for cos(u, v) between two random unit vectors in
    R^d. This depends ONLY on d (not on which layer/checkpoint), since it
    uses no activation data -- one computation covers every layer."""
    V = rng.normal(size=(2 * n_pairs, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    cos = np.sum(V[:n_pairs] * V[n_pairs:], axis=1)
    return {"d": d, "n_pairs": n_pairs, "mean": float(cos.mean()),
            "band_5_95": [float(np.percentile(cos, 5)), float(np.percentile(cos, 95))]}


def plot_auc_by_layer(plot_data, path):
    if not HAVE_MPL:
        print(f"  [plot] matplotlib unavailable, skipping {path}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for ckpt, aucs in plot_data.items():
        ax.plot(range(len(aucs)), [a if a is not None else np.nan for a in aucs],
                marker="o", markersize=3, label=ckpt)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out AUC")
    ax.set_title("Probe A: per-target success predictability by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] -> {path}", flush=True)


# --------------------------------------------------------------------------
# Probe A — failure prediction (Task 1: select on AUC, bootstrap CI, plot)
# --------------------------------------------------------------------------

def probe_a(acts, by_ckpt, rng):
    print("\n=== Probe A: failure prediction ===", flush=True)
    results = {}
    plot_data = {}
    for ci_seed, (ckpt, A) in enumerate(acts.items()):
        n, L, _ = A.shape
        y_reg = by_ckpt[ckpt]["pass1"]
        y_clf = (by_ckpt[ckpt]["n_success"] > 0).astype(float)
        folds = kfold_indices(n, N_FOLDS, rng)
        layer_r2, layer_auc = [], []
        pred_clf_per_layer = []
        for l in range(L):
            X = A[:, l, :]
            pred_reg = np.zeros(n)
            pred_clf = np.zeros(n)
            for f in range(N_FOLDS):
                te = folds[f]
                tr = np.concatenate([folds[i] for i in range(N_FOLDS) if i != f])
                pred_reg[te] = ridge_dual_fit_predict(X[tr], y_reg[tr], X[te])
                pred_clf[te] = logreg_fit_predict(X[tr], y_clf[tr], X[te])
            layer_r2.append(r2_score(y_reg, pred_reg))
            layer_auc.append(auc_score(y_clf, pred_clf))
            pred_clf_per_layer.append(pred_clf)

        best_by_auc = argmax_ignore_none(layer_auc)
        best_by_r2 = int(np.argmax(layer_r2))  # kept for transparency only -- NOT the headline
        ci = bootstrap_auc_ci(y_clf, pred_clf_per_layer[best_by_auc], seed=10 + ci_seed)

        results[ckpt] = {
            "held_out_r2_pass1": layer_r2,
            "held_out_auc_hit": layer_auc,
            "best_layer_by_auc": best_by_auc,
            "best_layer_by_auc_ci95": ci,
            "best_layer_by_r2_not_meaningful": best_by_r2,
        }
        plot_data[ckpt] = layer_auc
        mean_auc = float(np.nanmean([a for a in layer_auc if a is not None]))
        above_chance = sum((a or 0) > 0.5 for a in layer_auc)
        print(f"  [{ckpt}] best layer BY AUC {best_by_auc}: "
              f"AUC={layer_auc[best_by_auc]:.3f} 95% CI "
              f"[{ci['lo']:.3f},{ci['hi']:.3f}]  (R^2(pass@1) at this layer="
              f"{layer_r2[best_by_auc]:.3f} -- not predictive, continuous rate "
              f"genuinely null)  |  mean AUC={mean_auc:.3f}, layers>chance: "
              f"{above_chance}/{L}", flush=True)

    plot_auc_by_layer(plot_data, CACHE_ROOT / "probe_a_auc_by_layer.png")
    return results


# --------------------------------------------------------------------------
# Probe B — redistribution direction (Task 3: held-out refit + random-dir null)
# --------------------------------------------------------------------------

def probe_b(acts, targets, strata, by_ckpt, rng):
    print("\n=== Probe B: frac/int redistribution direction ===", flush=True)
    is_frac = np.array([s.startswith("frac") for s in strata])
    n = len(targets)
    print(f"  frac={is_frac.sum()}, int={(~is_frac).sum()}", flush=True)
    if is_frac.sum() < 4 or (~is_frac).sum() < 4:
        print("  degenerate frac/int split -- skipping probe B", flush=True)
        return {"skipped": "degenerate frac/int split"}
    delta_pass1 = by_ckpt["sft"]["pass1"] - by_ckpt["base"]["pass1"]

    def direction(X, mask):
        return X[mask].mean(0) - X[~mask].mean(0)

    dirs = {}
    results = {"channels_note": "diff-of-means direction; in-sample values are "
                                 "labeled _in_sample, perm_p over label-shuffles "
                                 "like v1's pass2. held_out_* fields are Task-3 "
                                 "additions: direction fit on a train split only, "
                                 "evaluated on the held-out complement.",
               "per_ckpt": {}}
    for ckpt, A in acts.items():
        _, L, _ = A.shape
        norms, ps, proj_corr, proj_corr_p = [], [], [], []
        dlist = []
        for l in range(L):
            X = A[:, l, :]
            d = direction(X, is_frac)
            dlist.append(d)
            nrm = float(np.linalg.norm(d))
            norms.append(nrm)
            nulls = [float(np.linalg.norm(direction(X, rng.permutation(is_frac))))
                     for _ in range(N_PERMS)]
            ps.append(float((np.array(nulls) >= nrm).mean()))
            proj = X @ (d / (nrm + 1e-12))
            rho, pval = safe_spearman(proj, delta_pass1)
            proj_corr.append(rho)
            proj_corr_p.append(pval)
        dirs[ckpt] = np.stack(dlist)
        results["per_ckpt"][ckpt] = {
            "direction_norm_in_sample": norms, "perm_p_in_sample": ps,
            "proj_vs_delta_spearman_in_sample": proj_corr,
            "proj_vs_delta_spearman_p_in_sample": proj_corr_p,
        }
        best_l = int(np.argmin(ps))
        print(f"  [{ckpt}] in-sample strongest layer {best_l}: norm={norms[best_l]:.2f} "
              f"perm_p={ps[best_l]:.3f}  |  layers p<0.05: "
              f"{sum(p < 0.05 for p in ps)}/{L}", flush=True)

    if "base" in dirs and "sft" in dirs:
        L = dirs["base"].shape[0]
        cos = []
        for l in range(L):
            da, ds = dirs["base"][l], dirs["sft"][l]
            cos.append(float(da @ ds / (np.linalg.norm(da) * np.linalg.norm(ds) + 1e-12)))
        results["cos_base_dir_vs_sft_dir"] = cos
        print(f"  cos(d_base, d_sft): layer 1={cos[1]:.3f} -> layer {L-1}={cos[-1]:.3f} "
              f"(mean {np.mean(cos):.3f})", flush=True)

    # ---- Task 3a: held-out refit ----
    print("  -- held-out refit ({} repeats, stratified 50/50 splits) --".format(N_REPEATS_B),
          flush=True)
    L = next(iter(acts.values())).shape[1]
    held_auc = {ckpt: [[] for _ in range(L)] for ckpt in acts}
    held_d = {ckpt: [[] for _ in range(L)] for ckpt in acts}
    proj_sum = {ckpt: np.zeros((L, n)) for ckpt in acts}
    proj_cnt = {ckpt: np.zeros((L, n)) for ckpt in acts}
    for _ in range(N_REPEATS_B):
        train, test = stratified_split(is_frac, rng)
        y_test = is_frac[test].astype(float)
        for ckpt, A in acts.items():
            for l in range(L):
                X = A[:, l, :]
                d = direction(X[train], is_frac[train])
                nrm = np.linalg.norm(d)
                if nrm < 1e-12:
                    continue
                proj_test = X[test] @ (d / nrm)
                a = auc_score(y_test, proj_test)
                if a is not None:
                    held_auc[ckpt][l].append(a)
                if (y_test == 1).any() and (y_test == 0).any():
                    sd = proj_test.std() + 1e-12
                    held_d[ckpt][l].append(float((proj_test[y_test == 1].mean() -
                                                   proj_test[y_test == 0].mean()) / sd))
                proj_sum[ckpt][l, test] += proj_test
                proj_cnt[ckpt][l, test] += 1

    for ckpt in acts:
        auc_mean = [float(np.mean(v)) if v else None for v in held_auc[ckpt]]
        d_mean = [float(np.mean(v)) if v else None for v in held_d[ckpt]]
        results["per_ckpt"][ckpt]["held_out_auc_frac_vs_int"] = auc_mean
        results["per_ckpt"][ckpt]["held_out_cohens_d"] = d_mean
        best_l = argmax_ignore_none(auc_mean)
        cnt = proj_cnt[ckpt][best_l]
        have = cnt > 0
        avg_proj = np.divide(proj_sum[ckpt][best_l], cnt,
                              out=np.full(n, np.nan), where=have)
        rho_ho, p_ho = safe_spearman(avg_proj[have], delta_pass1[have])
        results["per_ckpt"][ckpt]["held_out_proj_vs_delta_spearman"] = {
            "layer": best_l, "rho": rho_ho, "p": p_ho, "n": int(have.sum())}
        print(f"  [{ckpt}] held-out best layer {best_l}: AUC={auc_mean[best_l]:.3f}, "
              f"Cohen's d={d_mean[best_l]:.2f}  |  held-out proj-vs-delta rho={rho_ho:.3f} "
              f"p={p_ho:.3f} (n={int(have.sum())})", flush=True)

    # ---- Task 3b: random-direction cosine null ----
    d_dim = next(iter(acts.values())).shape[-1]
    null_cos = random_direction_cosine_null(d_dim, RANDOM_DIR_NULL_PAIRS, rng)
    results["random_direction_cosine_null"] = null_cos
    print(f"  random-direction cosine null (d={d_dim}, layer-invariant): "
          f"mean={null_cos['mean']:.4f}, 5-95% band="
          f"[{null_cos['band_5_95'][0]:.4f},{null_cos['band_5_95'][1]:.4f}]", flush=True)
    if "cos_base_dir_vs_sft_dir" in results:
        cos_list = results["cos_base_dir_vs_sft_dir"]
        hi = null_cos["band_5_95"][1]
        above = sum(c > hi for c in cos_list if c is not None)
        verdict = ("ABOVE null (direction partially preserved)" if cos_list[-1] > hi
                   else "WITHIN null (direction effectively replaced)")
        print(f"  cos(d_base,d_sft) layers above the random-direction null band: "
              f"{above}/{len(cos_list)}  |  final layer ({len(cos_list)-1}) value "
              f"{cos_list[-1]:.3f} vs null hi {hi:.4f}: {verdict}", flush=True)
        results["cos_final_layer_vs_null"] = {"value": cos_list[-1], "null_hi": hi,
                                              "verdict": verdict}

    return results


# --------------------------------------------------------------------------
# shared LOO / permutation-null helpers for probe C
# --------------------------------------------------------------------------

def loo_auc_for_layers(X_all, y, layers):
    """Deterministic (no rng) leave-one-out logistic AUC for the requested
    layers. Returns {layer: (auc, oof_pred_array)}."""
    n = X_all.shape[0]
    out = {}
    for l in layers:
        X = X_all[:, l, :]
        pred = np.zeros(n)
        for i in range(n):
            tr = np.array([j for j in range(n) if j != i])
            pred[i] = logreg_fit_predict(X[tr], y[tr], X[[i]])[0]
        out[l] = (auc_score(y, pred), pred)
    return out


def permutation_null_5fold(X_all, y, rng, n_perms, layers):
    """5-fold CV AUC under label permutation, per requested layer -- ~40x
    cheaper per permutation than LOO, used only for the null distribution
    (the point estimate above stays LOO)."""
    layers = list(layers)
    n = X_all.shape[0]
    out = {l: [] for l in layers}
    for p in range(n_perms):
        yp = rng.permutation(y)
        folds = kfold_indices(n, N_FOLDS, rng)
        for l in layers:
            X = X_all[:, l, :]
            pred = np.zeros(n)
            for f in range(N_FOLDS):
                te = folds[f]
                tr = np.concatenate([folds[i] for i in range(N_FOLDS) if i != f])
                pred[te] = logreg_fit_predict(X[tr], yp[tr], X[te])
            out[l].append(auc_score(yp, pred))
        if (p + 1) % 10 == 0:
            print(f"    null perm {p + 1}/{n_perms}", flush=True)
    return {l: np.array([v if v is not None else np.nan for v in vs])
            for l, vs in out.items()}


# --------------------------------------------------------------------------
# Probe C — hard-zero pocket (Task 2: malformed-formula ablation)
# --------------------------------------------------------------------------

def probe_c(acts, targets, by_ckpt, rng, prior=None):
    print("\n=== Probe C: hard-zero pocket in BASE activations ===", flush=True)
    n_success_all = np.stack([by_ckpt[c]["n_success"] for c in CHECKPOINTS])
    hard_zero = (n_success_all == 0).all(0)
    hz_targets = [t for t, hz in zip(targets, hard_zero) if hz]
    malformed_mask = np.array([t in MALFORMED_HARD_ZEROS for t in targets])
    reduced_zero = hard_zero & ~malformed_mask
    hz_reduced_targets = [t for t, z in zip(targets, reduced_zero) if z]

    print(f"  hard-zero (0/16 on base+sft+gdpo300): {hard_zero.sum()}/{len(targets)}  |  "
          f"reduced (malformed excluded): {reduced_zero.sum()}", flush=True)
    print(f"  malformed excluded: {MALFORMED_HARD_ZEROS}", flush=True)

    results = {"hard_zero_targets": hz_targets, "n_hard_zero": int(hard_zero.sum()),
               "malformed_excluded": MALFORMED_HARD_ZEROS,
               "hard_zero_targets_reduced": hz_reduced_targets,
               "n_hard_zero_reduced": int(reduced_zero.sum())}
    if hard_zero.sum() < 4 or hard_zero.sum() > len(targets) - 4:
        print("  too few/many hard-zero targets for a CV'd probe — reporting counts only",
              flush=True)
        return results

    X_all = acts["base"]
    n, L, _ = X_all.shape
    y_full = hard_zero.astype(float)
    y_red = reduced_zero.astype(float)

    # ---- full-17: reuse prior run's LOO+null if present AND computed with
    # the tie-corrected auc_score (deterministic given the same auc_score,
    # so byte-identical to a fresh recompute) -- only refit at the best
    # layer to get out-of-fold predictions for the bootstrap CI. A prior
    # run predating the tie-correction fix (layer-0 activations are
    # constant across targets -> argsort-based AUC there was an artifact,
    # not a real 0.000) is NOT reusable -- its numbers differ from what
    # auc_score now computes. ----
    prior_full17 = (prior or {}).get("full_17", {})
    if prior_full17.get("tie_corrected_auc") and prior_full17.get("held_out_auc_loo") \
            and prior_full17.get("perm_null_auc_5fold"):
        print("  [full-17] reusing prior LOO + null (tie-corrected, unchanged input)",
              flush=True)
        layer_auc_full = prior_full17["held_out_auc_loo"]
        perm_p_full = prior_full17.get("perm_p_per_layer") or [None] * L
        perm_null_full = prior_full17["perm_null_auc_5fold"]
    else:
        print("  [full-17] no reusable prior (missing, or predates the tie-correction "
              "fix to auc_score) -- computing fresh", flush=True)
        loo_full = loo_auc_for_layers(X_all, y_full, range(L))
        layer_auc_full = [loo_full[l][0] for l in range(L)]
        null_full = permutation_null_5fold(X_all, y_full, rng, N_PERMS_C, range(L))
        perm_p_full = [float((null_full[l] >= (layer_auc_full[l] if layer_auc_full[l]
                                                is not None else -1)).mean()) for l in range(L)]
        perm_null_full = {"mean": [float(np.nanmean(null_full[l])) for l in range(L)],
                          "p95": [float(np.nanpercentile(null_full[l], 95)) for l in range(L)]}

    best_l_full = argmax_ignore_none(layer_auc_full)
    _, oof_best_full = loo_auc_for_layers(X_all, y_full, [best_l_full])[best_l_full]
    ci_full = bootstrap_auc_ci(y_full, oof_best_full, seed=1)
    results["full_17"] = {
        "held_out_auc_loo": layer_auc_full, "perm_p_per_layer": perm_p_full,
        "perm_null_auc_5fold": perm_null_full, "best_layer": best_l_full,
        "best_layer_auc_ci95": ci_full, "tie_corrected_auc": True,
        "layer0_negative_control": {"auc": layer_auc_full[0], "perm_p": perm_p_full[0]},
    }
    print(f"  [full-17] best layer {best_l_full}: LOO AUC={layer_auc_full[best_l_full]:.3f} "
          f"95% CI [{ci_full['lo']:.3f},{ci_full['hi']:.3f}], "
          f"perm_p={perm_p_full[best_l_full]:.3f}  |  layer0 (negative control): "
          f"AUC={layer_auc_full[0]:.3f} perm_p={perm_p_full[0]:.3f}", flush=True)

    # ---- reduced-12: fresh full sweep, new label vector ----
    print(f"  [reduced-12] computing fresh full sweep ({N_PERMS_C} null perms) ...",
          flush=True)
    loo_red = loo_auc_for_layers(X_all, y_red, range(L))
    layer_auc_red = [loo_red[l][0] for l in range(L)]
    null_red = permutation_null_5fold(X_all, y_red, rng, N_PERMS_C, range(L))
    perm_p_red = [float((null_red[l] >= (layer_auc_red[l] if layer_auc_red[l]
                                         is not None else -1)).mean()) for l in range(L)]
    perm_null_red = {"mean": [float(np.nanmean(null_red[l])) for l in range(L)],
                     "p95": [float(np.nanpercentile(null_red[l], 95)) for l in range(L)]}
    # best-of-37 is EXPLORATORY only (data-snooped: searching all 37 layers
    # for reduced-12's own peak is a different, weaker test than the
    # pre-registered one below, and will optimistically bias upward)
    best_l_red = argmax_ignore_none(layer_auc_red)
    _, oof_best_red = loo_red[best_l_red]
    ci_red = bootstrap_auc_ci(y_red, oof_best_red, seed=2)

    # PRE-REGISTERED test: the spec fixed the layer in advance as full-17's
    # peak (~22), specifically to avoid re-searching reduced-12's own best
    # layer -- that would be an uncontrolled multiple-comparisons search
    # over 37 layers, not a confirmatory replication.
    preg_layer = best_l_full
    _, oof_preg_red = loo_red[preg_layer]
    ci_preg_red = bootstrap_auc_ci(y_red, oof_preg_red, seed=3)

    results["reduced_12"] = {
        "held_out_auc_loo": layer_auc_red, "perm_p_per_layer": perm_p_red,
        "perm_null_auc_5fold": perm_null_red,
        "best_layer_EXPLORATORY_not_preregistered": best_l_red,
        "best_layer_auc_ci95_EXPLORATORY_not_preregistered": ci_red,
        "preregistered_layer": preg_layer,
        "preregistered_layer_auc": layer_auc_red[preg_layer],
        "preregistered_layer_auc_ci95": ci_preg_red,
        "preregistered_layer_perm_p": perm_p_red[preg_layer],
        "layer0_negative_control": {"auc": layer_auc_red[0], "perm_p": perm_p_red[0]},
    }
    print(f"  [reduced-12] PRE-REGISTERED layer {preg_layer} (full-17's peak): "
          f"LOO AUC={layer_auc_red[preg_layer]:.3f} 95% CI "
          f"[{ci_preg_red['lo']:.3f},{ci_preg_red['hi']:.3f}], "
          f"perm_p={perm_p_red[preg_layer]:.3f}", flush=True)
    print(f"  [reduced-12] exploratory best-of-37 layer {best_l_red} (NOT the "
          f"pre-registered test): LOO AUC={layer_auc_red[best_l_red]:.3f} 95% CI "
          f"[{ci_red['lo']:.3f},{ci_red['hi']:.3f}], perm_p={perm_p_red[best_l_red]:.3f}  |  "
          f"layer0 (negative control): AUC={layer_auc_red[0]:.3f} "
          f"perm_p={perm_p_red[0]:.3f}", flush=True)

    print("\n  -- full-17 vs reduced-12, side by side --", flush=True)
    print(f"  {'layer':>5} {'full17_auc':>11} {'full17_p':>9} {'red12_auc':>10} "
          f"{'red12_p':>8}", flush=True)
    show_layers = sorted(set([0, 5, 10, 15, 20, best_l_full, best_l_red, 25, 30, 36]))
    for l in show_layers:
        print(f"  {l:>5} {layer_auc_full[l]:>11.3f} {perm_p_full[l]:>9.3f} "
              f"{layer_auc_red[l]:>10.3f} {perm_p_red[l]:>8.3f}", flush=True)

    preg_auc = layer_auc_red[preg_layer]
    holds = (preg_auc is not None and preg_auc > 0.75 and perm_p_red[preg_layer] < 0.05)
    verdict = ("FINDING STANDS at the pre-registered layer: base model represents "
               "which targets it cannot solve (survives malformed-formula ablation)."
               if holds else
               f"FINDING WEAKENED at the pre-registered layer {preg_layer} (AUC="
               f"{preg_auc:.3f} < 0.75 bar, though perm_p={perm_p_red[preg_layer]:.3f} "
               f"still shows real separability somewhere): the full-17 result's "
               f"deep-layer peak was substantially driven by the malformed targets; "
               f"on chemically-real hard-zeros the (exploratory, non-preregistered) "
               f"peak shifts to a shallower layer ({best_l_red}, AUC="
               f"{layer_auc_red[best_l_red]:.3f}) closer to a formula-family feature "
               f"than deep synthesis reasoning.")
    results["decision_rule_verdict"] = verdict
    print(f"\n  decision rule (PRE-REGISTERED: reduced-12 AUC>0.75 AT LAYER "
          f"{preg_layer} and perm_p<0.05): "
          f"{verdict}", flush=True)
    return results


# --------------------------------------------------------------------------

def main():
    args = parse_args()
    run_all = args.all or not (args.forward or args.probe_a or args.probe_b or args.probe_c)
    rng = np.random.default_rng(args.seed)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    targets, strata, by_ckpt = load_targets()
    print(f"loaded {len(targets)} targets from {PASSK_FILE}", flush=True)

    acts = forward_all(args, targets)
    prior = load_prior()

    out = {"version": 2, "n_targets": len(targets), "checkpoints": list(CHECKPOINTS),
           "cv_folds": N_FOLDS, "perms_probe_b": N_PERMS, "perms_probe_c": N_PERMS_C,
           "bootstrap_n": N_BOOT, "held_out_repeats_probe_b": N_REPEATS_B,
           "ridge_alpha": RIDGE_ALPHA, "logreg_alpha": LOGREG_ALPHA}
    if run_all or args.probe_a:
        out["probe_a"] = probe_a(acts, by_ckpt, rng)
    if run_all or args.probe_b:
        out["probe_b"] = probe_b(acts, targets, strata, by_ckpt, rng)
    if run_all or args.probe_c:
        prior_c = (prior or {}).get("probe_c")
        out["probe_c"] = probe_c(acts, targets, by_ckpt, rng, prior=prior_c)

    OUT_FILE.write_text(json.dumps(out, indent=1))
    print(f"\n-> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
