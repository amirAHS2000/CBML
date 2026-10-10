"""
Generalization diagnostics for metric learning (no change to the training loss).

Everything here is computed from L2-normalised embeddings that the trainer has
already extracted in eval mode (no augmentation) for
  * the TRAIN classes   (train-eval subsample), and
  * the TEST  classes   (validation set),
so the two splits are measured with exactly the same code and are comparable.

Definitions (m = cosine similarity; anchor x in class c; P_x = other images of c;
N_x = images of all other classes; triplet margin  D = m(x,x+) - m(x,x-)):

  V1 = E_x[ Var_{j in P_x} m(x,j) + Var_{k in N_x} m(x,k) ]     within-anchor
  V2 = E_c[ Var_{x in c} Dbar_x ]                                across anchors, within class
  V3 = Var_c[ E_{x in c} Dbar_x ]                                across classes
  with Dbar_x = mean_{j in P_x} m - mean_{k in N_x} m.

  Law of total variance (exact for a class-balanced sample):  Var(D) = V1 + V2 + V3.
  `identity_err` below checks this numerically against a brute-force Var(D).

The sample is made exactly class-balanced (same number of images p per class,
classes with fewer than p images are dropped) so that the identity is exact and
the train / test numbers are not distorted by different class sizes.
All variances are population variances (ddof=0) for the identity; V_cls (the
across-class variance of the per-class surrogate loss) uses ddof=1.
"""
import json
import math
import os

import numpy as np


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _to_numpy(x):
    if hasattr(x, "detach"):          # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _unit_rows(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _balanced_subsample(feats, labels, per_class, seed):
    """Exactly `per_class` images for every class that has at least that many."""
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < per_class:
            continue
        keep.extend(rng.choice(idx, per_class, replace=False).tolist())
    keep = np.array(sorted(keep), dtype=np.int64)
    return feats[keep], labels[keep]


def _phi(x):
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _effective_rank(X):
    """Participation ratio (sum lambda)^2 / sum lambda^2 of the covariance spectrum."""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    s = s / max(len(X) - 1, 1)
    return float((s.sum() ** 2) / max((s ** 2).sum(), 1e-30))


# ----------------------------------------------------------------------------
# main computation
# ----------------------------------------------------------------------------
def compute_generalization_stats(feats, labels, per_class=8, seed=0,
                                 ramp_gamma=0.1, xi_gamma=0.2):
    X = _to_numpy(feats).astype(np.float64)
    y = _to_numpy(labels).astype(np.int64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    X, y = _balanced_subsample(X, y, per_class, seed)
    classes = np.unique(y)
    C, p = len(classes), per_class
    if C < 3 or p < 3:
        raise ValueError(f"need >=3 classes and >=3 images/class, got C={C}, p={p}")
    order = np.argsort(y, kind="stable")
    X, y = X[order], y[order]
    assert len(X) == C * p

    S = X @ X.T                                   # [N, N] cosine similarities
    Ntot = C * p
    # exact: sum_ij <x_i,x_j> = N^2 ||xbar||^2  ->  squared norm of the mean embedding
    mean_emb_norm_sq = float(S.sum() / Ntot ** 2)
    mean_sim_offdiag = float((S.sum() - Ntot) / (Ntot * (Ntot - 1)))
    Sfull = S.reshape(C, p, C * p)

    # positives: off-diagonal of each class block -> [C, p, p-1]
    off = ~np.eye(p, dtype=bool)
    pos = np.empty((C, p, p - 1))
    for c in range(C):
        blk = Sfull[c][:, c * p:(c + 1) * p]      # [p, p]
        pos[c] = blk[off].reshape(p, p - 1)
    # negatives: all columns of other classes -> [C, p, (C-1)*p]
    neg = np.empty((C, p, (C - 1) * p))
    for c in range(C):
        neg[c] = np.concatenate([Sfull[c][:, :c * p], Sfull[c][:, (c + 1) * p:]], axis=1)

    muP, muN = pos.mean(-1), neg.mean(-1)         # [C, p]
    varP, varN = pos.var(-1), neg.var(-1)         # population variance
    dbar = muP - muN                              # [C, p]
    cls_dbar = dbar.mean(1)                       # [C]

    V1_pos, V1_neg = float(varP.mean()), float(varN.mean())
    V1 = V1_pos + V1_neg
    V2 = float(dbar.var(1).mean())
    V3 = float(cls_dbar.var())
    mean_delta = float(dbar.mean())

    # brute-force triplet margins D[c, a, j, k] = m(a,j) - m(a,k)
    D = pos[:, :, :, None] - neg[:, :, None, :]
    total_var = float(D.var())
    identity_err = abs(V1 + V2 + V3 - total_var)
    sd_delta = math.sqrt(max(total_var, 1e-30))
    z_trip = mean_delta / sd_delta
    emp_trip_err = float((D <= 0).mean())
    cls_trip_err = (D <= 0).mean(axis=(1, 2, 3))                      # [C]
    # ramp surrogate used in the generalization bound
    phi = np.clip(1.0 - D / ramp_gamma, 0.0, 1.0)
    L_cls = phi.mean(axis=(1, 2, 3))                                  # [C]
    del D, phi

    # pair-level (equal-variance Gaussian fit): alpha = (muP+muN)/2, beta = s^2/(muP-muN)
    gp, gn = pos.reshape(-1), neg.reshape(-1)
    mP, mN = float(gp.mean()), float(gn.mean())
    s2 = 0.5 * (float(gp.var()) + float(gn.var()))
    d_prime = (mP - mN) / math.sqrt(max(s2, 1e-30))
    g_alpha = 0.5 * (mP + mN)
    g_beta = s2 / (mP - mN) if abs(mP - mN) > 1e-12 else float("nan")

    # paper's MVC quantity and its exact split  Var_N + gamma^2 (muP-muN)^2
    xi = xi_gamma * muP + (1.0 - xi_gamma) * muN
    mvc_direct = ((neg - xi[..., None]) ** 2).mean(-1)
    mvc_var_part = varN
    mvc_gap_part = (xi_gamma ** 2) * dbar ** 2
    mvc_identity_err = float(np.abs(mvc_direct - (mvc_var_part + mvc_gap_part)).max())

    # geometry (Lemma 1): class centres, radii, separations (angles, radians)
    Xc = X.reshape(C, p, -1)
    mu = Xc.mean(1)
    mu = mu / np.maximum(np.linalg.norm(mu, axis=1, keepdims=True), 1e-12)
    rad = np.arccos(np.clip(np.einsum("cpd,cd->cp", Xc, mu), -1.0, 1.0))   # [C, p]
    r_cls = rad.mean(1)
    G = np.arccos(np.clip(mu @ mu.T, -1.0, 1.0))
    offc = ~np.eye(C, dtype=bool)
    nn_sep = np.where(offc, G, np.inf).min(1)
    # heuristic Lemma-1 test with class-mean radii: theta(c,c') > 3 r_c + r_c'
    cond = G > (3.0 * r_cls[:, None] + r_cls[None, :])
    lemma1_frac = float(cond[offc].mean())
    lemma1_frac_nn = float(np.mean([G[c, np.argmin(np.where(offc[c], G[c], np.inf))]
                                    > 3.0 * r_cls[c] + r_cls[np.argmin(np.where(offc[c], G[c], np.inf))]
                                    for c in range(C)]))
    deg = 180.0 / math.pi

    return {
        "n_classes": int(C), "per_class": int(p),
        "mean_emb_norm_sq": mean_emb_norm_sq, "mean_emb_norm": math.sqrt(max(mean_emb_norm_sq, 0.0)),
        "mean_sim_offdiag": mean_sim_offdiag,
        "V1": V1, "V1_pos": V1_pos, "V1_neg": V1_neg, "V2": V2, "V3": V3,
        "var_total": total_var, "identity_err": identity_err,
        "mean_delta": mean_delta, "sd_delta": sd_delta, "z_trip": z_trip,
        "trip_err_emp": emp_trip_err, "trip_err_gauss": _phi(-z_trip),
        "trip_err_cls_std": float(cls_trip_err.std()),
        "trip_err_cls_p90": float(np.percentile(cls_trip_err, 90)),
        "ramp_gamma": ramp_gamma, "L_cls_mean": float(L_cls.mean()),
        "V_cls": float(L_cls.var(ddof=1)),
        "d_prime_pair": d_prime, "gauss_alpha": g_alpha, "gauss_beta": g_beta,
        "mu_pos": mP, "mu_neg": mN,
        "mvc_paper": float(mvc_direct.mean()), "mvc_var_part": float(mvc_var_part.mean()),
        "mvc_gap_part": float(mvc_gap_part.mean()), "mvc_identity_err": mvc_identity_err,
        "radius_deg": float(rad.mean() * deg), "radius_cls_std_deg": float(r_cls.std() * deg),
        "center_sep_mean_deg": float(G[offc].mean() * deg),
        "center_sep_nn_deg": float(nn_sep.mean() * deg),
        "lemma1_frac": lemma1_frac, "lemma1_frac_nn": lemma1_frac_nn,
        "eff_rank_all": _effective_rank(X), "eff_rank_class_means": _effective_rank(mu),
    }


# keys reported as test - train in the "gap" record
_GAP_KEYS = ["V1", "V2", "V3", "mean_delta", "z_trip", "trip_err_emp", "V_cls",
             "d_prime_pair", "radius_deg", "center_sep_nn_deg", "eff_rank_all",
             "eff_rank_class_means", "mvc_paper", "mean_emb_norm_sq"]


# ----------------------------------------------------------------------------
# logger
# ----------------------------------------------------------------------------
class GenDiagLogger:
    """Writes one JSON line per (iteration, split) to <out_dir>/gen_diag.jsonl."""

    def __init__(self, out_dir, per_class=8, seed=0, ramp_gamma=0.1, xi_gamma=0.2):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, "gen_diag.jsonl")
        open(self.path, "w").close()              # fresh file for every run
        self.kw = dict(per_class=per_class, seed=seed, ramp_gamma=ramp_gamma, xi_gamma=xi_gamma)

    def _write(self, rec):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_pair(self, iteration, train_feats, train_labels, test_feats, test_labels, logger=None):
        """Never raises: diagnostics must not be able to kill a training run."""
        try:
            tr = compute_generalization_stats(train_feats, train_labels, **self.kw)
            te = compute_generalization_stats(test_feats, test_labels, **self.kw)
        except Exception as e:                     # noqa
            if logger is not None:
                logger.warning(f"[gen-diag] skipped at iteration {iteration}: {e!r}")
            return None
        self._write({"iter": int(iteration), "split": "train", **tr})
        self._write({"iter": int(iteration), "split": "test", **te})
        gap = {k: te[k] - tr[k] for k in _GAP_KEYS}
        self._write({"iter": int(iteration), "split": "gap", **gap})
        if logger is not None:
            for name, s in (("train", tr), ("test ", te)):
                logger.info(
                    f"[gen-diag {name}] it={iteration} C={s['n_classes']} "
                    f"V1={s['V1']:.5f} V2={s['V2']:.5f} V3={s['V3']:.5f} "
                    f"Dbar={s['mean_delta']:.4f} z={s['z_trip']:.3f} "
                    f"trip_err={s['trip_err_emp']:.4f} Vcls={s['V_cls']:.5f} "
                    f"radius={s['radius_deg']:.1f}deg sep_nn={s['center_sep_nn_deg']:.1f}deg "
                    f"effrank={s['eff_rank_all']:.1f}/{s['eff_rank_class_means']:.1f} "
                    f"|fbar|^2={s['mean_emb_norm_sq']:.3f} "
                    f"(identity_err={s['identity_err']:.1e})")

        # same statistics after removing the common direction (mean embedding):
        #   train_c    : train features centred with the TRAIN mean
        #   test_c     : test  features centred with the TRAIN mean   (deployable)
        #   test_c_own : test  features centred with their OWN mean   (transductive reference)
        try:
            Xtr = _unit_rows(_to_numpy(train_feats))
            Xte = _unit_rows(_to_numpy(test_feats))
            m_tr = Xtr.mean(0, keepdims=True)
            cen = {
                "train_c": compute_generalization_stats(Xtr - m_tr, train_labels, **self.kw),
                "test_c": compute_generalization_stats(Xte - m_tr, test_labels, **self.kw),
                "test_c_own": compute_generalization_stats(Xte - Xte.mean(0, keepdims=True), test_labels, **self.kw),
            }
            for name, s in cen.items():
                self._write({"iter": int(iteration), "split": name, **s})
            if logger is not None:
                logger.info(
                    "[gen-diag centered] it=%d " % iteration + " | ".join(
                        f"{n}: Dbar={s['mean_delta']:.3f} z={s['z_trip']:.2f} err={s['trip_err_emp']:.4f} "
                        f"sep_nn={s['center_sep_nn_deg']:.1f}deg effrank={s['eff_rank_all']:.1f}"
                        for n, s in cen.items()))
        except Exception as e:                     # noqa
            if logger is not None:
                logger.warning(f"[gen-diag] centered stats skipped at iteration {iteration}: {e!r}")
        return tr, te


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------
def plot_gen_diag(jsonl_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]
    if not tr or not te:
        return
    it = [r["iter"] for r in tr]

    def line(ax, key, title, both=True):
        ax.plot(it, [r[key] for r in tr], "-o", ms=3, label="train classes")
        if both:
            ax.plot(it, [r[key] for r in te], "-s", ms=3, label="test classes")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.3)

    fig, axs = plt.subplots(3, 5, figsize=(22, 11))
    line(axs[0, 0], "V1", "V1 within-anchor variance")
    line(axs[0, 1], "V2", "V2 across anchors (within class)")
    line(axs[0, 2], "V3", "V3 across classes")
    line(axs[0, 3], "var_total", "Var(D) = V1+V2+V3")
    line(axs[0, 4], "mean_emb_norm_sq", "|mean embedding|^2 (= mean pair sim.)")
    line(axs[1, 0], "mean_delta", "E[D] (mean triplet margin)")
    line(axs[1, 1], "z_trip", "z = E[D]/sd(D)")
    line(axs[1, 2], "trip_err_emp", "empirical triplet error P(D<=0)")
    line(axs[1, 3], "V_cls", "V_cls (across-class var. of ramp loss)")
    line(axs[1, 4], "d_prime_pair", "d' (pair level)")
    line(axs[2, 0], "radius_deg", "class radius (deg)")
    line(axs[2, 1], "center_sep_nn_deg", "nearest class-centre separation (deg)")
    line(axs[2, 2], "eff_rank_all", "effective rank (all features)")
    line(axs[2, 3], "mvc_paper", "paper MVC quantity (xi-gamma)")
    ax = axs[2, 4]
    for nm, lab in (("train_c", "train (train-mean centred)"), ("test_c", "test (train-mean centred)"),
                    ("test_c_own", "test (own-mean centred)")):
        rr = [r for r in recs if r["split"] == nm]
        if rr:
            ax.plot([r["iter"] for r in rr], [r["z_trip"] for r in rr], "-o", ms=3, label=lab)
    ax.set_title("z after centring", fontsize=9)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    axs[0, 0].legend(fontsize=8)
    for ax in axs[2]:
        ax.set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
