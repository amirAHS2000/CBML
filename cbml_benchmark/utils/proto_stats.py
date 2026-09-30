"""Prototype statistics logging + alpha/beta fitting for the proxy-CBML loss.

Training loop:
    logger = ProtoStatsLogger(criterion, out_dir=os.path.join(cfg.OUTPUT_DIR, "proto_stats"))
    ...
    loss = criterion(feats, labels)
    logger.update(feats, labels, step=it)          # cheap, no grad, flushes automatically
    ...
    # after reinitialize_prototypes(...):
    logger.set_reference()

Offline fit of alpha/beta from the saved samples:
    python proto_stats.py <out_dir>
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

QS = (0.05, 0.25, 0.5, 0.75, 0.95)


def _dist(x, prefix):
    x = x.float()
    out = {f"{prefix}_mean": x.mean().item(),
           f"{prefix}_std": x.std().item() if x.numel() > 1 else 0.0}
    qs = torch.quantile(x, torch.tensor(QS, device=x.device))
    for q, v in zip(QS, qs.tolist()):
        out[f"{prefix}_q{int(round(q * 100)):02d}"] = v
    return out


class ProtoStatsLogger:
    def __init__(self, criterion, out_dir, flush_every=100, geometry_every=500,
                 reset_counts_every=1000, save_samples=20000, min_cls_samples=5,
                 writer=None):
        self.crit = criterion
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.flush_every = flush_every
        self.geometry_every = geometry_every
        self.reset_counts_every = reset_counts_every
        self.save_samples = save_samples
        self.min_cls_samples = min_cls_samples
        self.writer = writer                      # optional tensorboard SummaryWriter
        self.jsonl = os.path.join(out_dir, "stats.jsonl")

        C, K = criterion.num_classes, criterion.prototypes_per_class
        dev = criterion.prototypes.device
        self.pos_counts = torch.zeros(C * K, dtype=torch.long, device=dev)
        self.neg_counts = torch.zeros(C * K, dtype=torch.long, device=dev)
        self.buf = []
        self._last_geom = None
        self._last_reset = 0
        self.set_reference()

    @torch.no_grad()
    def set_reference(self):
        self.ref = F.normalize(self.crit.prototypes.detach().clone(), dim=-1)
        self.pos_counts.zero_()
        self.neg_counts.zero_()
        self.buf = []
        self._last_geom = None      # forces a geometry snapshot at the next flush

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(self, feats, labels, step):
        c = self.crit
        C, K = c.num_classes, c.prototypes_per_class
        dev = c.prototypes.device
        feats = feats.detach().float().to(dev)     # used as-is, exactly like the loss
        labels = labels.to(dev).long()

        P = F.normalize(c.prototypes.detach(), dim=-1)
        sim = torch.einsum("bd,ckd->bck", feats, P)            # [B,C,K]
        oh = F.one_hot(labels, C).bool()                       # [B,C]
        s_pos, pos_k = sim[oh].max(1)                          # [B,K] -> [B]
        s_neg, neg_flat = sim.masked_fill(oh[:, :, None], float("-inf")).flatten(1).max(1)

        self.pos_counts += torch.bincount(labels * K + pos_k, minlength=C * K)
        self.neg_counts += torch.bincount(neg_flat, minlength=C * K)
        self.buf.append((s_pos, s_neg, feats.norm(dim=1)))

        if (step + 1) % self.flush_every == 0:
            self.flush(step + 1)

    # ------------------------------------------------------------------- flush
    @torch.no_grad()
    def flush(self, step):
        if not self.buf:
            return
        c = self.crit
        s_pos = torch.cat([b[0] for b in self.buf])
        s_neg = torch.cat([b[1] for b in self.buf])
        fnorm = torch.cat([b[2] for b in self.buf])
        self.buf = []

        a_p, b_p, a_n, b_n = c.pos_a, c.pos_b, c.neg_a, c.neg_b
        z_p = -(s_pos - a_p) / b_p
        z_n = (s_neg - a_n) / b_n
        g_p, g_n = torch.sigmoid(z_p), torch.sigmoid(z_n)    # = d(softplus)/dz, the "gate"

        st = {
            "step": step, "n": s_pos.numel(), "feat_norm": fnorm.mean().item(),
            "params_apbn": [a_p, b_p, a_n, b_n],
            "nearest_proto_err": (s_pos <= s_neg).float().mean().item(),
            "gate_pos_mean": g_p.mean().item(), "gate_neg_mean": g_n.mean().item(),
            "gate_pos_frac_gt_0.01": (g_p > 0.01).float().mean().item(),
            "gate_neg_frac_gt_0.01": (g_n > 0.01).float().mean().item(),
            "loss_pos": F.softplus(z_p).mean().item(),
            "loss_neg": F.softplus(z_n).mean().item(),
        }
        st.update(_dist(s_pos, "s_pos"))
        st.update(_dist(s_neg, "s_neg"))
        st.update(_dist(s_pos - s_neg, "gap"))
        st.update(self._usage())

        if self._last_geom is None or step - self._last_geom >= self.geometry_every:
            st.update(self._geometry())
            self._last_geom = step

        # subsample for offline alpha/beta fitting
        n = min(self.save_samples, s_pos.numel())
        idx = torch.randperm(s_pos.numel(), device=s_pos.device)[:n]
        np.savez_compressed(os.path.join(self.out_dir, f"samples_{step:06d}.npz"),
                            s_pos=s_pos[idx].cpu().numpy(), s_neg=s_neg[idx].cpu().numpy())

        if step - self._last_reset >= self.reset_counts_every:
            self.pos_counts.zero_()
            self.neg_counts.zero_()
            self._last_reset = step

        with open(self.jsonl, "a") as f:
            f.write(json.dumps(st) + "\n")
        if self.writer is not None:
            for k, v in st.items():
                if isinstance(v, (int, float)) and k != "step":
                    self.writer.add_scalar(f"proto/{k}", v, step)
        print(f"[proto {step}] s+={st['s_pos_mean']:.3f} s-={st['s_neg_mean']:.3f} "
              f"err={st['nearest_proto_err']:.3f} gate+={st['gate_pos_mean']:.3f} "
              f"gate-={st['gate_neg_mean']:.3f} L+={st['loss_pos']:.3f} L-={st['loss_neg']:.3f}")

    # ------------------------------------------------------------------- usage
    @torch.no_grad()
    def _usage(self):
        C, K = self.crit.num_classes, self.crit.prototypes_per_class
        pc = self.pos_counts.view(C, K).float()
        nc = self.neg_counts.float()
        n_cls = pc.sum(1)
        seen = n_cls >= self.min_cls_samples
        out = {"usage_samples": int(pc.sum().item()), "usage_classes_seen": int(seen.sum().item())}
        if K > 1 and seen.any():
            share = pc[seen] / n_cls[seen, None]
            top = share.max(1).values
            out["pos_dominant_share_mean"] = top.mean().item()
            out["pos_dominant_gt90_frac"] = (top > 0.9).float().mean().item()
            out["pos_unused_proxy_frac"] = (pc[seen] == 0).float().mean().item()
        tot = nc.sum().clamp(min=1)
        srt = nc.sort(descending=True).values
        out["neg_unused_proxy_frac"] = (nc == 0).float().mean().item()
        out["neg_top1_share"] = (srt[0] / tot).item()
        out["neg_top10_share"] = (srt[:10].sum() / tot).item()
        out["neg_classes_hit"] = int((nc.view(C, K).sum(1) > 0).sum().item())
        return out

    # ---------------------------------------------------------------- geometry
    @torch.no_grad()
    def _geometry(self, chunk=2048):
        c = self.crit
        C, K = c.num_classes, c.prototypes_per_class
        raw = c.prototypes.detach()
        P = F.normalize(raw, dim=-1)
        norms = raw.norm(dim=-1)
        out = {"proto_norm_mean": norms.mean().item(), "proto_norm_min": norms.min().item(),
               "proto_norm_max": norms.max().item()}

        if K > 1:
            g = torch.einsum("ckd,cld->ckl", P, P)
            off = ~torch.eye(K, dtype=torch.bool, device=P.device)
            out["intra_cos_mean"] = g[:, off].mean().item()

        Pf = P.reshape(C * K, -1)
        cls = torch.arange(C, device=P.device).repeat_interleave(K)
        best = []
        for i in range(0, C * K, chunk):
            s = Pf[i:i + chunk] @ Pf.T
            s.masked_fill_(cls[i:i + chunk, None] == cls[None, :], -2.0)
            best.append(s.max(1).values)
        best = torch.cat(best)
        out.update(_dist(best, "nn_inter_cos"))
        out["nn_inter_cos_max"] = best.max().item()

        drift = (P * self.ref).sum(-1).flatten()             # cosine to reference
        out["drift_cos_mean"] = drift.mean().item()
        out["drift_cos_q05"] = torch.quantile(drift, 0.05).item()
        return out


# --------------------------------------------------------------- alpha / beta
def fit_alpha_beta(s_pos, s_neg, C=1e3):
    """1-D logistic fit: P(same | s) = sigmoid((s - alpha) / beta)."""
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([s_pos, s_neg])[:, None]
    y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
    lr = LogisticRegression(C=C, max_iter=1000).fit(X, y)
    coef, b = lr.coef_[0, 0], lr.intercept_[0]
    return -b / coef, 1.0 / coef


def gaussian_alpha_beta(s_pos, s_neg):
    """Paper Eq. 19 (equal-variance Gaussians): alpha=(mu+ + mu-)/2, beta=sigma^2/(mu+ - mu-)."""
    mp, mn = s_pos.mean(), s_neg.mean()
    var = 0.5 * (s_pos.var() + s_neg.var())
    return (mp + mn) / 2, var / (mp - mn)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    print(f"{'step':>7} {'mean s+':>8} {'mean s-':>8} | {'alpha(LR)':>9} {'beta(LR)':>9} | {'alpha(G)':>9} {'beta(G)':>9}")
    for fn in sorted(glob.glob(os.path.join(out_dir, "samples_*.npz"))):
        d = np.load(fn)
        sp, sn = d["s_pos"], d["s_neg"]
        a1, b1 = fit_alpha_beta(sp, sn)
        a2, b2 = gaussian_alpha_beta(sp, sn)
        step = int(os.path.basename(fn)[8:14])
        print(f"{step:>7} {sp.mean():>8.3f} {sn.mean():>8.3f} | {a1:>9.3f} {b1:>9.4f} | {a2:>9.3f} {b2:>9.4f}")
