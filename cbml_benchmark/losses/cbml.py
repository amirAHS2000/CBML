import torch
from torch import nn

from cbml_benchmark.losses.registry import LOSS


@LOSS.register('cbml_loss')
class CBMLLoss(nn.Module):
    """
    Vectorized rewrite of CBMLLossKDEDirectEq3. Mathematically identical --
    verified against the loop-based reference on synthetic data to within
    floating-point noise (~1e-16) -- but replaces the nested Python loop
    (batch anchors x classes present, ~B*U iterations) with a handful of
    matrix operations, which is where essentially all the runtime was going.

    KEY TRICK: a_ij = -(beta/2)*dist(x_i,x_j) <= 0 always, and the diagonal
    a_ii = 0 exactly (self-distance is always 0). So every row's maximum
    entry is exactly 0, meaning exp(a) in [0,1] with NO overflow risk --
    this lets per-class sums be computed as a single matrix multiply against
    a one-hot class-membership matrix, instead of the usual logsumexp
    max-subtraction dance done per class per anchor.

    Let H in {0,1}^{B x U} be the one-hot class-membership matrix (H[i,u]=1
    iff sample i belongs to the u-th class present in the batch). Then:
        exp_a       = exp(-beta/2 * dist_mat)             [B, B], in (0,1]
        S_full      = exp_a @ H                           [B, U]  (per-class sums)
        n_full      = H.sum(0)                            [U]     (per-class counts)
        S_adj       = S_full - H       (subtract self-term, always exactly 1,
                                         only from each row's OWN class column)
        n_adj       = n_full[None,:] - H   (subtract 1 from own-class count only)
        log_ell     = log(S_adj) - log(n_adj)             [B, U]  (Eq. 44/46 fused)
        logsumexp_all = logsumexp(log_ell, dim=1)         [B]     (Eq. 45's first term)
        true_ld     = (log_ell * H).sum(dim=1)            [B]     (Eq. 45's second term,
                                                                    H picks the one
                                                                    true-class column)
        main_loss   = logsumexp_all - true_ld             [B]

    A sample is only invalid (skipped) when it is the SOLE batch member of
    its own class (n_adj at its own-class column would be 0) -- matches the
    original loop version's skip condition exactly.

    The MVC regularizer is likewise vectorized via same-class/different-class
    boolean masks instead of per-row masking inside a loop, using the exact
    same sigma_ formula (sum, not mean, over negatives -- matching the
    original cbml.py precisely) as every other file in this series.
    """

    def __init__(self, cfg):
        super(CBMLLoss, self).__init__()
        self.hyper_weight = cfg.LOSSES.CBML_LOSS.HYPER_WEIGHT
        self.weight = cfg.LOSSES.CBML_LOSS.WEIGHT

        self.device_name = getattr(cfg.MODEL, 'DEVICE', 'cuda')
        self.device = torch.device(self.device_name)

        theta_is_learnable = getattr(cfg.LOSSES.CBML_LOSS, 'THETA_IS_LEARNABLE', True)
        init_theta = getattr(cfg.LOSSES.CBML_LOSS, 'INIT_THETA', 1.0)
        self.theta = nn.Parameter(
            torch.tensor(init_theta, device=self.device),
            requires_grad=theta_is_learnable,
        )

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        eps = 1e-12

        feats = feats.to(self.device)
        labels = labels.to(self.device)
        beta = torch.exp(self.theta)

        dist_mat = torch.cdist(feats, feats, p=2) ** 2  # [B, B]

        unique_classes = torch.unique(labels)
        U = unique_classes.numel()
        if U < 2:
            return torch.zeros(1, requires_grad=True, device=self.device)

        # One-hot class-membership matrix [B, U].
        H = (labels.unsqueeze(1) == unique_classes.unsqueeze(0)).to(dist_mat.dtype)

        # --- Main loss: vectorized Eq. 44-46 + Eq. 45 ---
        exp_a = torch.exp(-0.5 * beta * dist_mat)          # [B, B], bounded in (0, 1]
        S_full = exp_a @ H                                  # [B, U]
        n_full = H.sum(dim=0)                               # [U]

        S_adj = S_full - H                                  # subtract self-term from own class only
        n_adj = n_full.unsqueeze(0) - H                     # subtract 1 from own-class count only

        log_ell = torch.log(torch.clamp(S_adj, min=eps)) - torch.log(torch.clamp(n_adj, min=1))

        logsumexp_all = torch.logsumexp(log_ell, dim=1)     # [B]
        true_ld = (log_ell * H).sum(dim=1)                  # [B]
        main_loss = logsumexp_all - true_ld                 # [B]

        own_class_count = (H * n_full.unsqueeze(0)).sum(dim=1)  # [B]
        main_valid = own_class_count > 1                        # False iff sample is sole member of its class

        # --- MVC regularizer: vectorized, same formula as every other file ---
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)          # [B, B]
        eye = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        pos_mask = same_class & ~eye
        neg_mask = ~same_class

        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        neg_count = neg_mask.sum(dim=1).clamp(min=1)
        pos_mean = (dist_mat * pos_mask).sum(dim=1) / pos_count
        neg_mean = (dist_mat * neg_mask).sum(dim=1) / neg_count
        mean_ = self.hyper_weight * pos_mean + (1 - self.hyper_weight) * neg_mean  # [B]

        diff_sq = (dist_mat - mean_.unsqueeze(1)) ** 2
        sigma_ = (diff_sq * neg_mask).sum(dim=1)  # [B], SUM not mean, matching cbml.py exactly

        mvc_valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)

        valid = main_valid & mvc_valid
        sample_loss = main_loss + self.weight * sigma_
        sample_loss = torch.where(valid, sample_loss, torch.zeros_like(sample_loss))

        return sample_loss.sum() / batch_size
