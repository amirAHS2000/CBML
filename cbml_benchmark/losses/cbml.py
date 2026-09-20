import math
import torch
from torch import nn

from cbml_benchmark.losses.registry import LOSS


@LOSS.register('cbml_loss')
class CBMLLoss(nn.Module):
    """
    Replaces the learned K-prototype Gaussian mixture (Eq. 1) with a
    leave-one-out KERNEL DENSITY ESTIMATE built from real training
    instances -- "a prototype per data point," moving from a prototype-based
    to a data-based density model -- and plugs it directly into the
    UNAPPROXIMATED Eq. 3 posterior (no hard-negative-class selection from
    Eq. 4, no hard-prototype selection from Eq. 5). The resulting loss is a
    direct instantiation of the MLE objective in Eq. 6-7, not the two-term
    softplus specialization in Eq. 10-14.

    DERIVATION SUMMARY (see chat for the full step-by-step):
      p_KDE(x_i | c) = (1/n_c) * sum_{k=1}^{n_c} N(x_i; x^c_k, sigma^2 I)
      log p_KDE(x_i | c) = -log(n_c) + logsumexp_k(-beta/2 * dist(x_i,x^c_k))
                           + const(beta, d)   [const cancels across classes,
                                                same shared-isotropic-sigma
                                                argument as before]
      L_i = logsumexp_{c in batch}(log p_KDE(x_i|c) + log p(c))
            - (log p_KDE(x_i|c+_i) + log p(c+_i))

    TWO DETAILS THAT MATTER AND ARE EASY TO GET WRONG:
      1. The -log(n_c) term is KEPT (not dropped): without it, a class with
         more same-class samples in the batch gets a systematically higher
         unnormalized density purely from summing more terms, not from
         being genuinely closer to x_i. Keeping it makes this an unbiased
         density estimate.
      2. LEAVE-ONE-OUT for the true class only: when a sample's own class is
         being scored, that sample itself is excluded from its own class's
         KDE sum (self-distance = 0 would otherwise let the loss cheat by
         recognizing the point itself rather than learning class structure).
         Other classes need no such exclusion since the anchor isn't a
         member of them.

    PRACTICAL CONSTRAINT, NOT A DESIGN CHOICE: with class-balanced batch
    sampling (NUM_INSTANCES per class), a batch contains only a SUBSET of
    all C classes. KDE cannot estimate a density for a class with zero
    batch samples, so "sum over c in C" is necessarily restricted to
    classes actually present in the current batch. If a sample is the ONLY
    batch member of its own class, leave-one-out leaves n_c = 0 for its true
    class and that sample is skipped (no valid density estimate possible).

    PRIOR ASSUMPTION: p(c) is assumed uniform across classes present in the
    batch, so the log p(c) term drops out. Flag if you want a class-frequency
    prior added back in.

    REGULARIZER: the Metric Variance Constraint (Eq. 28-30 of the original
    CBML paper) is kept EXACTLY as in step 1 / the MLE-form files -- same
    mean_/sigma_ formula over the full (non-leave-one-out, non-KDE)
    pos_pair_/neg_pair_ instance distances, added as +self.weight*sigma_.
    Only the main loss function is replaced; the regularizer is untouched,
    per your request.
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

        dist_mat = torch.cdist(feats, feats, p=2) ** 2
        epsilon = 1e-5
        beta = torch.exp(self.theta)
        unique_classes = torch.unique(labels)

        loss = list()

        for i in range(batch_size):
            y_i = labels[i]

            # --- MVC regularizer: unchanged, exactly as in step 1 / the
            # MLE-form files, computed over ordinary (non-leave-one-out)
            # same-class / different-class instance distances.
            pos_pair_ = dist_mat[i][labels == y_i]
            pos_pair_ = pos_pair_[pos_pair_ > epsilon]
            neg_pair_ = dist_mat[i][labels != y_i]
            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue
            mean_ = self.hyper_weight * torch.mean(pos_pair_) + (1 - self.hyper_weight) * torch.mean(neg_pair_)
            sigma_ = torch.mean(torch.sum(torch.pow(neg_pair_ - mean_, 2)))

            # --- Main loss: direct Eq. 3 posterior via leave-one-out KDE
            # over every class present in the batch.
            class_log_densities = []
            true_class_log_density = None

            for c in unique_classes:
                member_idx = (labels == c).nonzero(as_tuple=True)[0]
                if c == y_i:
                    member_idx = member_idx[member_idx != i]  # leave-one-out
                n_c = member_idx.numel()
                if n_c == 0:
                    continue  # only happens for the true class when x_i is its sole batch member

                d = dist_mat[i, member_idx]  # [n_c]
                log_density_c = -math.log(n_c) + torch.logsumexp(-0.5 * beta * d, dim=0)
                class_log_densities.append(log_density_c)
                if c == y_i:
                    true_class_log_density = log_density_c

            # Need the true class's density AND at least one other class to
            # form a meaningful softmax.
            if true_class_log_density is None or len(class_log_densities) < 2:
                continue

            all_logits = torch.stack(class_log_densities)
            logsumexp_all = torch.logsumexp(all_logits, dim=0)
            main_loss = logsumexp_all - true_class_log_density

            sample_loss = main_loss + self.weight * sigma_
            loss.append(sample_loss)

        if len(loss) == 0:
            return torch.zeros(1, requires_grad=True, device=self.device)

        loss = sum(loss) / batch_size
        return loss
