import torch
from torch import nn

from cbml_benchmark.losses.registry import LOSS


@LOSS.register('cbml_loss')
class CBMLLoss(nn.Module):
    """
    Positive side: replaced with a HARD (single, closest) prototype from
    the anchor's OWN class only. Negative side: UNCHANGED from step 1 --
    real instances from other classes in the batch, hard-mined via margin,
    soft-aggregated (summed) via the same exp/log formula as before.

    KEY CONSTRAINT (per supervisor): class c's prototypes are used, and
    therefore only ever receive gradient, when class c is the POSITIVE
    class for a given anchor. They never appear on the negative side and
    never interact with any other class's data. This is a deliberate
    correction versus the earlier soft-posterior diagnostic, where every
    class's prototypes appeared in every sample's denominator regardless
    of that sample's own class.

    "HARD" HERE MEANS CLOSEST, NOT FARTHEST -- this is a deliberate
    departure from step 2's "hardest positive instance" (which was the
    FARTHEST same-class instance, in the triplet-mining sense). Your own
    paper's positive-prototype selection rule (Eq. 22-24) picks the
    prototype with the highest likelihood, i.e. the prototype CLOSEST to
    the anchor -- the best-fitting mixture component, analogous to a
    hard cluster assignment. This implementation follows that rule. If you
    intended "hardest" in the farthest/triplet sense instead, that's a
    one-line change (torch.min -> torch.max below) -- flag it and I'll
    give you that variant too.

    No margin-based mining is applied to the positive side: with a small,
    fixed K per class, all K prototypes are simply the candidate set, and
    the single closest one is selected directly, with no filtering step.
    Negative-side mining is unchanged, but now compares against the
    positive PROTOTYPE distances (dist_pos, K values) in place of the
    positive INSTANCE distances (pos_pair_) used in step 1 -- i.e. wherever
    step 1's mining referenced pos_pair_, this uses dist_pos instead.

    The sigma_ regularizer is likewise recomputed from dist_pos (prototype
    distances) blended with neg_pair_ (real negative instances), replacing
    step 1's pos_pair_-based version -- same formula, new source for the
    positive-side statistics.
    """

    def __init__(self, cfg):
        super(CBMLLoss, self).__init__()
        self.pos_a = cfg.LOSSES.CBML_LOSS.POS_A
        self.pos_b = cfg.LOSSES.CBML_LOSS.POS_B
        self.neg_a = cfg.LOSSES.CBML_LOSS.NEG_A
        self.neg_b = cfg.LOSSES.CBML_LOSS.NEG_B
        self.margin = cfg.LOSSES.CBML_LOSS.MARGIN
        self.weight = cfg.LOSSES.CBML_LOSS.WEIGHT
        self.hyper_weight = cfg.LOSSES.CBML_LOSS.HYPER_WEIGHT
        self.adaptive_neg = cfg.LOSSES.CBML_LOSS.ADAPTIVE_NEG
        self.type = cfg.LOSSES.CBML_LOSS.TYPE
        self.loss_weight_p = cfg.LOSSES.CBML_LOSS.WEIGHT_P
        self.loss_weight_n = cfg.LOSSES.CBML_LOSS.WEIGHT_N

        self.device_name = getattr(cfg.MODEL, 'DEVICE', 'cuda')
        self.device = torch.device(self.device_name)
        self.embed_dim = getattr(cfg.MODEL.HEAD, 'DIM', 512)
        self.num_classes = getattr(cfg.LOSSES.CBML_LOSS, 'N_CLASSES', 100)
        self.prototype_per_class = getattr(cfg.LOSSES.CBML_LOSS, 'PROTOTYPE_PER_CLASS', 3)

        # Positive-only, per-class prototypes [C, K, D]. Never referenced on
        # the negative side, so a class's prototypes are structurally
        # isolated from every other class's data.
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

    @torch.no_grad()
    def set_prototypes_and_weights(self, prototypes, cluster_sizes=None):
        """Call with KMeans-derived prototypes before training. Zero-init
        prototypes would make the initial closest-prototype selection an
        arbitrary tie-break."""
        prototypes = prototypes.to(self.device)
        self.prototypes.copy_(prototypes)
        torch.cuda.empty_cache()

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)

        # Real-instance pairwise distances, needed only for the negative side.
        dist_mat = torch.cdist(feats, feats, p=2) ** 2
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            y_i = labels[i]

            # --- Positive side: distances to the anchor's OWN class's K
            # prototypes only. No other class's prototypes or data involved.
            own_prototypes = self.prototypes[y_i]                     # [K, D]
            dist_pos = torch.sum((feats[i].unsqueeze(0) - own_prototypes) ** 2, dim=1)  # [K]

            neg_pair_ = dist_mat[i][labels != labels[i]]
            if len(neg_pair_) < 1:
                continue

            # sigma_ regularizer: same formula as step 1, but the positive
            # side now comes from dist_pos (prototype distances) instead of
            # pos_pair_ (real same-class instance distances).
            mean_ = self.hyper_weight * torch.mean(dist_pos) + (1 - self.hyper_weight) * torch.mean(neg_pair_)
            sigma_ = torch.mean(torch.sum(torch.pow(neg_pair_ - mean_, 2)))

            # --- Negative-side mining: UNCHANGED logic from step 1, except
            # references to pos_pair_ are replaced by dist_pos.
            if self.adaptive_neg:
                np_ = neg_pair_ - self.margin < torch.max(dist_pos)
                neg_pair = neg_pair_[np_]
            else:
                np_ = torch.argsort(neg_pair_)
                neg_pair = neg_pair_[np_[:100]]

            if len(neg_pair) < 1:
                continue

            # --- Positive side: HARD selection = closest prototype
            # (highest likelihood, per Eq. 22-24). See class docstring for
            # the closest-vs-farthest distinction.
            hardest_pos = torch.min(dist_pos)

            if self.type == 'log' or self.type == 'sqrt':
                fp = 1. + torch.exp(1. / self.pos_b * (hardest_pos - self.pos_a))
                fn = 1. + torch.sum(torch.exp(-1. / self.neg_b * (neg_pair - self.neg_a)))
                if self.type == 'log':
                    pos_loss = torch.log(fp)
                    neg_loss = torch.log(fn)
                else:
                    pos_loss = torch.sqrt(fp)
                    neg_loss = torch.sqrt(fn)
            else:
                pos_loss = 1. + self.loss_weight_p * torch.exp(1. / self.pos_b * (hardest_pos - self.pos_a))
                neg_loss = 1. + self.loss_weight_n * torch.sum(torch.exp(-1. / self.neg_b * (neg_pair - self.neg_a)))

            pos_neg_loss = sigma_
            loss.append((pos_loss + neg_loss + self.weight * pos_neg_loss))

        if len(loss) == 0:
            return torch.zeros(1, requires_grad=True, device=self.device)

        loss = sum(loss) / batch_size
        return loss
