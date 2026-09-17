import torch
from torch import nn

from cbml_benchmark.losses.registry import LOSS


@LOSS.register('cbml_loss')
class CBMLLoss(nn.Module):
    """
    Same as CBMLLossPositiveHardPrototype (positive side = own-class-only
    prototypes, structurally isolated from negatives and other classes;
    negative side unchanged from step 1), EXCEPT the positive side sums the
    exp-margin term over ALL K of the anchor's own class's prototypes,
    instead of selecting only the single closest one. This is the direct
    prototype analogue of step 1's "sum over all mined positive instances"
    -- the only variable changed relative to
    CBMLLossPositiveHardPrototype is hard-select-one vs. soft-sum-over-all
    on the positive side.

    Comparing this against CBMLLossPositiveHardPrototype isolates
    hard-vs-soft specifically for the "positive side is now a prototype set"
    setting -- separate from the earlier step 1 vs step 2 comparison, which
    tested hard-vs-soft when the positive side was real instances.
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

        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

    @torch.no_grad()
    def set_prototypes_and_weights(self, prototypes, cluster_sizes=None):
        prototypes = prototypes.to(self.device)
        self.prototypes.copy_(prototypes)
        torch.cuda.empty_cache()

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)

        dist_mat = torch.cdist(feats, feats, p=2) ** 2
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            y_i = labels[i]

            own_prototypes = self.prototypes[y_i]                     # [K, D]
            dist_pos = torch.sum((feats[i].unsqueeze(0) - own_prototypes) ** 2, dim=1)  # [K]

            neg_pair_ = dist_mat[i][labels != labels[i]]
            if len(neg_pair_) < 1:
                continue

            mean_ = self.hyper_weight * torch.mean(dist_pos) + (1 - self.hyper_weight) * torch.mean(neg_pair_)
            sigma_ = torch.mean(torch.sum(torch.pow(neg_pair_ - mean_, 2)))

            if self.adaptive_neg:
                np_ = neg_pair_ - self.margin < torch.max(dist_pos)
                neg_pair = neg_pair_[np_]
            else:
                np_ = torch.argsort(neg_pair_)
                neg_pair = neg_pair_[np_[:100]]

            if len(neg_pair) < 1:
                continue

            # --- CHANGE FROM THE HARD VARIANT: sum the exp-margin term over
            # ALL K prototypes of the anchor's own class, instead of
            # selecting only the closest one.
            if self.type == 'log' or self.type == 'sqrt':
                fp = 1. + torch.sum(torch.exp(1. / self.pos_b * (dist_pos - self.pos_a)))
                fn = 1. + torch.sum(torch.exp(-1. / self.neg_b * (neg_pair - self.neg_a)))
                if self.type == 'log':
                    pos_loss = torch.log(fp)
                    neg_loss = torch.log(fn)
                else:
                    pos_loss = torch.sqrt(fp)
                    neg_loss = torch.sqrt(fn)
            else:
                pos_loss = 1. + self.loss_weight_p * torch.sum(torch.exp(1. / self.pos_b * (dist_pos - self.pos_a)))
                neg_loss = 1. + self.loss_weight_n * torch.sum(torch.exp(-1. / self.neg_b * (neg_pair - self.neg_a)))

            pos_neg_loss = sigma_
            loss.append((pos_loss + neg_loss + self.weight * pos_neg_loss))

        if len(loss) == 0:
            return torch.zeros(1, requires_grad=True, device=self.device)

        loss = sum(loss) / batch_size
        return loss
