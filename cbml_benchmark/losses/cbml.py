import torch
from torch import nn

from cbml_benchmark.losses.registry import LOSS


@LOSS.register('cbml_loss')
class CBMLLoss(nn.Module):
    """
    Step 2 of the ablation ladder, built on top of step 1
    (CBMLLossStep1Euclidean).

    SINGLE VARIABLE CHANGED FROM STEP 1: aggregation over mined pairs.

    Step 1 mines a set of "hard" positive and negative pairs (via the same
    margin-based filtering as the original CBML), then aggregates ALL of
    them softly via a LogSumExp-style sum inside pos_loss/neg_loss.

    This step keeps the mining step IDENTICAL to step 1 -- same pp/np_
    filtering conditions, same margin -- but instead of summing over every
    mined pair, it picks only the single hardest one from each mined set:

      - hardest positive = the mined positive with the LARGEST distance
        (farthest same-class sample -- hardest to pull together)
      - hardest negative = the mined negative with the SMALLEST distance
        (closest different-class sample -- hardest to push apart)

    and plugs only that single scalar into the exp/log formula, instead of
    a sum over the whole mined set. This isolates "soft aggregation over
    many hard pairs" vs. "hard selection of one pair" as its own variable,
    separate from the later step of switching from instances to prototypes.

    The sigma_ regularizer is left untouched (still computed over the full,
    unmined neg_pair_ set) since it's orthogonal to this change.

    Everything else (adaptive-negative mining, TYPE branching, sigma_
    regularizer, overall control flow) is unchanged from step 1.
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

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)

        dist_mat = torch.cdist(feats, feats, p=2) ** 2

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):

            pos_pair_ = dist_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ > epsilon]
            neg_pair_ = dist_mat[i][labels != labels[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            # sigma_ regularizer: unchanged, computed over the full,
            # unmined neg_pair_ set exactly as in step 1.
            mean_ = self.hyper_weight * torch.mean(pos_pair_) + (1 - self.hyper_weight) * torch.mean(neg_pair_)
            sigma_ = torch.mean(torch.sum(torch.pow(neg_pair_ - mean_, 2)))

            # Mining: identical to step 1.
            pp = pos_pair_ + self.margin > torch.min(neg_pair_)
            pos_pair = pos_pair_[pp]
            if self.adaptive_neg:
                np_ = neg_pair_ - self.margin < torch.max(pos_pair_)
                neg_pair = neg_pair_[np_]
            else:
                np_ = torch.argsort(neg_pair_)
                neg_pair = neg_pair_[np_[:100]]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # --- CHANGE FROM STEP 1: instead of summing over the whole
            # mined set, keep only the single hardest pair from each side.
            hardest_pos = torch.max(pos_pair)   # farthest same-class sample
            hardest_neg = torch.min(neg_pair)   # closest different-class sample

            if self.type == 'log' or self.type == 'sqrt':
                fp = 1. + torch.exp(1. / self.pos_b * (hardest_pos - self.pos_a))
                fn = 1. + torch.exp(-1. / self.neg_b * (hardest_neg - self.neg_a))
                if self.type == 'log':
                    pos_loss = torch.log(fp)
                    neg_loss = torch.log(fn)
                else:
                    pos_loss = torch.sqrt(fp)
                    neg_loss = torch.sqrt(fn)
            else:
                pos_loss = 1. + self.loss_weight_p * torch.exp(1. / self.pos_b * (hardest_pos - self.pos_a))
                neg_loss = 1. + self.loss_weight_n * torch.exp(-1. / self.neg_b * (hardest_neg - self.neg_a))

            pos_neg_loss = sigma_
            loss.append((pos_loss + neg_loss + self.weight * pos_neg_loss))

        if len(loss) == 0:
            return torch.zeros(1, requires_grad=True).cuda()

        loss = sum(loss) / batch_size
        return loss
