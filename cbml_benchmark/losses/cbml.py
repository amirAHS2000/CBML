import torch
from torch import nn

from cbml_benchmark.losses.registry import LOSS


@LOSS.register('cbml_loss')
class CBMLLoss(nn.Module):
    """
    Step 1 of the ablation ladder: identical to the original CBMLLoss, except
    the pairwise inner-product similarity matrix is replaced by a pairwise
    SQUARED EUCLIDEAN DISTANCE matrix.

    IMPORTANT: this is not a pure drop-in numerical equivalent of the
    original. For unit-normalized embeddings, similarity and squared
    distance are related by dist = 2 - 2*sim, which is a monotonic
    (order-preserving) transform -- but the loss below does NOT just
    substitute that formula into the old code unchanged, because every
    "high = good" comparison in similarity space becomes a "low = good"
    comparison in distance space. Concretely, three things flip:

      1. Hard-positive-mining condition:
         original: pos_pair - margin < max(neg_pair)   (similarity)
         here:     pos_pair + margin > min(neg_pair)   (distance)

      2. Hard-negative selection when NOT adaptive:
         original: take the 100 LARGEST similarities
         here:     take the 100 SMALLEST distances

      3. Loss exponent signs (pos_a/neg_a act as distance thresholds now,
         not similarity thresholds):
         original pos: exp(-1/pos_b * (pos_pair - pos_a))
         here     pos: exp(+1/pos_b * (pos_pair - pos_a))
         original neg: exp(+1/neg_b * (neg_pair - neg_a))
         here     neg: exp(-1/neg_b * (neg_pair - neg_a))

    Everything else (the adaptive-negative branch, the sigma_ regularizer,
    the log/sqrt/plain loss-type branching, the overall control flow) is
    kept structurally identical to the original so this is a clean
    single-variable change for the ablation.

    NOTE ON HYPERPARAMETERS: pos_a, neg_a, margin, pos_b, neg_b were tuned
    for a similarity range of [-1, 1]. Squared Euclidean distance for
    unit-normalized vectors ranges over [0, 4], so these will very likely
    need retuning here rather than reused verbatim from the original config.
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

        # --- CHANGE FROM ORIGINAL: squared Euclidean distance instead of
        # inner-product similarity. torch.cdist gives (unsquared) Euclidean
        # distance directly, so we square it; this works regardless of
        # whether feats are exactly unit-normalized.
        dist_mat = torch.cdist(feats, feats, p=2) ** 2

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):

            pos_pair_ = dist_mat[i][labels == labels[i]]
            # CHANGED: self-distance is ~0, not ~1, so exclude near-zero
            # distances instead of near-1 similarities.
            pos_pair_ = pos_pair_[pos_pair_ > epsilon]
            neg_pair_ = dist_mat[i][labels != labels[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            mean_ = self.hyper_weight * torch.mean(pos_pair_) + (1 - self.hyper_weight) * torch.mean(neg_pair_)
            sigma_ = torch.mean(torch.sum(torch.pow(neg_pair_ - mean_, 2)))

            # --- CHANGE FROM ORIGINAL: hard-mining conditions flipped for
            # distance space (see class docstring, point 1).
            pp = pos_pair_ + self.margin > torch.min(neg_pair_)
            pos_pair = pos_pair_[pp]
            if self.adaptive_neg:
                np_ = neg_pair_ - self.margin < torch.max(pos_pair_)
                neg_pair = neg_pair_[np_]
            else:
                # CHANGED: hardest negatives are now the SMALLEST distances,
                # so sort ascending and take the first 100, not the last 100.
                np_ = torch.argsort(neg_pair_)
                neg_pair = neg_pair_[np_[:100]]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # --- CHANGE FROM ORIGINAL: exponent signs flipped for distance
            # space (see class docstring, point 3). pos_a/neg_a now act as
            # distance thresholds, pos_b/neg_b as distance-scale factors.
            if self.type == 'log' or self.type == 'sqrt':
                fp = 1. + torch.sum(torch.exp(1. / self.pos_b * (pos_pair - self.pos_a)))
                fn = 1. + torch.sum(torch.exp(-1. / self.neg_b * (neg_pair - self.neg_a)))
                if self.type == 'log':
                    pos_loss = torch.log(fp)
                    neg_loss = torch.log(fn)
                else:
                    pos_loss = torch.sqrt(fp)
                    neg_loss = torch.sqrt(fn)
            else:
                pos_loss = 1. + self.loss_weight_p * torch.sum(torch.exp(1. / self.pos_b * (pos_pair - self.pos_a)))
                neg_loss = 1. + self.loss_weight_n * torch.sum(torch.exp(-1. / self.neg_b * (neg_pair - self.neg_a)))

            pos_neg_loss = sigma_
            loss.append((pos_loss + neg_loss + self.weight * pos_neg_loss))

        if len(loss) == 0:
            return torch.zeros(1, requires_grad=True).cuda()

        loss = sum(loss) / batch_size
        return loss
