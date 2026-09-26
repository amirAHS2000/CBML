import torch
from torch import nn
import torch.nn.functional as F

from cbml_benchmark.losses.registry import LOSS


@LOSS.register('cbml_loss')
class CBMLLoss(nn.Module):
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

        # prototypes [C, K, D]
        self.prototypes_per_class = getattr(cfg.LOSSES.CBML_LOSS, 'PROTOTYPE_PER_CLASS', 3)
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototypes_per_class, self.embed_dim, device=self.device)
        )

    @torch.no_grad()
    def set_prototypes(self, prototypes):
        prototypes = prototypes.to(self.device)
        self.prototypes.copy_(prototypes)
        torch.cuda.empty_cache()

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        if feats.device != self.prototypes.device:
            feats = feats.to(self.device)
            labels = labels.to(self.device)

        batch_size = feats.size(0)
        C = self.num_classes
        K = self.prototypes_per_class
        P_3d = F.normalize(self.prototypes, p=2, dim=-1)   # [C, K, D] -- used for indexing
        P_flat = P_3d.view(C * K, -1)                      # [C*K, D]  -- used for the matmul only

        sim_mat = torch.matmul(feats, torch.t(feats))
        feat_proto_sim_mat = torch.matmul(feats, torch.t(P_flat))       # [B, C*K]
        feat_proto_sim_mat = feat_proto_sim_mat.view(batch_size, C, K)  # [B, C, K]
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):

            positive_class = labels[i].item()
            pos_sim = feat_proto_sim_mat[i, positive_class, :]
            best_pos_proto_idx = torch.argmax(pos_sim).item()
            best_pos_proto = P_3d[positive_class, best_pos_proto_idx]

            # negative selection
            neg_mask = torch.arange(C, device=self.device) != positive_class
            neg_class_indices = torch.where(neg_mask)[0]  # absolute class indices
            neg_sim = feat_proto_sim_mat[i, neg_mask, :]

            flat_max_idx = torch.argmax(neg_sim)
            best_neg_class_idx_masked = flat_max_idx // K
            best_neg_proto_idx = flat_max_idx % K

            # map back to absolute class index
            best_neg_class_idx = neg_class_indices[best_neg_class_idx_masked].item()
            best_neg_proto_idx = best_neg_proto_idx.item()
            best_neg_proto = P_3d[best_neg_class_idx, best_neg_proto_idx]

            # ------------------------ MVC term ------------------------------
            # Unchanged from the original CBML loss -- still computed on real
            # instance-to-instance similarities, not prototypes.
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            mean_ = self.hyper_weight * torch.mean(pos_pair_) + (1 - self.hyper_weight) * torch.mean(neg_pair_)
            sigma_ = torch.mean(torch.sum(torch.pow(neg_pair_ - mean_, 2)))
            # ----------------------------------------------------------------

            if self.type == 'log' or self.type == 'sqrt':
                fp = 1. + torch.exp(-1. / self.pos_b * ((feats[i] @ best_pos_proto) - self.pos_a))
                fn = 1. + torch.exp(1. / self.neg_b * ((feats[i] @ best_neg_proto) - self.neg_a))
                if self.type == 'log':
                    pos_loss = torch.log(fp)
                    neg_loss = torch.log(fn)
                else:
                    pos_loss = torch.sqrt(fp)
                    neg_loss = torch.sqrt(fn)
            else:
                pos_loss = 1. + self.loss_weight_p * torch.exp(-1. / self.pos_b * ((feats[i] @ best_pos_proto) - self.pos_a))
                neg_loss = 1. + self.loss_weight_n * torch.exp(1. / self.neg_b * ((feats[i] @ best_neg_proto) - self.neg_a))
            pos_neg_loss = sigma_
            loss.append((pos_loss + neg_loss + self.weight * pos_neg_loss))

        if len(loss) == 0:
            return torch.zeros(1, requires_grad=True).cuda()

        loss = sum(loss) / batch_size
        return loss
