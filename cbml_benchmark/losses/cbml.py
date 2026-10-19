import torch
from torch import nn
import torch.nn.functional as F

from cbml_benchmark.losses.registry import LOSS


@LOSS.register('cbml_loss')
class CBMLLoss(nn.Module):
    """Prototype-based CBML main term + one switchable instance-level regularizer.

    LOSSES.CBML_LOSS.REG_TYPE
        'mvc'        : the paper's two-sided MVC (unchanged from your last run, xi NOT detached)
        'ratio'      : scale-invariant  Var(neg) / ((mean_pos - mean_neg)^2 + 1e-4)   per anchor
        'inst_hinge' : relu(hardest in-batch negative - xi), xi = alpha*EMA(pos) + (1-alpha)*EMA(hardest neg),
                       the EMAs are global buffers (no gradient)
    LOSSES.CBML_LOSS.WEIGHT multiplies the regularizer in all three cases.

    LOSSES.CBML_LOSS.NEG_MODE
        'proto' : negative term on the closest wrong-class prototype (default, as before)
        'inst'  : hybrid. Positive term on the closest OWN-class prototype only; negative term on the hard
                  in-batch INSTANCE negatives selected with the paper's rule, using the prototype positive as
                  the reference (m_ik + MARGIN > s+). Prototypes then receive gradient from own-class anchors
                  only. The instance negatives use INST_NEG_A / INST_NEG_B (paper: 1.0 / 0.01).
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

        self.reg_type = getattr(cfg.LOSSES.CBML_LOSS, 'REG_TYPE', 'mvc')
        assert self.reg_type in ('mvc', 'ratio', 'inst_hinge'), f"unknown REG_TYPE {self.reg_type}"
        self.reg_alpha = getattr(cfg.LOSSES.CBML_LOSS, 'REG_ALPHA', 0.9)
        self.reg_nu = getattr(cfg.LOSSES.CBML_LOSS, 'REG_NU', 0.99)

        self.neg_mode = getattr(cfg.LOSSES.CBML_LOSS, 'NEG_MODE', 'proto')
        assert self.neg_mode in ('proto', 'inst'), f"unknown NEG_MODE {self.neg_mode}"
        self.inst_neg_a = getattr(cfg.LOSSES.CBML_LOSS, 'INST_NEG_A', 1.0)
        self.inst_neg_b = getattr(cfg.LOSSES.CBML_LOSS, 'INST_NEG_B', 0.01)
        if self.neg_mode == 'inst':
            assert self.type in ('log', 'sqrt'), "NEG_MODE inst requires TYPE log or sqrt"

        self.device_name = getattr(cfg.MODEL, 'DEVICE', 'cuda')
        self.device = torch.device(self.device_name)
        self.embed_dim = getattr(cfg.MODEL.HEAD, 'DIM', 512)
        self.num_classes = getattr(cfg.LOSSES.CBML_LOSS, 'N_CLASSES', 100)

        # prototypes [C, K, D]
        self.prototypes_per_class = getattr(cfg.LOSSES.CBML_LOSS, 'PROTOTYPE_PER_CLASS', 3)
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototypes_per_class, self.embed_dim, device=self.device)
        )

        # logging (detached): batch-mean regularizer value, fraction of anchors with a non-zero value, xi
        self.last_mvc = None
        self.last_reg_active = None
        self.last_xi = None

        # global EMAs for 'inst_hinge' (buffers: never receive gradients)
        self.register_buffer('ema_pos', torch.zeros((), device=self.device))
        self.register_buffer('ema_neg', torch.zeros((), device=self.device))
        self.register_buffer('ema_ready', torch.zeros((), dtype=torch.bool, device=self.device))

        self.register_buffer('pos_proto_counts', torch.zeros(self.num_classes, self.prototypes_per_class, dtype=torch.long))
        self.register_buffer('neg_proto_counts', torch.zeros(self.num_classes, self.prototypes_per_class, dtype=torch.long))

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
        reg_vals = []

        # ---- batch levels + global threshold for 'inst_hinge' (no gradient) ----
        b_pos = b_neg = xi_glob = None
        if self.reg_type == 'inst_hinge':
            with torch.no_grad():
                same = labels.unsqueeze(0) == labels.unsqueeze(1)
                eye = torch.eye(batch_size, dtype=torch.bool, device=feats.device)
                pos_m = same & ~eye
                neg_m = ~same
                valid = pos_m.any(1) & neg_m.any(1)
                if bool(valid.any()):
                    pos_level = (sim_mat * pos_m).sum(1)[valid] / pos_m.sum(1)[valid]
                    neg_level = sim_mat.masked_fill(~neg_m, float('-inf')).max(1).values[valid]
                    b_pos, b_neg = pos_level.mean(), neg_level.mean()
                    if not bool(self.ema_ready):
                        self.ema_pos.copy_(b_pos)
                        self.ema_neg.copy_(b_neg)
                        self.ema_ready.fill_(True)
                    xi_glob = self.reg_alpha * self.ema_pos + (1 - self.reg_alpha) * self.ema_neg

        for i in range(batch_size):

            positive_class = labels[i].item()
            pos_sim = feat_proto_sim_mat[i, positive_class, :]
            best_pos_proto_idx = torch.argmax(pos_sim).item()
            best_pos_proto = P_3d[positive_class, best_pos_proto_idx]

            self.pos_proto_counts[positive_class, best_pos_proto_idx] += 1

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
            best_neg_proto = P_3d[best_neg_class_idx, best_neg_proto_idx].detach()

            self.neg_proto_counts[best_neg_class_idx, best_neg_proto_idx] += 1

            # ------------------ regularizer (instance-to-instance) ------------------
            pos_inst = sim_mat[i][labels == labels[i]]
            pos_inst = pos_inst[pos_inst < 1 - epsilon]          # drop the self-pair
            neg_inst = sim_mat[i][labels != labels[i]]
            reg_i = feats.new_zeros(())
            if len(pos_inst) >= 1 and len(neg_inst) >= 1:
                if self.reg_type == 'mvc':
                    xi = self.hyper_weight * pos_inst.mean() + (1 - self.hyper_weight) * neg_inst.mean()
                    reg_i = (neg_inst - xi).pow(2).mean()
                elif self.reg_type == 'ratio':
                    mu_p, mu_n = pos_inst.mean(), neg_inst.mean()
                    reg_i = (neg_inst - mu_n).pow(2).mean() / ((mu_p - mu_n).pow(2) + 1e-4)
                elif self.reg_type == 'inst_hinge' and xi_glob is not None:
                    reg_i = F.relu(neg_inst.max() - xi_glob)
            reg_vals.append(reg_i.detach())
            # -------------------------------------------------------------------------

            s_pos = feats[i] @ best_pos_proto
            if self.neg_mode == 'inst':
                # hard instance negatives (paper's rule, prototype positive as reference); no prototype gradient
                hard_neg = neg_inst[neg_inst + self.margin > s_pos.detach()]
                fn_inst = 1. + torch.sum(torch.exp((hard_neg - self.inst_neg_a) / self.inst_neg_b))

            if self.type == 'log' or self.type == 'sqrt':
                fp = 1. + torch.exp(-1. / self.pos_b * (s_pos - self.pos_a))
                if self.neg_mode == 'inst':
                    fn = fn_inst
                else:
                    fn = 1. + torch.exp(1. / self.neg_b * ((feats[i] @ best_neg_proto) - self.neg_a))
                if self.type == 'log':
                    pos_loss = torch.log(fp)
                    neg_loss = torch.log(fn)
                else:
                    pos_loss = torch.sqrt(fp)
                    neg_loss = torch.sqrt(fn)
            else:
                pos_loss = 1. + self.loss_weight_p * torch.exp(-1. / self.pos_b * (s_pos - self.pos_a))
                neg_loss = 1. + self.loss_weight_n * torch.exp(1. / self.neg_b * ((feats[i] @ best_neg_proto) - self.neg_a))
            loss.append((pos_loss + neg_loss + self.weight * reg_i))

        # EMA update after the batch (no gradient)
        if self.reg_type == 'inst_hinge' and b_pos is not None:
            with torch.no_grad():
                self.ema_pos.mul_(self.reg_nu).add_((1 - self.reg_nu) * b_pos)
                self.ema_neg.mul_(self.reg_nu).add_((1 - self.reg_nu) * b_neg)
                self.last_xi = float(xi_glob)

        if reg_vals:
            rv = torch.stack(reg_vals)
            self.last_mvc = rv.mean()
            self.last_reg_active = (rv > 0).float().mean()

        if len(loss) == 0:
            return torch.zeros(1, requires_grad=True).cuda()

        loss = sum(loss) / batch_size
        return loss
