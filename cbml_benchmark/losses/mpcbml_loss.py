import torch
import torch.nn as nn
import torch.nn.functional as F
from cbml_benchmark.losses.registry import LOSS


@LOSS.register('mpcbml_loss')
class MpcbmlLoss(nn.Module):
    def __init__(self, cfg):
        super(MpcbmlLoss, self).__init__()

        self.device_name = getattr(cfg.MODEL, 'DEVICE', 'cuda')
        self.device = torch.device(self.device_name)
        self.embed_dim = getattr(cfg.MODEL.HEAD, 'DIM', 512)
        self.num_classes = getattr(cfg.LOSSES.MPCBML_LOSS, 'N_CLASSES', 100)

        theta_is_learnable = getattr(cfg.LOSSES.MPCBML_LOSS, 'THETA_IS_LEARNABLE', False)
        init_theta = getattr(cfg.LOSSES.MPCBML_LOSS, 'INIT_THETA', 1.0)
        if theta_is_learnable:
            self.theta = nn.Parameter(
                torch.tensor(init_theta, device=self.device),
                requires_grad=True
            )
        else:
            self.theta = nn.Parameter(
                torch.tensor(init_theta, device=self.device),
                requires_grad=False
            )

        self.prototype_per_class = getattr(cfg.LOSSES.MPCBML_LOSS, 'PROTOTYPE_PER_CLASS', 3)
        # Prototypes [C, K, D]
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

        # Mixture weights [C, K]
        # Be updated with Langrange Multiplier
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class, device=self.device)
            / self.prototype_per_class,
            requires_grad=True
        )

        # Class priors [C]
        priors_list = getattr(cfg.LOSSES.MPCBML_LOSS, 'CLASS_PRIORS', [1.0/self.num_classes]*self.num_classes)
        self.register_buffer('class_priors',
            torch.tensor(priors_list, device=self.device)
        )

        self.hyper_weight = getattr(cfg.LOSSES.MPCBML_LOSS, 'GAMMA_REG', 0.2)
        self.reg_weight = getattr(cfg.LOSSES.MPCBML_LOSS, 'LAMBDA_REG', 0.3)

    @torch.no_grad()
    def set_prototypes_and_weights(self, prototypes, cluster_sizes):
        prototypes = prototypes.to(self.device)
        self.prototypes.copy_(prototypes)
        self.weights.fill_(1.0 / self.prototype_per_class)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def constrained_weight_update(self):
        if self.weights.grad is None:
            return

        # Get gradients [C, K]
        grad_w = self.weights.grad

        # Compute mean gradient per class
        mean_grad = grad_w.mean(dim=1, keepdim=True)  # [C, 1]

        grad_w.sub_(mean_grad)

    def forward(self, embeddings, targets):

        assert embeddings.size(0) == targets.size(0), \
            f"feats.size(0): {embeddings.size(0)} is not equal to labels.size(0): {targets.size(0)}"

        if embeddings.device != self.prototypes.device:
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)

        B = embeddings.size(0)
        if B == 0:
            return torch.zeros(1, requires_grad=True, device=self.device)

        # ------------- Regularization term (vectorized, same logic as original CBML) -------------
        epsilon = 1e-5
        sim_mat = embeddings @ embeddings.t()                              # [B, B]
        same = targets.unsqueeze(0) == targets.unsqueeze(1)                # [B, B]
        pos_mask = same & (sim_mat < 1 - epsilon)                          # positives, self / duplicates removed
        neg_mask_b = ~same                                                 # negatives

        pos_cnt = pos_mask.sum(dim=1)                                      # [B]
        neg_cnt = neg_mask_b.sum(dim=1)                                    # [B]

        pos_f = pos_mask.to(sim_mat.dtype)
        neg_f = neg_mask_b.to(sim_mat.dtype)

        pos_mean = (sim_mat * pos_f).sum(dim=1) / pos_cnt.clamp(min=1)
        neg_mean = (sim_mat * neg_f).sum(dim=1) / neg_cnt.clamp(min=1)
        mean_ = self.hyper_weight * pos_mean + (1 - self.hyper_weight) * neg_mean   # [B]

        sigma_ = (((sim_mat - mean_.unsqueeze(1)) ** 2) * neg_f).sum(dim=1)         # [B]
        valid = ((pos_cnt > 0) & (neg_cnt > 0)).to(sim_mat.dtype)                   # skip rows like the original `continue`
        reg_loss = (self.reg_weight * sigma_ * valid).sum() / B
        # -------------------------------------------------------------------------------------------

        z = embeddings                       # [B, D]
        P = self.prototypes                  # [C, K, D]
        W = self.weights                     # [C, K]
        C = self.num_classes
        K = self.prototype_per_class
        beta = torch.exp(self.theta)

        # logits [B, C, K]
        flat_protos = P.view(C * K, -1)
        logits = z @ flat_protos.T - 0.5 * (flat_protos.norm(p=2, dim=1) ** 2)
        logits = logits.view(B, C, K)

        log_W = torch.log(W)                          # [C, K]
        log_priors = torch.log(self.class_priors)     # [C]
        arange_B = torch.arange(B, device=logits.device)

        # ---------------- positive selection ----------------
        pos_logits = logits[arange_B, targets]                        # [B, K]
        pos_score = log_W[targets] + pos_logits                       # [B, K]
        best_pos_idx = torch.argmax(pos_score, dim=1)                 # [B]
        best_pos_logit = pos_logits[arange_B, best_pos_idx]           # [B]
        best_pos_logW = log_W[targets, best_pos_idx]                  # [B]
        pos_log_prior = log_priors[targets]                           # [B]

        # ---------------- negative selection ----------------
        neg_score = log_priors.view(1, C, 1) + log_W.unsqueeze(0) + logits   # [B, C, K]
        class_ids = torch.arange(C, device=logits.device).unsqueeze(0)       # [1, C]
        target_mask = (class_ids == targets.unsqueeze(1)).unsqueeze(2)       # [B, C, 1]
        neg_score = neg_score.masked_fill(target_mask, float('-inf'))

        flat_max_idx = torch.argmax(neg_score.view(B, C * K), dim=1)         # [B]
        best_neg_class = flat_max_idx // K                                    # [B]
        best_neg_proto = flat_max_idx % K                                     # [B]

        best_neg_logit = logits[arange_B, best_neg_class, best_neg_proto]    # [B]
        best_neg_logW = log_W[best_neg_class, best_neg_proto]                # [B]
        neg_log_prior = log_priors[best_neg_class]                           # [B]

        # ---------------- loss ----------------
        # logit_neg - logit_pos == z.pn - 0.5|pn|^2 - z.pp + 0.5|pp|^2
        current_loss = F.softplus(
            (neg_log_prior - pos_log_prior) +
            (best_neg_logW - best_pos_logW) +
            beta * (best_neg_logit - best_pos_logit)
        )

        main_loss = current_loss.sum() / B
        loss = main_loss + reg_loss

        return loss
