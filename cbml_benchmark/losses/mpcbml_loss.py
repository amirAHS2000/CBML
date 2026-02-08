import torch
import torch.nn as nn
import torch.nn.functional as F
from cbml_benchmark.losses.registry import LOSS
# from cbml_benchmark.utils.mpcbml_logger import compute_statistics


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
        self.reg_weight = getattr(cfg.LOSSES.MPCBML_LOSS, 'LAMBDA_REG', 1.0)

    @torch.no_grad()
    def set_prototypes_and_weights(self, prototypes, cluster_sizes):
        prototypes = prototypes.to(self.device)
        # self.prototypes.copy_(prototypes)
        self.prototypes.copy_(F.normalize(prototypes, p=2, dim=2))

        # Save a frozen copy of the initial prototypes for monitoring
        self.initial_prototypes = F.normalize(prototypes.detach(), p=2, dim=2)

        if cluster_sizes is not None and \
            cluster_sizes.shape == (self.num_classes, self.prototype_per_class):
            cluster_sizes = cluster_sizes.to(self.device).float()
            normalized_weights = cluster_sizes / (cluster_sizes.sum(dim=1, keepdim=True) + 1e-9)
            self.weights.copy_(normalized_weights)
        else:
            # Fallback to uniform
            self.weights.fill_(1.0 / self.prototype_per_class)
           
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
        if embeddings.device != self.prototypes.device:
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)

        # sim_mat = torch.matmul(embeddings, torch.t(embeddings))
        # epsilon = 1e-5
        # reg_term = list()
        # for i in range(embeddings.size(0)):
        #     pos_pair_ = sim_mat[i][targets == targets[i]]
        #     pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
        #     neg_pair_ = sim_mat[i][targets != targets[i]]

        #     if len(neg_pair_) < 1 or len(pos_pair_) < 1:
        #         continue

        #     mean_ = self.hyper_weight * torch.mean(pos_pair_) + (1 - self.hyper_weight) * torch.mean(neg_pair_)
        #     sigma_ = torch.mean(torch.sum(torch.pow(neg_pair_ - mean_, 2)))
        #     reg_term.append(self.reg_weight * sigma_)
        # reg_loss = sum(reg_term) / embeddings.size(0)

        P = self.prototypes
        P = F.normalize(P, p=2, dim=-1) # [C, K, D]
        z = F.normalize(embeddings, p=2, dim=-1) # [B, D]
        W = self.weights # [C, K]

        B = z.shape[0] # batch size
        C = self.num_classes
        K = self.prototype_per_class
        beta = torch.exp(self.theta)
        eps = 1e-9

        flat_protos = P.view(C * K, -1) # [C*K, D]
        sims = torch.matmul(z, flat_protos.t()).view(B, C, K) # [B, C, K]
        weighted_sims = sims * W.unsqueeze(0) # [B, C, K]
        # weighted_sims = sims * W.unsqueeze(0) / 0.1

        target_mask = F.one_hot(targets, num_classes=C).bool() # [B, C]

        # selecting best positive
        pos_weighted = weighted_sims[target_mask].view(B, K)
        pos_raw = sims[target_mask].view(B, K)
        pos_w = W[targets]
        prior_pos = self.class_priors[targets]

        best_pos_weighted_val, best_pos_idx = pos_weighted.max(dim=-1) # [B]
        best_pos_val = pos_raw[torch.arange(B), best_pos_idx] # [B]
        best_pos_w = pos_w[torch.arange(B), best_pos_idx] # [B]

        # selecting best negative
        neg_weighted = weighted_sims[~target_mask].view(B, C-1, K)
        neg_raw = sims[~target_mask].view(B, C-1, K)
        W_expanded = W.unsqueeze(0).expand(B, C, K)
        neg_w = W_expanded[~target_mask].view(B, C-1, K)

        class_priors_expanded = self.class_priors.unsqueeze(0).expand(B, C)
        neg_priors = class_priors_expanded[~target_mask].view(B, C-1)

        # Best prototype of each negative class
        neg_weighted_max, neg_best_k = neg_weighted.max(dim=-1)  # [B,C-1]

        # Best negative class
        best_neg_class = neg_weighted_max.argmax(dim=-1)  # [B]
        b_idx = torch.arange(B, device=self.device)

        # Extract selected negative prototype info
        best_neg_val = neg_raw[b_idx, best_neg_class, neg_best_k[b_idx, best_neg_class]]    # [B]
        best_neg_w = neg_w[b_idx, best_neg_class, neg_best_k[b_idx, best_neg_class]]        # [B]
        prior_neg = neg_priors[b_idx, best_neg_class]

        main_loss = F.softplus(
            torch.log(prior_neg / prior_pos) +
            torch.log(best_neg_w / best_pos_w) +
            (beta * (best_neg_val - best_pos_val))
        ).mean()

        # loss = main_loss + reg_loss
        loss = main_loss

        return loss
