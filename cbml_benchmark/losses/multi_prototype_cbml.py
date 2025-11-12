import torch
import torch.nn as nn
import torch.nn.functional as F

from cbml_benchmark.losses.registry import LOSS
from cbml_benchmark.utils.prototype_weight_monitor import compute_proto_stats, compute_weight_stats

@LOSS.register('multi_prototype_cbml')
class MultiPrototypeCBMLLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiPrototypeCBMLLoss, self).__init__()
        self.num_classes = cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES
        self.prototype_per_class = cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS
        self.embed_dim = cfg.MODEL.HEAD.DIM
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.gamma = cfg.LOSSES.MULTI_PROTOTYPE_CBML.HYPER_WEIGHT
        self.lambda_mvc = cfg.LOSSES.MULTI_PROTOTYPE_CBML.REG_WEIGHT

        self.n_negatives = cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_NEGATIVES  # e.g., 2-3

        # self.proto_anchor_weight = cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTO_ANCHOR_WEIGHT
        # self.weight_entropy_var = cfg.LOSSES.MULTI_PROTOTYPE_CBML.WEIGHT_ENTROPY_VAR

        # Theta learnable (init to config value or 2.1)
        init_theta = cfg.LOSSES.MULTI_PROTOTYPE_CBML.INIT_THETA if hasattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'INIT_THETA') else 2.1
        self.theta = nn.Parameter(torch.tensor(init_theta, device=self.device))

        # Prototypes: [num_classes, prototype_per_class, embed_dim]
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

        # Weights: [num_classes, prototype_per_class], initialized based on cluster sizes
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class, device=self.device) / self.prototype_per_class
        )

        # Class priors: [num_classes]
        self.class_priors = nn.Parameter(
            torch.tensor(cfg.LOSSES.MULTI_PROTOTYPE_CBML.CLASS_PRIORS, device=self.device),
            requires_grad=False
        )

    def set_prototypes_and_weights(self, prototypes, cluster_sizes):
        """Set prototypes and initialize weights based on k-means cluster sizes."""
        with torch.no_grad():
            if prototypes.device != self.device:
                prototypes = prototypes.to(self.device)
            self.prototypes.data = prototypes

            # Save a frozen copy of the initial prototypes for monitoring
            self.initial_prototypes = prototypes.detach().clone().to(self.device)

            # Initialize weights based on cluster sizes (normalized per class)
            if cluster_sizes is not None and cluster_sizes.shape == (self.num_classes, self.prototype_per_class):
                normalized_weights = cluster_sizes.float() / torch.sum(cluster_sizes, dim=1, keepdim=True)
                self.weights.data = normalized_weights.to(self.device)
            else:
                # Fallback to uniform if cluster_sizes are invalid
                self.weights.data = torch.ones_like(self.weights) / self.prototype_per_class

            # Optionally also save the initial weights (for future analysis)
            # self.initial_weights = self.weights.detach().clone().to(self.device)

            torch.cuda.empty_cache()

    def show_theta(self):
        return self.theta.item()
    
    def show_prototype_stats(self, initial_prototypes=None):
        if initial_prototypes is None and hasattr(self, "initial_prototypes"):
            initial_prototypes = self.initial_prototypes
        return compute_proto_stats(self.prototypes.detach(), initial_prototypes)
        
    def show_weight_stats(self):
        return compute_weight_stats(self.weights.detach())

    def forward(self, embeddings, targets):
        # Device consistency
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        if targets.device != self.device:
            targets = targets.to(self.device)

        batch_size = embeddings.size(0)
        B, C, K, D = batch_size, self.num_classes, self.prototype_per_class, self.embed_dim

        # Normalize embeddings & prototypes
        normalized_embds = F.normalize(embeddings, p=2, dim=1)  # [B, D]
        normalized_protos = F.normalize(self.prototypes, p=2, dim=2)  # [C, K, D]

        # Normalize weights to be positive and sum to 1 for each class using softmax
        normalized_weights = F.softmax(self.weights, dim=1)  # [C, K]

        # Prototype-embedding similarities: [B, C, K]
        proto_embd_sim = torch.matmul(normalized_embds, normalized_protos.view(-1, D).t())
        proto_embd_sim = proto_embd_sim.view(B, C, K)

        beta = torch.exp(self.theta)  # Learnable beta

        # Precompute per-sample work
        idx = torch.arange(B, device=self.device)
        pos_proto_sims_all = proto_embd_sim[idx, targets]  # [B, K]
        pos_best_idx_all = pos_proto_sims_all.argmax(dim=-1)  # [B]

        total_mpcbml_loss = 0.0
        total_mvc_loss = 0.0

        for i in range(B):
            y = int(targets[i].item())  # class index

            # Positive prototype info
            pos_sims = pos_proto_sims_all[i]  # [K]
            best_pos_idx = int(pos_best_idx_all[i].item())
            pos_sim = pos_sims[best_pos_idx]
            w_pos = normalized_weights[y, best_pos_idx]
            prior_pos = self.class_priors[y]

            # ---------- Select top N dominant negative prototypes ----------
            neg_mask = torch.ones(C, dtype=torch.bool, device=self.device)
            neg_mask[y] = False
            neg_classes = torch.arange(C, device=self.device)[neg_mask]  # [C-1]

            # Find the hardest prototype (max sim) for EVERY negative class
            max_sim_per_class = torch.max(proto_embd_sim[i], dim=1)[0]  # [C]
            best_idx_per_class = torch.argmax(proto_embd_sim[i], dim=1)  # [C]

            # Extract info for the hardest prototype in each negative class
            neg_class_sims = max_sim_per_class[neg_mask]  # [C-1]
            neg_class_weights = normalized_weights[neg_classes, best_idx_per_class[neg_classes]]  # [C-1]
            neg_class_priors = self.class_priors[neg_classes]  # [C-1]

            # Calculate contribution for the HARDEST prototype of each negative class
            neg_class_contribution = beta * neg_class_sims + torch.log(neg_class_priors * neg_class_weights + 1e-9)

            top_n_indices = torch.topk(neg_class_contribution, k=self.n_negatives, dim=0, sorted=False)[1]  # [N]

            top_n_neg_sims = neg_class_sims[top_n_indices]  # [N]
            top_n_neg_weights = neg_class_weights[top_n_indices]  # [N]
            top_n_neg_priors = neg_class_priors[top_n_indices]  # [N]

            # ---------- Loss calculation (for N negatives) ----------
            neg_exp_sum = torch.sum(torch.exp(beta * top_n_neg_sims))
            sim_term = beta * pos_sim - torch.log(neg_exp_sum + 1e-9)

            avg_neg_prior_w = torch.mean(top_n_neg_priors * top_n_neg_weights)
            eps = 1e-9
            bias_term = torch.log(prior_pos * w_pos + eps) - torch.log(avg_neg_prior_w + eps)

            total_mpcbml_loss += (sim_term + bias_term)

            # ----------------------- Regularization term (MVC loss) ------------------------------------
            pos_sims_i = proto_embd_sim[i, y]  # [K]
            pos_weights_i = normalized_weights[y]  # [K]
            
            weighted_pos_sim_sum = torch.sum(pos_sims_i * pos_weights_i)
            weighted_mean_pos = weighted_pos_sim_sum 
            
            neg_mask = torch.ones(C, dtype=torch.bool, device=self.device)
            neg_mask[y] = False
            
            neg_sims_flat = proto_embd_sim[i][neg_mask].reshape(-1)
            neg_weights_flat = normalized_weights[neg_mask].reshape(-1)
            
            weighted_neg_sim_sum = torch.sum(neg_sims_flat * neg_weights_flat)
            weighted_mean_neg = weighted_neg_sim_sum / (C - 1.0)
            
            xi_w_i = self.gamma * weighted_mean_pos + (1.0 - self.gamma) * weighted_mean_neg
            
            sq_diff = (neg_sims_flat - xi_w_i)**2
            weighted_sq_diff_sum = torch.sum(sq_diff * neg_weights_flat)
            
            mvc_loss_i = weighted_sq_diff_sum / (C - 1.0 + 1e-9)
            
            total_mvc_loss += mvc_loss_i

        # --- Final Loss Combination ---
        mpcbml_loss = - total_mpcbml_loss / B
        avg_mvc_loss = total_mvc_loss / B

        # ======================= Regularization Terms ============================
        # proto_anchor_reg = 0.0
        # weight_entropy_reg = 0.0

        # # --- (1) Prototype Anchor Regularization ---
        # if hasattr(self, "initial_prototypes") and self.proto_anchor_weight > 0:
        #     proto_anchor_reg = torch.mean((self.prototypes - self.initial_prototypes) ** 2)
        #     proto_anchor_reg = self.proto_anchor_weight * proto_anchor_reg

        # # --- (2) Weight Entropy Variance Regularization ---
        # if self.weight_entropy_var > 0:
        #     normalized_weights = F.softmax(self.weights, dim=1)
        #     entropy_per_class = -torch.sum(
        #         normalized_weights * (normalized_weights.clamp_min(1e-9)).log(), dim=1
        #     )
        #     entropy_var = torch.var(entropy_per_class, unbiased=False)
        #     weight_entropy_reg = self.weight_entropy_var * entropy_var

        # ======================= Total Loss ============================
        loss = (
            mpcbml_loss
            + self.lambda_mvc * avg_mvc_loss
            # + proto_anchor_reg
            # + weight_entropy_reg
        )

        return loss