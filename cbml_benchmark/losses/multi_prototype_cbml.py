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

        # Theta learnable (init to config value or 2.1)
        init_theta = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'INIT_THETA', 2.1)
        self.theta = nn.Parameter(torch.tensor(init_theta, device=self.device))

        # Prototypes: [num_classes, prototype_per_class, embed_dim]
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

        # Weights: [num_classes, prototype_per_class], EM-updated (no gradients)
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class, device=self.device)
            / self.prototype_per_class,
            requires_grad=False   # IMPORTANT for EM updates
        )

        # Class priors: [num_classes]
        self.class_priors = nn.Parameter(
            torch.tensor(cfg.LOSSES.MULTI_PROTOTYPE_CBML.CLASS_PRIORS, device=self.device),
            requires_grad=False
        )

        # For logging MVC-related stats
        self.current_mvc_value = 0.0
        self.current_positive_mean = 0.0
        self.current_negative_mean = 0.0
        self.current_xi = 0.0

    @torch.no_grad()
    def set_prototypes_and_weights(self, prototypes, cluster_sizes):
        """Set prototypes and initialize weights based on k-means cluster sizes."""
        # Prototypes
        prototypes = prototypes.to(self.device)
        self.prototypes.copy_(prototypes)

        # Save a frozen copy of the initial prototypes for monitoring
        self.initial_prototypes = prototypes.detach().clone()

        # Initialize weights based on cluster sizes (normalized per class)
        if (
            cluster_sizes is not None
            and cluster_sizes.shape == (self.num_classes, self.prototype_per_class)
        ):
            cluster_sizes = cluster_sizes.to(self.device).float()
            normalized_weights = cluster_sizes / (cluster_sizes.sum(dim=1, keepdim=True) + 1e-9)
            self.weights.copy_(normalized_weights)
        else:
            # Fallback to uniform if cluster_sizes are invalid
            self.weights.fill_(1.0 / self.prototype_per_class)

        torch.cuda.empty_cache()

    def show_theta(self):
        return self.theta.item()
    
    def show_prototype_stats(self, initial_prototypes=None):
        if initial_prototypes is None and hasattr(self, "initial_prototypes"):
            initial_prototypes = self.initial_prototypes
        return compute_proto_stats(self.prototypes.detach(), initial_prototypes)
        
    def show_weight_stats(self):
        return compute_weight_stats(self.weights.detach())
    
    def show_mvc_value(self):
        return getattr(self, 'current_mvc_value', None)

    @torch.no_grad()
    def em_update_weights(self, model, data_loader):
        """
        Re-estimate mixture weights using EM-style responsibilities.
        model: embedding model
        data_loader: full train set (no augmentation)
        """
        print("\n[EM] Updating prototype weights ...")

        C = self.num_classes
        K = self.prototype_per_class

        # responsibility sums per class
        weight_accum = torch.zeros(C, K, device=self.device)
        count_accum = torch.zeros(C, device=self.device)

        model.eval()

        for images, targets in data_loader:
            images = images.to(self.device)
            targets = torch.stack([t.to(self.device) for t in targets])

            # compute embeddings
            emb = model(images)                      # [B, D]
            emb = F.normalize(emb, p=2, dim=1)       # [B, D]

            protos = F.normalize(self.prototypes, p=2, dim=2)  # [C, K, D]
            B, D = emb.size()

            # compute sims: [B, C, K]
            sims = torch.matmul(emb, protos.view(C * K, D).t()).view(B, C, K)

            beta = torch.exp(self.theta).detach()

            for i in range(B):
                c = int(targets[i])

                # responsibilities only over prototypes of class c
                s = sims[i, c]                      # [K]
                r = torch.softmax(beta * s, dim=0)  # [K], sum to 1

                weight_accum[c] += r
                count_accum[c] += 1

        # normalize to get per-class prototype weights
        for c in range(C):
            if count_accum[c] > 0:
                new_w_c = weight_accum[c] / count_accum[c]  # average responsibilities
                # keep numeric safety & normalization
                new_w_c = new_w_c / (new_w_c.sum() + 1e-9)
                self.weights[c].copy_(new_w_c)
            else:
                # fallback to uniform if no samples seen for class c
                self.weights[c].fill_(1.0 / K)

        print("[EM] Weight update complete.\n")
        

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

        # Here self.weights are already normalized via EM (sum to 1 per class), so use directly.
        normalized_weights = self.weights  # [C, K]

        # Prototype-embedding similarities: [B, C, K]
        proto_embd_sim = torch.matmul(normalized_embds, normalized_protos.view(-1, D).t())
        proto_embd_sim = proto_embd_sim.view(B, C, K)

        beta = torch.exp(self.theta)  # Learnable beta

        # Precompute per-sample work
        idx = torch.arange(B, device=self.device)
        pos_proto_sims_all = proto_embd_sim[idx, targets]  # [B, K]
        pos_best_idx_all = pos_proto_sims_all.argmax(dim=-1)  # [B]

        total_mpcbml_loss = 0.0  # main contrastive-bayesian loss
        mvc_batch = []           # list of per-sample MVC values (with grad)
        mu_pos_batch = []        # for logging
        mu_neg_batch = []        # for logging
        xi_batch = []            # for logging

        for i in range(B):
            y = int(targets[i].item())  # class index

            # ---------------------- Main MP-CBML term ----------------------
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

            # Find the hardest prototype (max sim) for EVERY class
            max_sim_per_class = torch.max(proto_embd_sim[i], dim=1)[0]  # [C]
            best_idx_per_class = torch.argmax(proto_embd_sim[i], dim=1)  # [C]

            # Extract info for the hardest prototype in each negative class
            neg_class_sims = max_sim_per_class[neg_mask]  # [C-1]
            neg_class_weights = normalized_weights[neg_classes, best_idx_per_class[neg_classes]]  # [C-1]
            neg_class_priors = self.class_priors[neg_classes]  # [C-1]

            # Calculate contribution for the HARDEST prototype of each negative class
            neg_class_contribution = beta * neg_class_sims + torch.log(
                neg_class_priors * neg_class_weights + 1e-9
            )

            top_n_indices = torch.topk(
                neg_class_contribution, k=self.n_negatives, dim=0, sorted=False
            )[1]  # [N]

            top_n_neg_sims = neg_class_sims[top_n_indices]      # [N]
            top_n_neg_weights = neg_class_weights[top_n_indices]  # [N]
            top_n_neg_priors = neg_class_priors[top_n_indices]    # [N]

            # ---------- Loss calculation (for N negatives) ----------
            neg_exp_sum = torch.sum(torch.exp(beta * top_n_neg_sims))
            sim_term = beta * pos_sim - torch.log(neg_exp_sum + 1e-9)

            avg_neg_prior_w = torch.mean(top_n_neg_priors * top_n_neg_weights)
            eps = 1e-9
            bias_term = torch.log(prior_pos * w_pos + eps) - torch.log(avg_neg_prior_w + eps)

            total_mpcbml_loss += (sim_term + bias_term)

            # ----------------------- MVC regularization term -----------------------
            # positive prototypes for class y
            pos_sims_i = proto_embd_sim[i, y]         # [K]
            pos_weights_i = normalized_weights[y]     # [K], sum to 1

            # weighted positive mean similarity μ_pos_i
            mu_pos_i = torch.sum(pos_sims_i * pos_weights_i)      # scalar

            # all negative prototypes (classes != y)
            neg_mask_full = torch.ones(C, dtype=torch.bool, device=self.device)
            neg_mask_full[y] = False

            neg_sims_all = proto_embd_sim[i, neg_mask_full]       # [(C-1), K]
            neg_weights_all = normalized_weights[neg_mask_full]   # [(C-1), K]

            # flatten negatives
            neg_sims_flat = neg_sims_all.reshape(-1)              # [(C-1)*K]
            neg_weights_flat = neg_weights_all.reshape(-1)        # [(C-1)*K]

            # re-normalize negative weights to sum to 1
            neg_weights_flat = neg_weights_flat / (neg_weights_flat.sum() + 1e-9)

            # weighted negative mean similarity μ_neg_i
            mu_neg_i = torch.sum(neg_sims_flat * neg_weights_flat)  # scalar

            # CBML-style center ξ_i
            xi_i = self.gamma * mu_pos_i + (1.0 - self.gamma) * mu_neg_i

            # MVC_i = E[(s_neg - ξ_i)^2] over negative prototypes (weighted)
            mvc_i = torch.sum((neg_sims_flat - xi_i) ** 2 * neg_weights_flat)

            mvc_batch.append(mvc_i)                       # keep grad for loss
            mu_pos_batch.append(mu_pos_i.detach())        # logging only
            mu_neg_batch.append(mu_neg_i.detach())
            xi_batch.append(xi_i.detach())

        # --- Final Loss Combination ---
        mpcbml_loss = - total_mpcbml_loss / B

        # --- Aggregate MVC loss (if we had at least one valid sample) ---
        if len(mvc_batch) > 0:
            mvc_loss = torch.mean(torch.stack(mvc_batch))
            self.current_mvc_value = mvc_loss.detach().item()
            self.current_positive_mean = torch.mean(torch.stack(mu_pos_batch)).item()
            self.current_negative_mean = torch.mean(torch.stack(mu_neg_batch)).item()
            self.current_xi = torch.mean(torch.stack(xi_batch)).item()
        else:
            mvc_loss = torch.tensor(0.0, device=self.device)
            self.current_mvc_value = 0.0
            self.current_positive_mean = 0.0
            self.current_negative_mean = 0.0
            self.current_xi = 0.0

        # ======================= Total Loss ============================
        loss = mpcbml_loss + self.lambda_mvc * mvc_loss

        return loss
