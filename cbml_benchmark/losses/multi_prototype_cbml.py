import torch
import torch.nn as nn
import torch.nn.functional as F

from cbml_benchmark.losses.registry import LOSS
from cbml_benchmark.utils.prototype_weight_monitor import compute_proto_stats, compute_weight_stats


@LOSS.register('multi_prototype_cbml')
class MultiPrototypeCBMLLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiPrototypeCBMLLoss, self).__init__()

        self.num_classes = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'N_CLASSES', 100)
        self.embed_dim = getattr(cfg.MODEL.HEAD, 'DIM', 512)
        self.device_name = getattr(cfg.MODEL, 'DEVICE', 'cuda')
        self.device = torch.device(self.device_name)

        self.gamma = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'HYPER_WEIGHT', 0.2)
        self.lambda_mvc = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'REG_WEIGHT', 20.0)

        # Learnable theta (log beta)
        # init_theta = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'INIT_THETA', 2.3)
        # self.theta = nn.Parameter(torch.tensor(init_theta, device=self.device))

        # Prototypes [C, K, D]
        self.prototype_per_class = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'PROTOTYPE_PER_CLASS', 3)
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

        # choose between EM-based weights or learnable softmax weights
        self.use_learnable_weights = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'LEARNABLE_WEIGHTS', False)

        if self.use_learnable_weights:
            # learnable log-weights with gradients (softmax approach)
            self.log_weights = nn.Parameter(
                torch.zeros(self.num_classes, self.prototype_per_class, device=self.device),
                requires_grad=True
            )
        else:
            # EM-based weights (no gradients)
            self.weights = nn.Parameter(
                torch.ones(self.num_classes, self.prototype_per_class, device=self.device)
                / self.prototype_per_class,
                requires_grad=False
            )

        # Class priors [C]
        self.class_priors = nn.Parameter(
            torch.tensor(cfg.LOSSES.MULTI_PROTOTYPE_CBML.CLASS_PRIORS, device=self.device),
            requires_grad=False
        )

        # ==========================================
        # ENHANCED LOGGING VARIABLES
        # ==========================================
        
        # Core loss components (for oscillation debugging)
        self.mpcbml_total = 0.0              # Total MP-CBML loss
        self.sim_mpcbml_total = 0.0          # Similarity term only: -β(s+ - s-)
        self.bias_mpcbml_total = 0.0         # Bias term only: -log(p+w+/p-w-)
        
        # Bias term breakdown
        self.prior_bias_total = 0.0          # log(p+/p-) component
        self.weight_bias_total = 0.0         # log(w+/w-) component
        
        # MVC components
        self.current_mvc_value = 0.0
        self.current_positive_mean = 0.0     # μ+
        self.current_negative_mean = 0.0     # μ-
        self.current_xi = 0.0                # ξ decision center
        
        # Selected similarities
        self.current_pos_sim = 0.0           # s+ (selected positive similarity)
        self.current_neg_sim = 0.0           # s- (selected negative similarity)
        self.current_sim_margin = 0.0        # s+ - s-
        
        # Total loss
        self.current_total_loss = 0.0        # mpcbml + λ*mvc
        self.current_mvc_contribution = 0.0  # λ*mvc

    @property
    def weights(self):
        """
        Return normalized weights.
        If using learnable weights, apply softmax. Otherwise, return EM-based weights directly.
        """
        if self.use_learnable_weights:
            return F.softmax(self.log_weights, dim=1)
        else:
            return self._parameters['weights']

    @torch.no_grad()
    def set_prototypes_and_weights(self, prototypes, cluster_sizes):
        """Set prototypes and initialize weights based on k-means cluster sizes."""
        # Prototypes
        prototypes = prototypes.to(self.device)
        self.prototypes.copy_(prototypes)

        # Save a frozen copy of the initial prototypes for monitoring
        self.initial_prototypes = prototypes.detach().clone()

        # Initialize weights based on cluster sizes
        if (
            cluster_sizes is not None
            and cluster_sizes.shape == (self.num_classes, self.prototype_per_class)
        ):
            cluster_sizes = cluster_sizes.to(self.device).float()
            normalized_weights = cluster_sizes / (cluster_sizes.sum(dim=1, keepdim=True) + 1e-9)
            
            if self.use_learnable_weights:
                # Convert to log-space for softmax parameterization
                self.log_weights.copy_(torch.log(normalized_weights + 1e-9))
            else:
                # Direct copy for EM-based weights
                self._parameters['weights'].copy_(normalized_weights)
        else:
            # Fallback to uniform
            if self.use_learnable_weights:
                self.log_weights.fill_(0.0)  # softmax([0,0,0]) = uniform
            else:
                self._parameters['weights'].fill_(1.0 / self.prototype_per_class)

        torch.cuda.empty_cache()

    def _enforce_constraints(self):
        """Enforce normalization constraints."""
        with torch.no_grad():
            # Normalize prototypes to unit norm
            norms = self.prototypes.norm(p=2, dim=2, keepdim=True)
            self.prototypes.div_(norms.clamp(min=1e-8))
            
            # Normalize weights (only if using EM-based approach)
            if not self.use_learnable_weights:
                weight_sums = self._parameters['weights'].sum(dim=1, keepdim=True)
                self._parameters['weights'].div_(weight_sums.clamp(min=1e-8))
            # If using learnable weights, softmax automatically normalizes
    
    def show_prototype_stats(self, initial_prototypes=None):
        if initial_prototypes is None and hasattr(self, "initial_prototypes"):
            initial_prototypes = self.initial_prototypes
        return compute_proto_stats(self.prototypes.detach(), initial_prototypes)
        
    def show_weight_stats(self):
        return compute_weight_stats(self.weights.detach())
    
    def show_log_weight_stats(self):
        """Show statistics of raw log-weights (only for learnable weights)."""
        if self.use_learnable_weights:
            log_w = self.log_weights.detach()
            return {
                'mean': log_w.mean().item(),
                'std': log_w.std().item(),
                'min': log_w.min().item(),
                'max': log_w.max().item(),
                'range': (log_w.max() - log_w.min()).item()
            }
        else:
            return None
        
    def show_weight_entropy(self):
        """Compute entropy of weight distribution per class."""
        W = self.weights.detach()  # [C, K]
        entropy = -(W * torch.log(W + 1e-9)).sum(dim=1)  # [C]
        return {
            'mean_entropy': entropy.mean().item(),
            'min_entropy': entropy.min().item(),
            'max_entropy': entropy.max().item(),
            'std_entropy': entropy.std().item()
        }
    
    def show_mvc_value(self):
        return getattr(self, 'current_mvc_value', None)

    # ------------------------------------------------------
    # EM UPDATE (for weights) - only used when not using learnable weights
    # ------------------------------------------------------
    @torch.no_grad()
    def em_update_weights(self, model, data_loader):
        if self.use_learnable_weights:
            print("[EM] Skipped: using learnable softmax weights instead.")
            return
            
        print("\n[EM] Updating prototype weights ...")
        C, K = self.num_classes, self.prototype_per_class

        weight_accum = torch.zeros(C, K, device=self.device)
        count_accum = torch.zeros(C, device=self.device)

        model.eval()
        protos = self.prototypes
        beta = 1.0

        for images, targets in data_loader:
            images = images.to(self.device)
            targets = torch.stack([t.to(self.device) for t in targets])

            emb = F.normalize(model(images), p=2, dim=1)
            sims = torch.matmul(emb, protos.view(C*K, -1).t()).view(-1, C, K)

            for i in range(emb.size(0)):
                c = int(targets[i])
                s = sims[i, c]         # [K]
                w = self.weights[c]    # [K]

                # EM responsibility: r ∝ w_l * exp(beta * s_l)
                r = w * torch.exp(beta * s)
                r = r / (r.sum() + 1e-9)

                weight_accum[c] += r
                count_accum[c] += 1

        for c in range(C):
            if count_accum[c] > 0:
                new_w = weight_accum[c] / count_accum[c]
                new_w = new_w / (new_w.sum() + 1e-9)
                self._parameters['weights'][c].copy_(new_w)
            else:
                self._parameters['weights'][c].fill_(1.0 / K)

        self._enforce_constraints()
        print("[EM] Weight update complete.\n")

    # ------------------------------------------------------
    # FORWARD: Complete MP-CBML + MVC
    # ------------------------------------------------------
    def forward(self, embeddings, targets):
        self._enforce_constraints()

        embeddings = embeddings.to(self.device)
        targets = targets.to(self.device)

        B = embeddings.size(0)
        C, K, D = self.num_classes, self.prototype_per_class, self.embed_dim
        eps = 1e-9

        # -----------------------------------------------------
        # 0. Normalize
        # -----------------------------------------------------
        z = F.normalize(embeddings, p=2, dim=1)              # [B, D]
        protos = self.prototypes    # [C, K, D]
        W = self.weights                                     # [C, K]

        # -----------------------------------------------------
        # 1. Compute similarities
        # -----------------------------------------------------
        sims = torch.matmul(z, protos.view(C*K, D).t()).view(B, C, K)  # [B,C,K]
        weighted_sims = sims * W.unsqueeze(0)                          # [B,C,K]

        # beta = torch.exp(self.theta)
        beta = 1.0

        # -----------------------------------------------------
        # 2. Masks
        # -----------------------------------------------------
        y_onehot = F.one_hot(targets, num_classes=C).bool()  # [B,C]
        neg_mask = ~y_onehot                                 # [B,C]

        # -----------------------------------------------------
        # 3. POSITIVE SELECTION
        # -----------------------------------------------------
        pos_weighted = weighted_sims[y_onehot].view(B, K)      # [B,K]
        pos_raw = sims[y_onehot].view(B, K)                    # [B,K]
        pos_w = W[targets]                                     # [B,K]
        prior_pos = self.class_priors[targets]                 # [B]

        best_pos_idx = pos_weighted.argmax(dim=-1)             # [B]
        pos_sim = pos_raw[torch.arange(B), best_pos_idx]       # [B]
        w_pos = pos_w[torch.arange(B), best_pos_idx]           # [B]

        # -----------------------------------------------------
        # 4. NEGATIVE SELECTION (corrected masking)
        # -----------------------------------------------------
        # Expand weights to [B,C,K]
        W_expanded = W.unsqueeze(0).expand(B, C, K)  # [B,C,K]

        # Extract negative prototype info
        neg_weighted = weighted_sims[neg_mask].view(B, C-1, K)  # [B,C-1,K]
        neg_raw = sims[neg_mask].view(B, C-1, K)                # [B,C-1,K]
        neg_W = W_expanded[neg_mask].view(B, C-1, K)            # [B,C-1,K]

        # Negative priors (correct batching)
        class_priors_exp = self.class_priors.unsqueeze(0).expand(B, C)  # [B,C]
        neg_priors = class_priors_exp[neg_mask].view(B, C-1)            # [B,C-1]

        # Best prototype each negative class
        neg_weighted_max, neg_best_k = neg_weighted.max(dim=-1)  # [B,C-1]

        # Best negative class
        best_neg_class = neg_weighted_max.argmax(dim=-1)  # [B]
        b_idx = torch.arange(B, device=self.device)

        # Extract selected negative prototype info
        best_neg_sim = neg_raw[b_idx, best_neg_class, neg_best_k[b_idx, best_neg_class]]    # [B]
        w_neg = neg_W[b_idx, best_neg_class, neg_best_k[b_idx, best_neg_class]]             # [B]
        prior_neg = neg_priors[b_idx, best_neg_class]                                       # [B]

        # Log selected similarities
        self.current_pos_sim = pos_sim.mean().item()
        self.current_neg_sim = best_neg_sim.mean().item()
        self.current_sim_margin = (pos_sim - best_neg_sim).mean().item()

        # -----------------------------------------------------
        # 5. MAIN LOSS
        # -----------------------------------------------------
        # Similarity term: β(s+ - s-)
        sim_term = beta * (pos_sim - best_neg_sim)  # [B]
        
        # Bias components
        log_prior_pos = torch.log(prior_pos + eps)
        log_prior_neg = torch.log(prior_neg + eps)
        log_w_pos = torch.log(w_pos + eps)
        log_w_neg = torch.log(w_neg + eps)
        
        prior_bias = log_prior_pos - log_prior_neg  # [B]
        weight_bias = log_w_pos - log_w_neg         # [B]
        bias_term = prior_bias + weight_bias        # [B]
        
        # Total MP-CBML loss
        mpcbml_loss = -(sim_term + bias_term).mean()

        # Log ALL components individually
        self.sim_mpcbml_total = (-sim_term).mean().item()      # Negative similarity term
        self.bias_mpcbml_total = (-bias_term).mean().item()    # Negative bias term
        self.prior_bias_total = prior_bias.mean().item()       # Prior contribution
        self.weight_bias_total = weight_bias.mean().item()     # Weight contribution
        self.mpcbml_total = mpcbml_loss.item()                 # Total contrastive loss

        # -----------------------------------------------------
        # 6. MVC (global negative)
        # -----------------------------------------------------
        neg_sims_flat = neg_raw.reshape(B, (C-1)*K)               # [B,(C-1)K]
        neg_weights_flat = neg_W.reshape(B, (C-1)*K)              # [B,(C-1)K]
        neg_weights_flat = neg_weights_flat / (neg_weights_flat.sum(dim=-1, keepdim=True)+eps)

        mu_pos = (pos_raw * pos_w).sum(dim=-1)                   # [B]
        mu_neg = (neg_sims_flat * neg_weights_flat).sum(dim=-1)  # [B]
        xi = self.gamma * mu_pos + (1-self.gamma) * mu_neg       # [B]

        mvc = ((neg_sims_flat - xi[:,None])**2 * neg_weights_flat).sum(dim=-1)
        mvc_loss = mvc.mean()

        # Log MVC components
        self.current_mvc_value = mvc_loss.item()
        self.current_positive_mean = mu_pos.mean().item()
        self.current_negative_mean = mu_neg.mean().item()
        self.current_xi = xi.mean().item()

        # -----------------------------------------------------
        # 7. FINAL LOSS
        # -----------------------------------------------------
        total_loss = mpcbml_loss + self.lambda_mvc * mvc_loss
        
        self.current_total_loss = total_loss.item()
        self.current_mvc_contribution = (self.lambda_mvc * mvc_loss).item()
        
        return total_loss