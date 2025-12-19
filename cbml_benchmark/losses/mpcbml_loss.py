import torch
import torch.nn as nn
import torch.nn.functional as F

from cbml_benchmark.losses.registry import LOSS
from cbml_benchmark.utils.prototype_weight_monitor import compute_proto_stats, compute_weight_stats


@LOSS.register('mpcbml_loss')
class MpcbmlLoss(nn.Module):
    def __init__(self, cfg):
        super(MpcbmlLoss, self).__init__()

        self.device_name = getattr(cfg.MODEL, 'DEVICE', 'cuda')
        self.device = torch.device(self.device_name)
        self.embed_dim = getattr(cfg.MODEL.HEAD, 'DIM', 512)
        self.num_classes = getattr(cfg.LOSSES.MPCBML_LOSS, 'N_CLASSES', 100)

        self.ma_momentum = getattr(cfg.LOSSES.MPCBML_LOSS, 'MA_MOMENTUM', 0.99)
        self.gamma_reg = getattr(cfg.LOSSES.MPCBML_LOSS, 'GAMMA_REG', 0.2)
        self.lambda_reg = getattr(cfg.LOSSES.MPCBML_LOSS, 'LAMBDA_REG', 0.5)
        # Use register_buffer so these are part of state_dict but do not get gradients
        self.register_buffer('global_s_pos', torch.tensor(0.5))
        self.register_buffer('global_s_neg', torch.tensor(0.0))
        # Flag to initialize on first batch
        self.is_initialized = False


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
        self.class_priors = nn.Parameter(
            torch.tensor(cfg.LOSSES.MPCBML_LOSS.CLASS_PRIORS, device=self.device),
            requires_grad=False
        )

        # ==========================================
        # LOGGING VARIABLES
        # ==========================================        

        # Core loss components
        self.mpcbml_total = 0.0
        self.sim_mpcbml_total = 0.0
        self.bias_mpcbml_total = 0.0
        
        # Bias term breakdown
        self.prior_bias_total = 0.0
        self.weight_bias_total = 0.0
        
        # Regularization term components
        self.current_reg_value = 0.0
        self.current_positive_mean = 0.0
        self.current_negative_mean = 0.0
        self.current_xi = 0.0
        
        # Selected similarities
        self.current_pos_sim = 0.0
        self.current_neg_sim = 0.0
        self.current_sim_margin = 0.0
        
        # Total loss
        self.current_total_loss = 0.0
        self.current_reg_contribution = 0.0
        
        # Beta tracking
        self.current_beta = 1.0

    def update_moving_averages(self, current_pos_mean, current_neg_mean):
        """
        Update the global estimates using Exponential Moving Average (EMA).
        No gradients flow through this update.
        """
        with torch.no_grad():
            if not self.is_initialized:
                self.global_s_pos.fill_(current_pos_mean)
                self.global_s_neg.fill_(current_neg_mean)
                self.is_initialized = True
            else:
                self.global_s_pos = (self.ma_momentum * self.global_s_pos +
                                     (1 - self.ma_momentum) * current_pos_mean)
                self.global_s_neg = (self.ma_momentum * self.global_s_neg +
                                     (1 - self.ma_momentum) * current_neg_mean)

    # This function is called at the begining (before training starts)
    @torch.no_grad()
    def set_prototypes_and_weights(self, prototypes, cluster_sizes):
        """
        Set prototypes based on K-means on the training set.
        Set weights based on size of each cluster.
        """

        prototypes = prototypes.to(self.device)
        self.prototypes.copy_(prototypes)

        # Save a frozen copy of the initial prototypes for monitoring
        self.initial_prototypes = prototypes.detach().clone()

        if cluster_sizes is not None and \
            cluster_sizes.shape == (self.num_classes, self.prototype_per_class):
            cluster_sizes = cluster_sizes.to(self.device).float()
            normalized_weights = cluster_sizes / (cluster_sizes.sum(dim=1, keepdim=True) + 1e-9)
            self.weights.copy_(normalized_weights)
        else:
            # Fallback to uniform
            self.weights.fill_(1.0 / self.prototype_per_class)
            
        torch.cuda.empty_cache()

    def _enforce_constraints(self):
        """Enforce normalization constraints - Eq. 3, 33"""
        with torch.no_grad():
            # Normalize prototypes to unit norm (Eq. 3)
            norms = self.prototypes.norm(p=2, dim=2, keepdim=True)
            self.prototypes.div_(norms.clamp(min=1e-8))
            
            # Ensure non-negativity for weights
            # Note: Sum constraint (Eq. 33) is preserved by mean-subtracted gradients (Eq. 41-43)
            # so we don't need explicit normalization here
            self.weights.clamp_(min=0.0)

    def show_prototype_stats(self, initial_prototypes=None):
        if initial_prototypes is None and hasattr(self, "initial_prototypes"):
            initial_prototypes = self.initial_prototypes
        return compute_proto_stats(self.prototypes.detach(), initial_prototypes)
        
    def show_weight_stats(self):
        return compute_weight_stats(self.weights.detach())
    
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

    def constrained_weight_update(self):
        if self.weights.grad is None:
            return
        
        # Get gradients [C, K]
        grad_w = self.weights.grad
        
        # Compute mean gradient per class
        mean_grad = grad_w.mean(dim=1, keepdim=True)  # [C, 1]
        
        # Compute mean-subtracted gradient
        grad_tilde = grad_w - mean_grad  # [C, K]
        
        # Replace gradients with mean-subtracted version
        self.weights.grad.copy_(grad_tilde)

    def forward(self, embeddings, targets):
        # Enforce unit-norm prototypes
        self._enforce_constraints()

        embeddings = embeddings.to(self.device)
        targets = targets.to(self.device)

        B = embeddings.size(0)
        C, K, D = self.num_classes, self.prototype_per_class, self.embed_dim
        eps = 1e-9

        # -----------------------------------------------------
        # 0. Normalize embeddings
        # -----------------------------------------------------
        z = F.normalize(embeddings, p=2, dim=1)  # [B, D]
        protos = self.prototypes                  # [C, K, D]
        W = self.weights                          # [C, K]

        # Get current beta
        beta = torch.exp(self.theta)
        self.current_beta = beta.item()

        # -----------------------------------------------------
        # 1. Compute similarities
        # -----------------------------------------------------
        # sims[b, c, k] = z_b · μ_{c,k}
        sims = torch.matmul(z, protos.view(C * K, D).t()).view(B, C, K)  # [B, C, K]

        # -----------------------------------------------------
        # 2. Log-probability contribution for each prototype
        #    log p(z | c,k) ∝ log w_{c,k} + β * s_{b,c,k}
        # -----------------------------------------------------
        log_prob_contrib = torch.log(W.unsqueeze(0) + eps) + beta * sims  # [B, C, K]

        # -----------------------------------------------------
        # 3. Masks
        # -----------------------------------------------------
        y_onehot = F.one_hot(targets, num_classes=C).bool()  # [B, C]
        neg_mask = ~y_onehot                                 # [B, C]

        # -----------------------------------------------------
        # 4. POSITIVE SELECTION (dominant positive prototype)
        #    ℓ_i^+ = argmax_ℓ [log w_{y_i,ℓ} + β s_{i,y_i,ℓ}]
        # -----------------------------------------------------
        pos_log_contrib = log_prob_contrib[y_onehot].view(B, K)  # [B, K]
        pos_raw = sims[y_onehot].view(B, K)                      # [B, K]
        pos_w = W[targets]                                       # [B, K]
        prior_pos = self.class_priors[targets]                   # [B]

        best_pos_idx = pos_log_contrib.argmax(dim=-1)            # [B]
        pos_sim = pos_raw[torch.arange(B, device=self.device), best_pos_idx]  # [B]
        w_pos = pos_w[torch.arange(B, device=self.device), best_pos_idx]      # [B]

        # -----------------------------------------------------
        # 5. NEGATIVE SELECTION (dominant negative prototype)
        #    1) best prototype per negative class:
        #       ℓ_i^{(c)} = argmax_ℓ [log w_{c,ℓ} + β s_{i,c,ℓ}]
        #    2) dominant negative class:
        #       c_i^- = argmax_c [log p(c) + log w_{c,ℓ*} + β s_{i,c,ℓ*}]
        # -----------------------------------------------------
        # Extract negative entries: shapes [B, C-1, K]
        neg_log_contrib = log_prob_contrib[neg_mask].view(B, C - 1, K)  # [B, C-1, K]
        neg_raw = sims[neg_mask].view(B, C - 1, K)                      # [B, C-1, K]

        # Expand weights to [B, C, K] and then mask to negatives: [B, C-1, K]
        W_expanded = W.unsqueeze(0).expand(B, C, K)                      # [B, C, K]
        neg_W = W_expanded[neg_mask].view(B, C - 1, K)                   # [B, C-1, K]

        # Negative class priors: [B, C-1]
        class_priors_exp = self.class_priors.unsqueeze(0).expand(B, C)   # [B, C]
        neg_priors = class_priors_exp[neg_mask].view(B, C - 1)           # [B, C-1]

        # Step 1: best prototype per negative class (by log_prob_contrib)
        best_neg_log_contrib, best_neg_k = neg_log_contrib.max(dim=-1)   # [B, C-1]

        # Step 2: score each negative class at its best prototype
        # Build batched indices [B, C-1] for advanced indexing
        b_idx_full = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, C - 1)      # [B, C-1]
        c_idx_full = torch.arange(C - 1, device=self.device).unsqueeze(0).expand(B, -1)      # [B, C-1]

        # log w_{c,ℓ*} and s_{i,c,ℓ*} at best prototype per neg class
        best_neg_log_w = torch.log(
            neg_W[b_idx_full, c_idx_full, best_neg_k] + eps
        )  # [B, C-1]

        best_neg_raw_sim = neg_raw[b_idx_full, c_idx_full, best_neg_k]   # [B, C-1]

        # Score for each negative class:
        # log p(c) + log w_{c,ℓ*} + β s_{i,c,ℓ*}
        neg_class_scores = (
            torch.log(neg_priors + eps) + best_neg_log_w + beta * best_neg_raw_sim
        )  # [B, C-1]

        # Dominant negative class index (in compressed negative-class axis)
        best_neg_class = neg_class_scores.argmax(dim=-1)  # [B]

        # Now extract the final dominant negative prototype for each sample
        b_idx = torch.arange(B, device=self.device)  # [B]

        best_neg_sim = neg_raw[
            b_idx, best_neg_class, best_neg_k[b_idx, best_neg_class]
        ]  # [B]
        w_neg = neg_W[
            b_idx, best_neg_class, best_neg_k[b_idx, best_neg_class]
        ]  # [B]
        prior_neg = neg_priors[b_idx, best_neg_class]  # [B]

        # Log similarities for monitoring
        self.current_pos_sim = pos_sim.mean().item()
        self.current_neg_sim = best_neg_sim.mean().item()
        self.current_sim_margin = (pos_sim - best_neg_sim).mean().item()

        # -----------------------------------------------------
        # 6. BAYESIAN LOSS: -log p(c+ | z_i)
        # -----------------------------------------------------
        log_A_pos = (
            torch.log(prior_pos + eps) +
            torch.log(w_pos + eps) +
            beta * pos_sim
        )  # [B]

        log_A_neg = (
            torch.log(prior_neg + eps) +
            torch.log(w_neg + eps) +
            beta * best_neg_sim
        )  # [B]

        # log(A+ + A-) in a numerically stable way
        log_denominator = torch.logsumexp(
            torch.stack([log_A_pos, log_A_neg], dim=0), dim=0
        )  # [B]

        # Per-sample loss: -log A+ + log(A+ + A-)
        mpcbml_loss = (-log_A_pos + log_denominator).mean()  # scalar

        # -----------------------------------------------------
        # 7. COMPONENT LOGGING (for debugging)
        # -----------------------------------------------------
        sim_term = beta * (pos_sim - best_neg_sim)  # [B]

        log_prior_pos = torch.log(prior_pos + eps)
        log_prior_neg = torch.log(prior_neg + eps)
        log_w_pos = torch.log(w_pos + eps)
        log_w_neg = torch.log(w_neg + eps)

        prior_bias = log_prior_pos - log_prior_neg  # [B]
        weight_bias = log_w_pos - log_w_neg         # [B]
        bias_term = prior_bias + weight_bias        # [B]

        self.sim_mpcbml_total = (-sim_term).mean().item()
        self.bias_mpcbml_total = (-bias_term).mean().item()
        self.prior_bias_total = prior_bias.mean().item()
        self.weight_bias_total = weight_bias.mean().item()
        self.mpcbml_total = mpcbml_loss.item()

        # -----------------------------------------------------
        # 8. REGULARIZATION LOGIC
        # -----------------------------------------------------
        # 1. Compute batch means (detach to stop gradient flow into the MA update)
        batch_pos_mean = pos_sim.detach().mean()
        batch_neg_mean = best_neg_sim.detach().mean()

        # 2. Update Global Moving Averages
        if self.training:
            self.update_moving_averages(batch_pos_mean, batch_neg_mean)

        # 3. Compute Threshold (xi)
        # xi is a constant regarding gradients, it's a "target" derived from history
        xi = (self.gamma_reg * self.global_s_pos + 
              (1 - self.gamma_reg) * self.global_s_neg)
        
        reg_loss = F.relu(xi - best_neg_sim).mean() 

        # 5. Add to Total Loss
        total_loss = mpcbml_loss + self.lambda_reg * reg_loss

        # ==========================================
        # LOGGING UPDATES
        # ==========================================
        self.current_reg_value = reg_loss.item()
        self.current_xi = xi.item()
        self.current_positive_mean = self.global_s_pos.item()
        self.current_negative_mean = self.global_s_neg.item()

        # -----------------------------------------------------
        # 9. FINAL LOSS
        # -----------------------------------------------------
        return total_loss

