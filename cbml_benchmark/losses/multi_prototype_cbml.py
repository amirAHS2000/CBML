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

        self.mu_pos = 0.0
        self.mu_neg = 0.0
        self.momentum_coef = 0.99
        self.momentum_coef_power = 0.0

        self.gamma = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'HYPER_WEIGHT', 0.2)
        self.lambda_mvc = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'REG_WEIGHT', 10.0)

        # Learnable theta (log beta) - Eq. 6
        init_theta = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'INIT_THETA', 1.0)
        self.theta = nn.Parameter(torch.tensor(init_theta, device=self.device))

        # Prototypes [C, K, D] - Eq. 2
        self.prototype_per_class = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'PROTOTYPE_PER_CLASS', 3)
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

        # Mixture weights [C, K] with constrained gradient descent (Section 2)
        # These require gradients and will be updated with Lagrange multipliers
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class, device=self.device)
            / self.prototype_per_class,
            requires_grad=True  # Enable gradients for constrained optimization
        )

        # Class priors [C] - Eq. 14-15
        self.class_priors = nn.Parameter(
            torch.tensor(cfg.LOSSES.MULTI_PROTOTYPE_CBML.CLASS_PRIORS, device=self.device),
            requires_grad=False
        )

        # Learning rate for weight updates (α in Eq. 35)
        self.weight_lr = getattr(cfg.LOSSES.MULTI_PROTOTYPE_CBML, 'WEIGHT_LR', 0.01)

        # ==========================================
        # ENHANCED LOGGING VARIABLES
        # ==========================================
        
        # Core loss components
        self.mpcbml_total = 0.0
        self.sim_mpcbml_total = 0.0
        self.bias_mpcbml_total = 0.0
        
        # Bias term breakdown
        self.prior_bias_total = 0.0
        self.weight_bias_total = 0.0
        
        # MVC components (for future use)
        self.current_mvc_value = 0.0
        self.current_positive_mean = 0.0
        self.current_negative_mean = 0.0
        self.current_xi = 0.0
        
        # Selected similarities
        self.current_pos_sim = 0.0
        self.current_neg_sim = 0.0
        self.current_sim_margin = 0.0
        
        # Total loss
        self.current_total_loss = 0.0
        self.current_mvc_contribution = 0.0
        
        # Beta tracking
        self.current_beta = 1.0

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
    
    def show_mvc_value(self):
        return getattr(self, 'current_mvc_value', None)
    
    def get_beta(self):
        """Return current beta value"""
        return torch.exp(self.theta).item()

    def constrained_weight_update(self):
        """
        Perform constrained gradient descent on weights using Lagrange multipliers.
        Implements Eq. 41-43 from the PDF.
        
        Key insight: By subtracting the mean gradient, the sum of weights is automatically
        preserved (∑w remains 1) without explicit normalization, since ∑g̃ = 0 by construction.
        
        This should be called AFTER loss.backward() but BEFORE optimizer.step()
        """
        if self.weights.grad is None:
            return
        
        # Get gradients [C, K]
        grad_w = self.weights.grad
        
        # Compute mean gradient per class (related to λ_j in Eq. 40)
        mean_grad = grad_w.mean(dim=1, keepdim=True)  # [C, 1]
        
        # Compute mean-subtracted gradient (Eq. 42)
        # This ensures ∑ℓ g̃_j^ℓ = 0, which preserves ∑ℓ w_j^ℓ = 1
        grad_tilde = grad_w - mean_grad  # [C, K]
        
        # Replace gradients with mean-subtracted version
        # The optimizer will then apply: w ← w - α·g̃
        self.weights.grad.copy_(grad_tilde)
        
        # No explicit normalization needed! The constraint is preserved by construction.

    # ------------------------------------------------------
    # FORWARD: Complete MP-CBML Loss (Eq. 24)
    # ------------------------------------------------------
    def forward(self, embeddings, targets):
        self._enforce_constraints()

        embeddings = embeddings.to(self.device)
        targets = targets.to(self.device)

        B = embeddings.size(0)
        C, K, D = self.num_classes, self.prototype_per_class, self.embed_dim
        eps = 1e-9

        # -----------------------------------------------------
        # 0. Normalize embeddings (Eq. 1)
        # -----------------------------------------------------
        z = F.normalize(embeddings, p=2, dim=1)  # [B, D]
        protos = self.prototypes  # [C, K, D]
        W = self.weights  # [C, K]

        # Get current beta (Eq. 6)
        beta = torch.exp(self.theta)
        self.current_beta = beta.item()

        # -----------------------------------------------------
        # 1. Compute similarities (Eq. 7-8)
        # -----------------------------------------------------
        sims = torch.matmul(z, protos.view(C*K, D).t()).view(B, C, K)  # [B,C,K]
        weighted_sims = sims * W.unsqueeze(0)  # [B,C,K]

        # -----------------------------------------------------
        # 2. Masks
        # -----------------------------------------------------
        y_onehot = F.one_hot(targets, num_classes=C).bool()  # [B,C]
        neg_mask = ~y_onehot  # [B,C]

        # -----------------------------------------------------
        # 3. POSITIVE SELECTION (Eq. 9-10)
        # -----------------------------------------------------
        pos_weighted = weighted_sims[y_onehot].view(B, K)  # [B,K]
        pos_raw = sims[y_onehot].view(B, K)  # [B,K]
        pos_w = W[targets]  # [B,K]
        prior_pos = self.class_priors[targets]  # [B]

        best_pos_idx = pos_weighted.argmax(dim=-1)  # [B]
        pos_sim = pos_raw[torch.arange(B), best_pos_idx]  # [B]
        w_pos = pos_w[torch.arange(B), best_pos_idx]  # [B]

        # -----------------------------------------------------
        # 4. NEGATIVE SELECTION (Eq. 11-13)
        # -----------------------------------------------------
        # Expand weights to [B,C,K]
        W_expanded = W.unsqueeze(0).expand(B, C, K)  # [B,C,K]

        # Extract negative prototype info
        neg_weighted = weighted_sims[neg_mask].view(B, C-1, K)  # [B,C-1,K]
        neg_raw = sims[neg_mask].view(B, C-1, K)  # [B,C-1,K]
        neg_W = W_expanded[neg_mask].view(B, C-1, K)  # [B,C-1,K]

        # Negative priors
        class_priors_exp = self.class_priors.unsqueeze(0).expand(B, C)  # [B,C]
        neg_priors = class_priors_exp[neg_mask].view(B, C-1)  # [B,C-1]

        # Best prototype per negative class
        neg_weighted_max, neg_best_k = neg_weighted.max(dim=-1)  # [B,C-1]

        # Best negative class
        best_neg_class = neg_weighted_max.argmax(dim=-1)  # [B]
        b_idx = torch.arange(B, device=self.device)

        # Extract selected negative prototype info
        best_neg_sim = neg_raw[b_idx, best_neg_class, neg_best_k[b_idx, best_neg_class]]  # [B]
        w_neg = neg_W[b_idx, best_neg_class, neg_best_k[b_idx, best_neg_class]]  # [B]
        prior_neg = neg_priors[b_idx, best_neg_class]  # [B]

        # Log selected similarities
        self.current_pos_sim = pos_sim.mean().item()
        self.current_neg_sim = best_neg_sim.mean().item()
        self.current_sim_margin = (pos_sim - best_neg_sim).mean().item()

        # -----------------------------------------------------
        # 5. BAYESIAN LOSS (Eq. 21-24)
        # -----------------------------------------------------
        # Compute log of A+ and A- terms (Eq. 19-20)
        log_A_pos = torch.log(prior_pos + eps) + torch.log(w_pos + eps) + beta * pos_sim  # [B]
        log_A_neg = torch.log(prior_neg + eps) + torch.log(w_neg + eps) + beta * best_neg_sim  # [B]
        
        # Compute log(A+ + A-) using logsumexp for numerical stability
        log_denominator = torch.logsumexp(torch.stack([log_A_pos, log_A_neg], dim=0), dim=0)  # [B]
        
        # MP-CBML loss: -log p(c+ | z_i) (Eq. 23-24)
        # = -log(A+) + log(A+ + A-)
        mpcbml_loss = (-log_A_pos + log_denominator).mean()  # scalar
        
        # -----------------------------------------------------
        # 6. COMPONENT LOGGING (for debugging)
        # -----------------------------------------------------
        # Decompose for logging purposes
        sim_term = beta * (pos_sim - best_neg_sim)  # [B]
        
        # Bias components
        log_prior_pos = torch.log(prior_pos + eps)
        log_prior_neg = torch.log(prior_neg + eps)
        log_w_pos = torch.log(w_pos + eps)
        log_w_neg = torch.log(w_neg + eps)
        
        prior_bias = log_prior_pos - log_prior_neg  # [B]
        weight_bias = log_w_pos - log_w_neg  # [B]
        bias_term = prior_bias + weight_bias  # [B]
        
        # Log components (note: these are for monitoring, not used in actual loss)
        self.sim_mpcbml_total = (-sim_term).mean().item()
        self.bias_mpcbml_total = (-bias_term).mean().item()
        self.prior_bias_total = prior_bias.mean().item()
        self.weight_bias_total = weight_bias.mean().item()
        self.mpcbml_total = mpcbml_loss.item()

        # -----------------------------------------------------
        # 7. MVC REGULARIZER (Corrected EMA Implementation)
        # -----------------------------------------------------
        
        # 1. Calculate Batch Statistics (Scalar values)
        # We detach() because the Target (xi) should be a fixed reference point,
        # not a variable we backpropagate through.
        batch_pos_mean = pos_sim.detach().mean()      # Scalar
        batch_neg_mean = best_neg_sim.detach().mean() # Scalar

        # 2. Update EMA States (Accumulators)
        # Note: We do NOT divide by the correction factor here. We keep the raw state.
        # if self.training:
        
        # Compute bias correction factor
        # Protect against division by zero in first iteration
        self.momentum_coef_power = self.momentum_coef * self.momentum_coef_power + \
                                (1 - self.momentum_coef)
        
        # Update EMA
        self.mu_pos = self.momentum_coef * self.mu_pos + \
                    (1 - self.momentum_coef) * batch_pos_mean
        self.mu_neg = self.momentum_coef * self.mu_neg + \
                    (1 - self.momentum_coef) * batch_neg_mean

        # 3. Apply Bias Correction (Temporary variables for calculation)
        # This handles the "cold start" problem where EMA starts at 0.
        correction_factor = max(1.0 - self.momentum_coef_power, 1e-8)
        
        debiased_pos = self.mu_pos / correction_factor
        debiased_neg = self.mu_neg / correction_factor

        # 4. Calculate Global Decision Center (xi)
        # We use the STABLE, DEBIASED global averages
        xi = self.gamma * debiased_pos + (1 - self.gamma) * debiased_neg
        
        # Ensure xi is treated as a constant for the loss calculation
        xi = xi.detach()

        # 5. MVC Loss Calculation
        # We penalize the Hardest Negative (best_neg_sim) for deviating from the Global Center (xi).
        # This keeps the "Spring" anchored to the global average, not the jittery batch average.
        mvc_batch = (best_neg_sim - xi) ** 2
        
        mvc_loss = mvc_batch.mean()

        # Log MVC components (Log the debiased global values to see the trend)
        self.current_mvc_value = mvc_loss.item()
        self.current_positive_mean = debiased_pos.item()
        self.current_negative_mean = debiased_neg.item()
        self.current_xi = xi.item()
        
        # -----------------------------------------------------
        # 8. FINAL LOSS (Eq. 32)
        # -----------------------------------------------------
        # L_total = L_MP + λ_MVC · L_MVC
        total_loss = mpcbml_loss + self.lambda_mvc * mvc_loss
        
        self.current_total_loss = total_loss.item()
        self.current_mvc_contribution = (self.lambda_mvc * mvc_loss).item()
        
        return total_loss