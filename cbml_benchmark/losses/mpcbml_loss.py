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
        self.register_buffer('class_priors', 
            torch.tensor(getattr(cfg.LOSSES.MPCBML_LOSS, 'CLASS_PRIORS', 
                                 [1.0/self.num_classes]*self.num_classes))
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

    def update_moving_averages(self, batch_pos_mean, batch_neg_mean):
        if not self.is_initialized:
            self.global_s_pos.data.copy_(batch_pos_mean)
            self.global_s_neg.data.copy_(batch_neg_mean)
            self.is_initialized = True
        else:
            self.global_s_pos.mul_(self.ma_momentum).add_(batch_pos_mean * (1 - self.ma_momentum))
            self.global_s_neg.mul_(self.ma_momentum).add_(batch_neg_mean * (1 - self.ma_momentum))

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
            # Normalize Prototypes
            self.prototypes.div_(self.prototypes.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8))
            
            # Ensure non-negativity for weights
            # Note: Sum constraint (Eq. 33) is preserved by mean-subtracted gradients (Eq. 41-43)
            # so we don't need explicit normalization here
            self.weights.clamp_(min=1e-6)

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
        
        grad_w.sub_(mean_grad)

    def forward(self, embeddings, targets):
        # 1. Enforce constraints immediately to ensure valid probabilities
        self._enforce_constraints()
        
        embeddings = embeddings.to(self.device)
        targets = targets.to(self.device)
        
        # 2. Setup
        # Use functional normalize for embeddings (cleaner for gradients)
        z = F.normalize(embeddings, p=2, dim=1)           # [B, D]
        # We can use self.prototypes directly as they were normalized in _enforce_constraints
        
        B = z.shape[0]
        C, K = self.num_classes, self.prototype_per_class
        beta = torch.exp(self.theta)
        eps = 1e-9

        # -----------------------------------------------------
        # 3. Compute Log-Contributions
        # -----------------------------------------------------
        # Flatten protos for one big matmul: [C*K, D]
        flat_protos = self.prototypes.view(C * K, -1)
        
        # Similarity: [B, C, K]
        sims = torch.matmul(z, flat_protos.t()).view(B, C, K)

        # Log terms: log p(c) + log w + beta*s
        # Expand dims for broadcasting: [1, C, 1] and [1, C, K]
        log_prior = torch.log(self.class_priors + eps).view(1, C, 1)
        log_weight = torch.log(self.weights + eps).unsqueeze(0)
        
        # r_score: [B, C, K]
        r_scores = log_prior + log_weight + (beta * sims)

        # -----------------------------------------------------
        # 4. Mining Dominant Components
        # -----------------------------------------------------
        target_mask = F.one_hot(targets, num_classes=C).bool() # [B, C]
        
        # --- Positive ---
        # Get scores for true class [B, K]
        pos_scores = r_scores[target_mask].view(B, K)
        # Dominant positive prototype index
        r_pos, best_pos_k = pos_scores.max(dim=1) # [B]
        
        # Extract raw similarity for regularization (s_pos)
        # We need to grab the specific sim value corresponding to best_pos_k
        pos_sims = sims[target_mask].view(B, K)
        s_pos = pos_sims.gather(1, best_pos_k.unsqueeze(1)).squeeze(1) # [B]

        # --- Negative ---
        # 1. Best prototype per class (max over K) -> [B, C]
        r_best_k_val, best_k_idx = r_scores.max(dim=2)
        s_best_k_val = sims.gather(2, best_k_idx.unsqueeze(2)).squeeze(2)

        # 2. Mask positive class to -inf so it isn't selected as negative
        r_neg_candidates = r_best_k_val.clone()
        r_neg_candidates[target_mask] = -float('inf')

        # 3. Dominant Negative Class (max over C)
        r_neg, best_neg_c = r_neg_candidates.max(dim=1) # [B]
        
        # Extract raw similarity for regularization (s_neg)
        s_neg = s_best_k_val.gather(1, best_neg_c.unsqueeze(1)).squeeze(1) # [B]

        # -----------------------------------------------------
        # 5. Loss & Regularization
        # -----------------------------------------------------
        # MP-CBML Loss: Softplus(r_neg - r_pos)
        loss_main = F.softplus(r_neg - r_pos).mean()

        # Update stats (detached)
        if self.training:
            self.update_moving_averages(s_pos.detach().mean(), s_neg.detach().mean())

        # Regularization
        # xi is treated as a constant target (stop gradient implied by using buffer values)
        xi = (self.alpha * self.global_s_pos) + ((1 - self.alpha) * self.global_s_neg)
        
        # Reg Loss: max(0, s_neg - xi)
        loss_reg = F.relu(s_neg - xi).mean()

        total_loss = loss_main + (self.reg_weight * loss_reg)

        # -----------------------------------------------------
        # 6. Logging Helpers
        # -----------------------------------------------------
        with torch.no_grad():
            self.current_beta = beta.item()
            self.current_pos_sim = s_pos.mean().item()
            self.current_neg_sim = s_neg.mean().item()
            self.current_xi = xi.item()

        return total_loss