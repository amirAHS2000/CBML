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

        self.gamma_reg = getattr(cfg.LOSSES.MPCBML_LOSS, 'GAMMA_REG', 0.2)
        self.lambda_reg = getattr(cfg.LOSSES.MPCBML_LOSS, 'LAMBDA_REG', 10.0)

        theta_is_learnable = getattr(cfg.LOSSES.MPCBML_LOSS, 'THETA_IS_LEARNABLE', False)
        init_theta = getattr(cfg.LOSSES.MPCBML_LOSS, 'INIT_THETA', 1.0)        
        if theta_is_learnable:
            self.theta = nn.Parameter(torch.tensor(init_theta, device=self.device))
        else:
            self.theta = init_theta
        
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
        with torch.no_grad():
            self.prototypes = F.normalize(self.prototypes, p=2, dim=2)

    @torch.no_grad()
    def compute_proto_stats(prototypes: torch.Tensor, initial_prototypes: torch.Tensor = None):
        """
        Compute intra/inter distances, norms, and displacement of prototypes.
        Args:
            prototypes: [C, K, D] tensor of current prototypes
            initial_prototypes: [C, K, D] optional tensor for displacement measurement
        Returns:
            dict with intra, inter, norm, and (optional) displacement stats
        """
        protos = F.normalize(prototypes, p=2, dim=-1)
        C, K, D = protos.shape

        # --- Intra-class distances ---
        intra = torch.stack([
            torch.pdist(protos[c], p=2).mean()
            if K > 1 else torch.tensor(0., device=protos.device)
            for c in range(C)
        ])
        mean_intra = intra.mean().item()

        # --- Inter-class distances ---
        flat = protos.view(C * K, D)
        dist = torch.cdist(flat, flat, p=2)
        mask = torch.ones_like(dist, dtype=torch.bool)
        for c in range(C):
            mask[c*K:(c+1)*K, c*K:(c+1)*K] = False
        inter = dist[mask].mean().item()

        # --- Norms ---
        norms = protos.norm(dim=-1).mean().item()

        stats = {
            "mean_intra_dist": round(mean_intra, 5),
            "mean_inter_dist": round(inter, 5),
            "mean_proto_norm": round(norms, 5)
        }

        if initial_prototypes is not None:
            disp = torch.norm(protos - F.normalize(initial_prototypes, p=2, dim=-1), dim=-1).mean().item()
            stats["mean_displacement"] = round(disp, 5)

        return stats
    
    @torch.no_grad()
    def compute_weight_stats(weights: torch.Tensor):
        """
        Compute statistics over class-prototype weights.
        Args:
            weights: [C, K] raw weight tensor (before softmax)
        Returns:
            dict with entropy, variance, and dominance statistics
        """

        w = F.softmax(weights, dim=1)
        entropy = (-w * (w.clamp_min(1e-9)).log()).sum(dim=1)
        mean_entropy = entropy.mean().item()
        var_entropy = entropy.var(unbiased=False).item()
        weight_var = w.var(dim=1).mean().item()
        max_weight_mean = w.max(dim=1)[0].mean().item()

        return {
            "mean_entropy": round(mean_entropy, 5),
            "var_entropy": round(var_entropy, 5),
            "mean_weight_var": round(weight_var, 5),
            "mean_max_weight": round(max_weight_mean, 5),
        }

    def show_prototype_stats(self, initial_prototypes=None):
        if initial_prototypes is None and hasattr(self, "initial_prototypes"):
            initial_prototypes = self.initial_prototypes
        prototype_statistics = self.compute_proto_stats(self.prototypes.detach(), initial_prototypes)
        return prototype_statistics
    
    def show_weight_stats(self):
        weight_statistics = self.compute_weight_stats(self.weights.detach())
        return weight_statistics
    
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
    self._enforce_constraints()

    embeddings = embeddings.to(self.device)
    targets = targets.to(self.device)

    B = embeddings.size(0)
    C, K, D = self.num_classes, self.prototype_per_class, self.embed_dim
    eps = 1e-9

    # =====================================================
    # 0. Normalize embeddings
    # =====================================================
    z = F.normalize(embeddings, p=2, dim=1)  # [B, D]
    protos = self.prototypes  # [C, K, D]
    W = self.weights  # [C, K]

    # Get current beta
    beta = torch.exp(self.theta)
    self.current_beta = beta.item()

    # =====================================================
    # 1. Compute similarities
    # =====================================================
    sims = torch.matmul(z, protos.view(C*K, D).t()).view(B, C, K)  # [B,C,K]

    # =====================================================
    # 2. Compute log-probability contribution
    # =====================================================
    log_prob_contrib = torch.log(W.unsqueeze(0) + eps) + beta * sims  # [B, C, K]

    # =====================================================
    # 3. Masks
    # =====================================================
    y_onehot = F.one_hot(targets, num_classes=C).bool()  # [B,C]
    neg_mask = ~y_onehot  # [B,C]

    # =====================================================
    # 4. POSITIVE SELECTION
    # =====================================================
    pos_log_contrib = log_prob_contrib[y_onehot].view(B, K)  # [B,K]
    pos_raw = sims[y_onehot].view(B, K)  # [B,K]
    pos_w = W[targets]  # [B,K]
    prior_pos = self.class_priors[targets]  # [B]

    best_pos_idx = pos_log_contrib.argmax(dim=-1)  # [B]
    pos_sim = pos_raw[torch.arange(B), best_pos_idx]  # [B]
    w_pos = pos_w[torch.arange(B), best_pos_idx]  # [B]

    # =====================================================
    # 5. NEGATIVE SELECTION
    # =====================================================
    # Step 1: Extract info for all negative classes
    neg_log_contrib = log_prob_contrib[neg_mask].view(B, C-1, K)  # [B, C-1, K]
    neg_raw = sims[neg_mask].view(B, C-1, K)  # [B, C-1, K]
    
    # Expand weights and priors
    W_expanded = W.unsqueeze(0).expand(B, C, K)  # [B, C, K]
    neg_W = W_expanded[neg_mask].view(B, C-1, K)  # [B, C-1, K]
    
    class_priors_exp = self.class_priors.unsqueeze(0).expand(B, C)  # [B, C]
    neg_priors = class_priors_exp[neg_mask].view(B, C-1)  # [B, C-1]

    # Step 2: Find best prototype per negative class using log-probability criterion
    best_neg_log_contrib, best_neg_k = neg_log_contrib.max(dim=-1)  # [B, C-1]
    
    b_idx = torch.arange(B, device=self.device)
    
    # Get log-weights at best prototypes
    best_neg_log_w = torch.log(neg_W[b_idx, torch.arange(C-1), best_neg_k] + eps)  # [B, C-1]
    best_neg_raw_sim = neg_raw[b_idx, torch.arange(C-1), best_neg_k]  # [B, C-1]
    
    # Score = log(p(c)) + log(w_c^ℓ*) + β * s_c^ℓ*
    neg_class_scores = (torch.log(neg_priors + eps) + 
                        best_neg_log_w + 
                        beta * best_neg_raw_sim)  # [B, C-1]
    
    # Select dominant negative class
    best_neg_class = neg_class_scores.argmax(dim=-1)  # [B]

    # Extract selected negative prototype info
    best_neg_sim = neg_raw[b_idx, best_neg_class, best_neg_k[b_idx, best_neg_class]]  # [B]
    w_neg = neg_W[b_idx, best_neg_class, best_neg_k[b_idx, best_neg_class]]  # [B]
    prior_neg = neg_priors[b_idx, best_neg_class]  # [B]

    # Log selected similarities
    self.current_pos_sim = pos_sim.mean().item()
    self.current_neg_sim = best_neg_sim.mean().item()
    self.current_sim_margin = (pos_sim - best_neg_sim).mean().item()

    # =====================================================
    # 6. BAYESIAN LOSS
    # =====================================================
    log_A_pos = torch.log(prior_pos + eps) + torch.log(w_pos + eps) + beta * pos_sim  # [B]
    log_A_neg = torch.log(prior_neg + eps) + torch.log(w_neg + eps) + beta * best_neg_sim  # [B]
    
    log_denominator = torch.logsumexp(torch.stack([log_A_pos, log_A_neg], dim=0), dim=0)  # [B]
    
    mpcbml_loss = (-log_A_pos + log_denominator).mean()  # scalar

    # =====================================================
    # 7. COMPONENT LOGGING (for debugging)
    # =====================================================
    sim_term = beta * (pos_sim - best_neg_sim)  # [B]
    
    log_prior_pos = torch.log(prior_pos + eps)
    log_prior_neg = torch.log(prior_neg + eps)
    log_w_pos = torch.log(w_pos + eps)
    log_w_neg = torch.log(w_neg + eps)
    
    prior_bias = log_prior_pos - log_prior_neg  # [B]
    weight_bias = log_w_pos - log_w_neg  # [B]
    bias_term = prior_bias + weight_bias  # [B]
    
    self.sim_mpcbml_total = (-sim_term).mean().item()
    self.bias_mpcbml_total = (-bias_term).mean().item()
    self.prior_bias_total = prior_bias.mean().item()
    self.weight_bias_total = weight_bias.mean().item()
    self.mpcbml_total = mpcbml_loss.item()

    # =====================================================
    # 8. FINAL LOSS
    # =====================================================
    return mpcbml_loss
