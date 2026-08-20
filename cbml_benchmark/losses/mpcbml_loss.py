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
        # Be updated with Lagrange Multiplier
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

        # regularization term parameters
        self.hyper_weight = getattr(cfg.LOSSES.MPCBML_LOSS, 'GAMMA_REG', 0.2)
        self.reg_weight = getattr(cfg.LOSSES.MPCBML_LOSS, 'LAMBDA_REG', 0.3)
        self.ma_momentum = getattr(cfg.LOSSES.MPCBML_LOSS, 'MA_MOMENTUM', 0.9)
        self.register_buffer('global_ma_pos', torch.tensor(0.0))
        self.register_buffer('global_ma_neg', torch.tensor(0.0))
        # bias-correction step counter for the EMA (Adam-style warm-up fix)
        self.register_buffer('ma_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('pos_proto_counts', torch.zeros(self.num_classes, self.prototype_per_class, dtype=torch.long))
        self.register_buffer('neg_proto_counts', torch.zeros(self.num_classes, self.prototype_per_class, dtype=torch.long))

    @torch.no_grad()
    def update_moving_average(self, current_pos_mean, current_neg_mean):
        """
        Update the global estimates using Exponential Moving Average (EMA).
        No gradients flow through this update. Inputs are detached defensively
        even though the decorator already disables grad tracking.
        """
        current_pos_mean = current_pos_mean.detach()
        current_neg_mean = current_neg_mean.detach()

        self.global_ma_pos = (self.ma_momentum * self.global_ma_pos + (1 - self.ma_momentum) * current_pos_mean)
        self.global_ma_neg = (self.ma_momentum * self.global_ma_neg + (1 - self.ma_momentum) * current_neg_mean)
        self.ma_step += 1

    def _bias_corrected_ma(self):
        """Adam-style bias correction so xi isn't badly underestimated near step 0."""
        if self.ma_step.item() == 0:
            return self.global_ma_pos, self.global_ma_neg
        correction = 1 - (self.ma_momentum ** self.ma_step.item())
        correction = max(correction, 1e-9)
        return self.global_ma_pos / correction, self.global_ma_neg / correction
    
    @torch.no_grad()
    def set_prototypes_and_weights(self, prototypes, cluster_sizes):
        prototypes = prototypes.to(self.device)
        self.prototypes.copy_(prototypes)

        # Save a frozen copy of the initial prototypes for monitoring
        # self.initial_prototypes = prototypes

        # if cluster_sizes is not None and \
        #     cluster_sizes.shape == (self.num_classes, self.prototype_per_class):
        #     cluster_sizes = cluster_sizes.to(self.device).float()
        #     normalized_weights = cluster_sizes / (cluster_sizes.sum(dim=1, keepdim=True) + 1e-9)
        #     self.weights.copy_(normalized_weights)
        # else:
        #     # Fallback to uniform
        #     self.weights.fill_(1.0 / self.prototype_per_class)
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

        assert embeddings.size(0) == targets.size(0), \
            f"feats.size(0): {embeddings.size(0)} is not equal to labels.size(0): {targets.size(0)}"

        if embeddings.device != self.prototypes.device:
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)

        z = embeddings # [B, D]
        P = self.prototypes # [C, K, D]
        # P = F.normalize(self.prototypes, p=2, dim=2)
        W = self.weights # [C, K]
        B = z.shape[0] # batch size
        C = self.num_classes
        K = self.prototype_per_class
        beta = torch.exp(self.theta)
        eps = 1e-9

        # Compute all similarities [B, C, K]
        flat_protos = P.view(C * K, -1) # [C * K, D]
        # Calculate Euclidean distance [B, C * K]
        logits = z @ flat_protos.T - 0.5 * (flat_protos.norm(p=2, dim=1) ** 2) # [B, C * K]
        logits = logits.view(B, C, K)

        batch_loss = []
        current_pos_dist = []
        current_neg_dist = []

        for i in range(B):
            
            target_class = targets[i].item()
            pos_weights = W[target_class] # [K]
            pos_priors = self.class_priors[target_class] # [1]
            pos_score = torch.log(pos_weights) + logits[i, target_class, :]

            best_pos_proto_idx = torch.argmax(pos_score).item()
            best_pos_proto = P[target_class, best_pos_proto_idx]
            best_pos_weight = W[target_class, best_pos_proto_idx]

            self.pos_proto_counts[target_class, best_pos_proto_idx] += 1

            # negative selection
            neg_mask = torch.arange(C, device=self.device) != target_class
            neg_class_indices = torch.where(neg_mask)[0] # absolute class indices

            neg_weights = W[neg_mask] # [C - 1, K]
            neg_priors = self.class_priors[neg_mask] # [C - 1]
            neg_score = torch.log(neg_priors.unsqueeze(1)) + torch.log(neg_weights) + logits[i, neg_mask, :]

            flat_max_idx = torch.argmax(neg_score)
            best_neg_class_idx_masked = flat_max_idx // K
            best_neg_proto_idx = flat_max_idx % K
            
            # map back to absolute class index
            best_neg_class_idx = neg_class_indices[best_neg_class_idx_masked].item()
            best_neg_proto_idx = best_neg_proto_idx.item()

            best_neg_proto = P[best_neg_class_idx, best_neg_proto_idx]
            best_neg_weight = W[best_neg_class_idx, best_neg_proto_idx]
            best_neg_class_prior = self.class_priors[best_neg_class_idx]

            self.neg_proto_counts[best_neg_class_idx, best_neg_proto_idx] += 1

            current_loss = F.softplus(
                (torch.log(best_neg_class_prior) - torch.log(pos_priors)) +
                (torch.log(best_neg_weight) - torch.log(best_pos_weight)) +
                (beta * ((embeddings[i] @ best_neg_proto) -
                         ((1/2) * (best_neg_proto @ best_neg_proto)) -
                         (embeddings[i] @ best_pos_proto) +
                         ((1/2) * (best_pos_proto @ best_pos_proto))
                         ))
            )

            batch_loss.append(current_loss)

            # regularization term components
            # -- positive side: monitoring only, no gradient needed
            with torch.no_grad():
                pos_dist = torch.norm(embeddings[i] - best_pos_proto, p=2)
                current_pos_dist.append(pos_dist)

            # -- negative side: MUST stay attached to the graph, this is what
            #    the regularizer actually pushes on
            neg_dist = torch.norm(embeddings[i] - best_neg_proto, p=2)
            current_neg_dist.append(neg_dist)

        # -------------- regularization term ----------------
        # Keep each dominant-negative distance attached to the graph.
        # The regularizer is applied sample-wise, while the batch mean is used
        # only to update the global EMA statistics.
        current_neg_dist_tensor = torch.stack(current_neg_dist)
        current_neg_mean = current_neg_dist_tensor.mean()

        with torch.no_grad():
            current_pos_mean = torch.stack(current_pos_dist).mean()

            # Compute xi from the EMA state *before* updating it with this
            # batch's statistics, so the threshold does not leak current-batch info.
            pos_ma, neg_ma = self._bias_corrected_ma()
            xi = (self.hyper_weight * pos_ma + (1 - self.hyper_weight) * neg_ma)

            if self.training:
                self.update_moving_average(current_pos_mean, current_neg_mean)

        # Sample-wise squared hinge penalty:
        #   l_reg(i) = max(0, xi - d_i^-)^2
        # This penalizes each collapsed dominant negative prototype directly,
        # rather than allowing violations to cancel out through a batch mean.
        reg_loss = torch.clamp(xi - current_neg_dist_tensor, min=0.0).pow(2).mean()
        # ----------------------------------------------------

        self.latest_xi = xi.detach()
        self.latest_current_neg_mean = current_neg_mean.detach()

        if len(batch_loss) == 0:
            return torch.zeros(1, requires_grad=True).cuda()
        
        main_loss = sum(batch_loss) / B
        loss = main_loss + self.reg_weight * reg_loss

        return loss