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

        self.register_buffer('pos_proto_counts', torch.zeros(self.num_classes, self.prototype_per_class, dtype=torch.long))
        self.register_buffer('neg_proto_counts', torch.zeros(self.num_classes, self.prototype_per_class, dtype=torch.long))

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
    
    @torch.no_grad()
    def compute_dominant_negative_stats(self, embeddings, targets):
        """
        Compute evaluation-only query-to-prototype geometry statistics using
        the same positive/negative prototype-selection rules as forward().

        This function does not participate in gradient computation and does not
        modify prototype-selection counters or any training state.
        """
        embeddings = embeddings.to(self.device)
        targets = targets.to(self.device).long()

        z = embeddings
        P = self.prototypes
        C, K, D = P.shape
        B = z.shape[0]

        flat_protos = P.view(C * K, D)

        logits = (
            z @ flat_protos.T
            - 0.5 * flat_protos.pow(2).sum(dim=1)
        ).view(B, C, K)

        weights = torch.clamp(self.weights, min=1e-9)
        priors = torch.clamp(self.class_priors, min=1e-9)

        rows = torch.arange(B, device=self.device)

        # ---------------------------------------------------------------
        # Dominant positive prototype
        # ---------------------------------------------------------------
        pos_score = (
            logits[rows, targets, :]
            + torch.log(weights[targets])
        )

        best_pos_idx = pos_score.argmax(dim=1)
        best_pos = P[targets, best_pos_idx]

        # ---------------------------------------------------------------
        # Dominant negative prototype
        # Same selection rule as forward(), excluding the target class.
        # ---------------------------------------------------------------
        neg_score = (
            logits
            + torch.log(weights).unsqueeze(0)
            + torch.log(priors).view(1, C, 1)
        )

        neg_score[rows, targets, :] = -torch.inf

        flat_neg_idx = neg_score.view(B, C * K).argmax(dim=1)
        best_neg = flat_protos[flat_neg_idx]

        # ---------------------------------------------------------------
        # Distances
        # ---------------------------------------------------------------
        pos_dist = torch.linalg.vector_norm(z - best_pos, dim=1)
        neg_dist = torch.linalg.vector_norm(z - best_neg, dim=1)

        # Positive-negative distance gap:
        # positive when the dominant negative is farther than the
        # dominant positive.
        gap = neg_dist - pos_dist

        result = {
            "pos_mean": pos_dist.mean(),
            "neg_mean": neg_dist.mean(),
            "neg_min": neg_dist.min(),
            "neg_p05": torch.quantile(neg_dist, 0.05),
            "neg_p10": torch.quantile(neg_dist, 0.10),
            "gap_mean": gap.mean(),
            "gap_p10": torch.quantile(gap, 0.10),
        }

        return {
            key: value.detach()
            for key, value in result.items()
        }

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

            best_neg_proto = P[best_neg_class_idx, best_neg_proto_idx].detach()
            best_neg_weight = W[best_neg_class_idx, best_neg_proto_idx]
            best_neg_class_prior = self.class_priors[best_neg_class_idx]
            # TODO: so the embedding network can still learn from the hard negative.

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

        if len(batch_loss) == 0:
            return torch.zeros(1, requires_grad=True).cuda()
        
        main_loss = torch.stack(batch_loss).mean()
        return main_loss