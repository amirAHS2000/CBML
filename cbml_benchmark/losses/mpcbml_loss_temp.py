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

        # self.hyper_weight = getattr(cfg.LOSSES.MPCBML_LOSS, 'GAMMA_REG', 0.2)
        # self.reg_weight = getattr(cfg.LOSSES.MPCBML_LOSS, 'LAMBDA_REG', 1.0)
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
    
    def forward(self, embeddings, targets):

        assert embeddings.size(0) == targets.size(0), \
            f"feats.size(0): {embeddings.size(0)} is not equal to labels.size(0): {targets.size(0)}"

        if embeddings.device != self.prototypes.device:
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)

        # ------------- Computing the regularization term (based on original CBML implementation) -----------------
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
        # --------------------------------------------------------------------------------------------------------

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
        distances = torch.cdist(z, flat_protos, p=2.0)
        # Negate distance to use as similarity (so argmax still finds the closest)
        sims = -distances.view(B, C, K)

        batch_loss = []
        # selected_pos_cls = []
        # selected_pos_idx = []
        # selected_neg_cls = []
        # selected_neg_idx = []

        for i in range(B):
            
            target_class = targets[i].item()
            pos_sims = sims[i, target_class, :] # [K]
            pos_weights = W[target_class] # [K]
            pos_priors = self.class_priors[target_class] # [1]

            # select best positive based on max(w * sim)
            pos_score = pos_weights * pos_sims # [K]

            # TODO:
            pos_score = torch.log(pos_weights) + torch.log(pos_sims)

            best_pos_proto_idx = torch.argmax(pos_score).item()
            best_pos_proto = P[target_class, best_pos_proto_idx]
            best_pos_weight = W[target_class, best_pos_proto_idx]

            # selected_pos_cls.append(target_class)
            # selected_pos_idx.append(best_pos_proto_idx)
            self.pos_proto_counts[target_class, best_pos_proto_idx] += 1

            # negative selection
            neg_mask = torch.arange(C, device=self.device) != target_class
            neg_class_indices = torch.where(neg_mask)[0] # absolute class indices

            neg_sims = sims[i, neg_mask, :] # [C - 1, K]
            neg_weights = W[neg_mask] # [C - 1, K]
            neg_priors = self.class_priors[neg_mask] # [C - 1]

            # select best negative based on max(class_prior * w * sim)
            neg_score = neg_priors.unsqueeze(1) * neg_weights * neg_sims # [C - 1, K]

            # TODO:
            neg_score = torch.log(neg_priors.unsqueeze(1)) + torch.log(neg_weights) + torch.log(neg_sims)

            flat_max_idx = torch.argmax(neg_score)
            best_neg_class_idx_masked = flat_max_idx // K
            best_neg_proto_idx = flat_max_idx % K
            
            # map back to absolute class index
            best_neg_class_idx = neg_class_indices[best_neg_class_idx_masked].item()
            best_neg_proto_idx = best_neg_proto_idx.item()

            best_neg_proto = P[best_neg_class_idx, best_neg_proto_idx]
            best_neg_weight = W[best_neg_class_idx, best_neg_proto_idx]
            best_neg_class_prior = self.class_priors[best_neg_class_idx]

            # selected_neg_cls.append(best_neg_class_idx)
            # selected_neg_idx.append(best_neg_proto_idx)
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
        
        loss = sum(batch_loss) / B

        # Index extraction
        # with torch.no_grad():
        #     self.last_best_pos_indices = {
        #         'c': torch.tensor(selected_pos_cls),
        #         'k': torch.tensor(selected_pos_idx)
        #     }
        #     self.last_best_neg_indices = {
        #         'c': torch.tensor(selected_neg_cls),
        #         'k': torch.tensor(selected_neg_idx)
        #     }

        return loss

    # def get_last_best_indices(self):
    #     """Returns dictionaries of absolute (class, k) indices for the last batch."""
    #     return self.last_best_pos_indices, self.last_best_neg_indices
