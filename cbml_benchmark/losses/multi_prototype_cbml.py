import torch
import torch.nn as nn
import torch.nn.functional as F

from cbml_benchmark.losses.registry import LOSS

@LOSS.register('multi_prototype_cbml')
class MultiPrototypeCBMLLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiPrototypeCBMLLoss, self).__init__()
        self.num_classes = cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES
        self.prototype_per_class = cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS
        self.embed_dim = cfg.MODEL.HEAD.DIM
        self.device = torch.device(cfg.MODEL.DEVICE)

        # theta = log(beta) and beta = 1 / sigma_sq
        self.theta = nn.Parameter(
            torch.tensor(2.0, device=self.device)
        )

        # Prototypes: [num_classes, prototype_per_class, embed_dim]
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

        # Weights: [num_classes, prototype_per_class]
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class, device=self.device) / self.prototype_per_class
        )

        # Class priors: [num_classes]
        self.class_priors = nn.Parameter(
            torch.tensor(cfg.LOSSES.MULTI_PROTOTYPE_CBML.CLASS_PRIORS, device=self.device),
            requires_grad=False
        )

    def set_prototypes(self, prototypes):
        with torch.no_grad():
            if prototypes.device != self.device:
                prototypes = prototypes.to(self.device)
            self.prototypes.data = prototypes
            torch.cuda.empty_cache()

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

        # Fixed beta (as in your code)
        # beta = 7.37
        beta = torch.exp(self.theta)

        # Precompute per-sample work
        idx = torch.arange(B, device=self.device)
        pos_proto_sims_all = proto_embd_sim[idx, targets]  # [B, K]
        pos_best_idx_all = pos_proto_sims_all.argmax(dim=-1)  # [B]
        all_flat = proto_embd_sim.view(B, -1)  # [B, C*K]
        class_offsets = (targets.unsqueeze(1) * K) + torch.arange(K, device=self.device).unsqueeze(0)  # [B, K]
        neg_mask_flat = torch.ones_like(all_flat, dtype=torch.bool)
        neg_mask_flat[idx.unsqueeze(1), class_offsets] = False

        total_loss = 0.0

        for i in range(B):
            x = normalized_embds[i]  # [D]
            y = int(targets[i].item())  # class index

            # Positive prototype info
            pos_sims = pos_proto_sims_all[i]  # [K]
            best_pos_idx = int(pos_best_idx_all[i].item())
            pos_sim = pos_sims[best_pos_idx]
            pos_proto = normalized_protos[y, best_pos_idx]
            w_pos = normalized_weights[y, best_pos_idx]
            prior_pos = self.class_priors[y]

            # All negative classes: select hardest prototype per class
            neg_mask = torch.ones(C, dtype=torch.bool, device=self.device)
            neg_mask[y] = False
            neg_classes = torch.arange(C, device=self.device)[neg_mask]  # [C-1]
            max_sims_per_class = torch.max(proto_embd_sim[i], dim=1)[0]  # [C]
            best_idx_per_class = torch.argmax(proto_embd_sim[i], dim=1)  # [C]
            neg_sims_per_class = max_sims_per_class[neg_mask]  # [C-1]
            neg_w_per_class = normalized_weights[neg_classes, best_idx_per_class[neg_classes]]  # [C-1]
            neg_prior_per_class = self.class_priors[neg_classes]  # [C-1]

            # Sim term: beta * pos_sim - log(sum(exp(beta * neg_sim)))
            neg_exp_sum = torch.sum(torch.exp(beta * neg_sims_per_class))
            sim_term = beta * pos_sim - torch.log(neg_exp_sum + 1e-9)

            # Bias term: log(prior_pos * w_pos) - log((1/M) * sum(prior_neg * w_neg))
            avg_neg_prior_w = torch.mean(neg_prior_per_class * neg_w_per_class)
            eps = 1e-9
            bias_term = torch.log(prior_pos * w_pos + eps) - torch.log(avg_neg_prior_w + eps)

            total_loss += (sim_term + bias_term)

        mpcbml_loss = - total_loss / B

        loss = mpcbml_loss
        return loss