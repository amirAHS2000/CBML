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

        # where N_NEGATIVES can be 1, 3, or -1 (to select all negatives)
        self.n_negatives = cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_NEGATIVES

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

        # flattened similarities and mask for ALL prototypes
        all_proto_sims_flat = proto_embd_sim.view(B, C * K) # shape: [B, C*K]

        total_loss = 0.0

        for i in range(B):
            # x = normalized_embds[i]  # [D]
            y = int(targets[i].item())  # class index

            # Positive prototype info
            pos_sims = pos_proto_sims_all[i]  # [K]
            best_pos_idx = int(pos_best_idx_all[i].item())
            pos_sim = pos_sims[best_pos_idx]
            # pos_proto = normalized_protos[y, best_pos_idx]
            w_pos = normalized_weights[y, best_pos_idx]
            prior_pos = self.class_priors[y]

            # ---------- select N most contributing negative prototypes ----------
            # # create a mask for all negative prototypes (C*K total)
            # neg_mask_flat = torch.ones(C * K, dtype=torch.bool, device=self.device)
            # # find the indices of the K positive prototypes in the C*K flattened list
            # pos_start_idx = y * K
            # pos_indices = torch.arange(pos_start_idx, pos_start_idx + K, device=self.device)
            # # set mask to False for positive prototypes
            # neg_mask_flat[pos_indices] = False

            # # get similarities and corresponding weights/priors for ALL negative prototypes
            # all_neg_sims = all_proto_sims_flat[i][neg_mask_flat] # shape: [C*K - K]
            # all_neg_weights = normalized_weights.view(-1)[neg_mask_flat] # shape: [C*K - K]
            # all_neg_priors_flat = self.class_priors.unsqueeze(1).repeat(1, K).view(-1)[neg_mask_flat] # shape: [C*K - K]

            # # calculate the "contribution" (beta * similarity + log(prior * weight))
            # neg_contribution = beta * all_neg_sims + torch.log(all_neg_priors_flat * all_neg_weights + 1e-9)

            # # select the top N contributing negatives
            # # if n_negatives > (C*K - K), select all
            # N = min(self.n_negatives, len(neg_contribution)) if self.n_negatives > 0 else len(neg_contribution)

            # # get the top N indices based on contribution
            # top_n_indices = torch.topk(neg_contribution, k=N, dim=0, sorted=False)[1]

            # # extract the similarities, weights, and priors for the Top N
            # top_n_neg_sims = all_neg_sims[top_n_indices] # [N]
            # top_n_neg_weights = all_neg_weights[top_n_indices] # [N]
            # top_n_neg_priors = all_neg_priors_flat[top_n_indices] # [N]

            # ---------- Select N hardest prototypes from N hardest negative classes ----------
            # identify all negative classes
            neg_mask = torch.ones(C, dtype=torch.bool, device=self.device)
            neg_mask[y] = False
            neg_classes = torch.arange(C, device=self.device)[neg_mask] # [C-1]

            # Find the hardest prototype (max sim) for EVERY class
            max_sim_per_class = torch.max(proto_embd_sim[i], dim=1)[0] # [C]
            best_idx_per_class = torch.argmax(proto_embd_sim[i], dim=1) # [C]

            # extract info for the hardest prototype in each negative class
            neg_class_sims = max_sim_per_class[neg_mask] # [C-1]
            neg_class_weights = normalized_weights[neg_classes, best_idx_per_class[neg_classes]] # [C-1]
            neg_class_priors = self.class_priors[neg_classes] # [C-1]

            # calculate contribution for the HARDEST prototype of each negative class
            neg_class_contribution = beta * neg_class_sims + torch.log(neg_class_priors * neg_class_weights + 1e-9)

            # select the Top N classes based on their hardest prototype's contribution
            N_neg_classes = len(neg_class_contribution)
            N = min(self.n_negatives, N_neg_classes) if self.n_negatives > 0 else N_neg_classes

            # get the top N indices (which correspond to the negative class list)
            top_n_class_indices = torch.topk(neg_class_contribution, k=N, dim=0, sorted=False)[1]

            # extract the similarities, weights, and priors for the TOP N hardest negative prototypes
            top_n_neg_sims = neg_class_sims[top_n_class_indices] # [N]
            top_n_neg_weights = neg_class_weights[top_n_class_indices] # [N]
            top_n_neg_priors = neg_class_priors[top_n_class_indices] # [N]

            # ---------- Loss calculation ----------
            # # Sim term: beta * pos_sim - log(sum(exp(beta * neg_sim)))
            # # Using only top_n_neg_sims:
            # neg_exp_sum = torch.sum(torch.exp(beta * top_n_neg_sims))
            # sim_term = beta * pos_sim - torch.log(neg_exp_sum + 1e-9)

            # # Bias term: log(prior_pos * w_pos) - log((1/N) * sum(prior_neg * w_neg))
            # # The denominator now averages over the N selected negatives
            # avg_neg_prior_w = torch.mean(top_n_neg_priors * top_n_neg_weights)
            # bias_term = torch.log(prior_pos * w_pos + 1e-9) - torch.log(avg_neg_prior_w + 1e-9)

            # total_loss += (sim_term + bias_term)

            # ---------- Loss calculation ----------
            # Sim term: log(sum(exp(beta * neg_sim)))
            neg_exp_sum = torch.sum(torch.exp(beta * top_n_neg_sims))
            sim_term = beta * pos_sim - torch.log(neg_exp_sum + 1e-9)

            # Bias term: log(prior_pos * w_pos) - log((1/N) * sum(prior_neg * w_neg))
            avg_neg_prior_w = torch.mean(top_n_neg_priors * top_n_neg_weights)
            eps = 1e-9
            bias_term = torch.log(prior_pos * w_pos + eps) - torch.log(avg_neg_prior_w + eps)

            total_loss += (sim_term + bias_term)

        mpcbml_loss = - total_loss / B
        loss = mpcbml_loss
        return loss
