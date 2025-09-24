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

        self.hyper_weight = cfg.LOSSES.MULTI_PROTOTYPE_CBML.HYPER_WEIGHT
        self.reg_weight = cfg.LOSSES.MULTI_PROTOTYPE_CBML.REG_WEIGHT
        # self.mvc_topk = cfg.LOSSES.MULTI_PROTOTYPE_CBML.MVC_TOPK
        self.mvc_topk = 20

        # initializing parameters
        # theta = log(beta) and beta = 1 / sigma_sq
        self.theta = nn.Parameter(
            torch.tensor(2.0, device=self.device)
        )
        
        # prototypes: [num_classes, prototype_per_class, embed_dim]
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

        # weights: [num_classes, prototype_per_class]
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class, device=self.device) / self.prototype_per_class
        )

        # class priors: uniform for simplicity [num_classes]
        self.class_priors = nn.Parameter(
            torch.tensor(cfg.LOSSES.MULTI_PROTOTYPE_CBML.CLASS_PRIORS, device=self.device),
            requires_grad=False
        )

    def set_prototypes(self, prototypes):
        with torch.no_grad():
            if prototypes.device != self.device:
                prototypes = prototypes.to(self.device)
            self.prototypes.data = prototypes # use assignment instead of copy_

            # clear any cached memory
            torch.cuda.empty_cache()

    def forward(self, embeddings, targets):
        # device consistency
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        if targets.device != self.device:
            targets = targets.to(self.device)

        # pos_thresh = 1e-5
        batch_size = embeddings.size(0)
        B = batch_size
        C = self.num_classes
        K = self.prototype_per_class
        D = self.embed_dim

        # normalize embeddings & prototypes
        normalized_embds = F.normalize(embeddings, p=2, dim=1) # [B, D]
        normalized_protos = F.normalize(self.prototypes, p=2, dim=2) # [C, K, D]

        # normalize weights
        weights = F.softmax(self.weights, dim=1) # [C, K]

        # prototype-embedding similarities: [B, C, K]
        proto_embd_sim = torch.matmul(normalized_embds, normalized_protos.view(-1, D).t())
        proto_embd_sim = proto_embd_sim.view(B, C, K)

        # learnable parameter (1 / sigma_squared)
        beta = torch.exp(self.theta)

        # ----- precompute some heavy per-sample work -----
        # per-sample positive prototype similarities: [B, K]
        idx = torch.arange(B, device=self.device)
        pos_proto_sims_all = proto_embd_sim[idx, targets] # [B, K]

        # best positive prototype index per sample: [B] (tensor of ints)
        pos_best_idx_all = pos_proto_sims_all.argmax(dim=-1) # [B]

        # flatten all prototypes per sample: [B, C*K]
        all_flat = proto_embd_sim.view(B, -1) # [B, C*K]

        # precompute flat indices of prototypes that belong to the true class for each sample
        # for class j, prototypes indices in flattened row are j*K + [0..K-1]
        # compute flat indices to exclude (shape [B, K])
        class_offsets = (targets.unsqueeze(1) * K) + torch.arange(K, device=self.device).unsqueeze(0) # [B, K]

        # build neg_mask_flat: True where negative prototypes: [B, C*K]
        neg_mask_flat = torch.ones_like(all_flat, dtype=torch.bool)
        # set positions of positive class prototypes to False
        neg_mask_flat[idx.unsqueeze(1), class_offsets] = False

        # loop: per-sample CBML + per-sample MVC (using precomputed slices)
        total_loss = 0.0
        mvc_terms = []

        for i in range(B):
            x = normalized_embds[i] # [D]
            y = int(targets[i].item()) # class index

            # positive prototype info
            pos_sims = pos_proto_sims_all[i] # [K] tensor
            best_pos_idx = int(pos_best_idx_all[i].item())
            pos_sim = pos_sims[best_pos_idx]
            pos_proto = normalized_protos[y, best_pos_idx] # [D]
            w_pos = weights[y, best_pos_idx] # tensor scalar
            prior_pos = self.class_priors[y] # tensor scalar

            # negative prototype (hardest) using flattened arr & mask
            neg_flat_row = all_flat[i] # [C*K]
            neg_mask_row = neg_mask_flat[i] # [C*K] boolean
            # select negatives (this produces a 1D tensor of length: C*K - K)
            neg_candidates = neg_flat_row[neg_mask_row] # [num_neg]
            # fastest way to get hardest negative:
            neg_val, neg_idx_in_candidates = torch.max(neg_candidates, dim=0)
            neg_sim = neg_val # scalar tensor

            # map the index in neg_candidates back to class/proto index:
            # We need the flat index in all_flat: find where neg_mask_row is True and pick that pos
            # To avoid .nonzero() overhead every iteration, we can compute the flat index directly:
            # Get the boolean mask indices (once) — but we didn't store them per-sample.
            true_positions = torch.nonzero(neg_mask_row, as_tuple=False).squeeze(1) # [num_neg]
            flat_neg_idx = int(true_positions[neg_idx_in_candidates].item()) # flat index in [0, C*K)
            neg_class = flat_neg_idx // K
            neg_proto_idx = flat_neg_idx % K

            neg_proto = normalized_protos[neg_class, neg_proto_idx]
            w_neg = weights[neg_class, neg_proto_idx]
            prior_neg = self.class_priors[neg_class]

            # CBML terms
            sim_term = beta * (pos_sim - neg_sim)
            eps = 1e-9
            bias_term = (torch.log(prior_pos + eps) + torch.log(w_pos + eps)
                         - torch.log(prior_neg + eps) - torch.log(w_neg + eps))
            total_loss += (sim_term + bias_term)

            # ----- Prototype-based MVC -----
            # pos_mean: mean over pos_sims
            pos_mean = pos_sims.mean()

            # neg_used: either top-k from neg_candidates or all
            if getattr(self, "mvc_topk", None) is not None and 0 < self.mvc_topk < neg_candidates.numel():
                neg_used_vals, _ = torch.topk(neg_candidates, k=self.mvc_topk)
                neg_used = neg_used_vals
            else:
                neg_used = neg_candidates
            
            if neg_used.numel() > 0:
                neg_mean = neg_used.mean()
                xi = self.hyper_weight * pos_mean + (1.0 - self.hyper_weight) * neg_mean
                diffs = neg_used - xi
                L2_i = torch.mean(diffs * diffs)
            else:
                L2_i = torch.tensor(0.0, device=self.device)
            
            mvc_terms.append(L2_i)

        # average and combine
        mpcbml_loss = - total_loss / B
        mvc_L2 = torch.stack(mvc_terms).mean() if len(mvc_terms) > 0 else torch.tensor(0.0, device=self.device)

        loss = mpcbml_loss + self.reg_weight * mvc_L2
        return loss
