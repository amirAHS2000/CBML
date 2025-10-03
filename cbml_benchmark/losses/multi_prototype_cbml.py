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
            self.prototypes.data = prototypes
            torch.cuda.empty_cache()

    def show_theta(self):
        return self.theta.item()

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

        # precompute flat indices of prototypes that belong to the true class
        class_offsets = (targets.unsqueeze(1) * K) + torch.arange(K, device=self.device).unsqueeze(0) # [B, K]

        # build neg_mask_flat: True where negative prototypes: [B, C*K]
        neg_mask_flat = torch.ones_like(all_flat, dtype=torch.bool)
        # set positions of positive class prototypes to False
        neg_mask_flat[idx.unsqueeze(1), class_offsets] = False

        total_loss = 0.0
        mvc_terms = []

        for i in range(B):
            x = normalized_embds[i] # [D]
            y = int(targets[i].item()) # class index

            # positive prototype info
            pos_sims = pos_proto_sims_all[i] # [K]
            best_pos_idx = int(pos_best_idx_all[i].item())
            pos_sim = pos_sims[best_pos_idx]
            pos_proto = normalized_protos[y, best_pos_idx]
            w_pos = weights[y, best_pos_idx]
            prior_pos = self.class_priors[y]

            # ----- Modified Negative Selection -----
            # Compute max similarity per negative class, exclude true class y
            neg_class_sims = proto_embd_sim[i].clone() # [C, K]
            neg_class_sims[y] = -float('inf') # Mask true class
            # Max sim per class over K prototypes: [C]
            max_sims_per_class, _ = torch.max(neg_class_sims, dim=1)
            # Find hardest negative class (excluding true class)
            neg_class = torch.argmax(max_sims_per_class) # scalar
            # Get similarities for that class's prototypes: [K]
            neg_class_proto_sims = proto_embd_sim[i, neg_class] # [K]
            # Find best prototype in hardest class
            neg_proto_idx = torch.argmax(neg_class_proto_sims) # scalar
            neg_sim = neg_class_proto_sims[neg_proto_idx] # scalar
            neg_proto = normalized_protos[neg_class, neg_proto_idx]
            w_neg = weights[neg_class, neg_proto_idx]
            prior_neg = self.class_priors[neg_class]
            # -------------------------------------

            # CBML terms
            sim_term = beta * (pos_sim - neg_sim)
            eps = 1e-9
            bias_term = (torch.log(prior_pos + eps) + torch.log(w_pos + eps)
                         - torch.log(prior_neg + eps) - torch.log(w_neg + eps))
            total_loss += (sim_term + bias_term)

            # Prototype-based MVC
            pos_mean = pos_sims.mean()
            neg_flat_row = all_flat[i]
            neg_mask_row = neg_mask_flat[i]
            neg_candidates = neg_flat_row[neg_mask_row]
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

        mpcbml_loss = - total_loss / B
        mvc_L2 = torch.stack(mvc_terms).mean() if len(mvc_terms) > 0 else torch.tensor(0.0, device=self.device)

        loss = mpcbml_loss + self.reg_weight * mvc_L2
        return loss
