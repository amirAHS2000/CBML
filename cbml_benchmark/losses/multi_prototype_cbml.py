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

        # self.hyper_weight = cfg.LOSSES.MULTI_PROTOTYPE_CBML.HYPER_WEIGHT
        # self.reg_weight = cfg.LOSSES.MULTI_PROTOTYPE_CBML.REG_WEIGHT
        
        # Prototypes: [num_classes, prototype_per_class, embed_dim]
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

        # Unnormalized weights (logits for softmax)
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class, device=self.device)
        )

        # Class priors: uniform for simplicity [num_classes]
        self.class_priors = nn.Parameter(
            torch.tensor(cfg.LOSSES.MULTI_PROTOTYPE_CBML.CLASS_PRIORS, device=self.device),
            requires_grad=False
        )
        
        # A fixed scale factor for logits, similar to beta in the original code
        self.scale = 7.37
        self.eps = 1e-9

    def set_prototypes(self, prototypes):
        with torch.no_grad():
            if prototypes.device != self.device:
                prototypes = prototypes.to(self.device)
            self.prototypes.data = prototypes
            torch.cuda.empty_cache()

    def forward(self, embeddings, targets):
        # Ensure tensors are on the correct device
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        if targets.device != self.device:
            targets = targets.to(self.device)

        batch_size = embeddings.size(0)
        B, C, K, D = batch_size, self.num_classes, self.prototype_per_class, self.embed_dim

        # Normalize embeddings & prototypes for cosine similarity calculation
        normalized_embds = F.normalize(embeddings, p=2, dim=1)
        normalized_protos = F.normalize(self.prototypes, p=2, dim=2)
        
        # Normalize weights to be positive and sum to 1 for each class using softmax
        normalized_weights = F.softmax(self.weights, dim=1)

        # Calculate all similarities at once for efficiency
        proto_embd_sim = torch.matmul(normalized_embds, normalized_protos.view(-1, D).t())
        proto_embd_sim = proto_embd_sim.view(B, C, K)
        
        # Tensor to store logits for each sample
        all_logits = torch.zeros(B, C, device=self.device)
        # mvc_terms = []

        # --- Loop-Based Calculation ---
        for i in range(B):
            # y = targets[i].item() # The true class for sample i

            # --- 1. Calculate Logits for the current sample ---
            # Similarities for this sample: [C, K]
            sims_i = proto_embd_sim[i]
            
            # Find the hardest prototype for each class
            hardest_proto_sims_i, hardest_proto_indices_i = torch.max(sims_i, dim=1) # [C]
            
            # Get the corresponding weights for these hardest prototypes
            hardest_weights_i = normalized_weights[torch.arange(C), hardest_proto_indices_i] # [C]
            
            # Construct the logits for sample i
            sim_term = self.scale * hardest_proto_sims_i
            bias_term = torch.log(hardest_weights_i + self.eps) + torch.log(self.class_priors + self.eps)
            logits_i = sim_term + bias_term # [C]
            
            all_logits[i] = logits_i
            
            # # --- 2. Calculate MVC Regularization for the current sample ---
            # if self.reg_weight > 0:
            #     pos_sims = sims_i[y] # [K]
            #     pos_mean = pos_sims.mean()

            #     neg_mask = torch.ones(C, dtype=torch.bool, device=self.device)
            #     neg_mask[y] = False
            #     neg_candidates = sims_i[neg_mask].view(-1) # All sims from negative classes

            #     if neg_candidates.numel() > 0:
            #         neg_mean = neg_candidates.mean()
            #         xi = self.hyper_weight * pos_mean + (1.0 - self.hyper_weight) * neg_mean
            #         diffs = neg_candidates - xi
            #         L2_i = torch.mean(diffs * diffs)
            #     else:
            #         L2_i = torch.tensor(0.0, device=self.device)
            #     mvc_terms.append(L2_i)

        # --- Calculate Final Loss Components ---

        # 1. Main loss from all collected logits
        ce_loss = F.cross_entropy(all_logits, targets)

        # 2. MVC regularization loss
        # mvc_L2 = torch.stack(mvc_terms).mean() if mvc_terms else torch.tensor(0.0, device=self.device)

        # 3. Combine all terms
        # loss = ce_loss + self.reg_weight * mvc_L2
        return ce_loss
