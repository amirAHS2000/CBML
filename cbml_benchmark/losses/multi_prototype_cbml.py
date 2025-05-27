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

        # initializing parameters
        # theta = log(beta) and beta = 1 / sigma_sq
        self.theta = nn.Parameter(
            torch.tensor(0.0, device=self.device)
        )

        # prototypes: [num_classes, prototype_per_class, embed_dim]
        self.prototypes = nn.Parameter(
            torch.randn(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )
        # self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=2)

        # weights: [num_classes, prototype_per_class]
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class, device=self.device) / self.prototype_per_class
        )

        # class priors: uniform for simplicity [num_classes]
        self.class_priors = nn.Parameter(
            torch.ones(self.num_classes, device=self.device) / self.num_classes, requires_grad=False
        )

    def forward(self, embeddings, targets):
        embeddings = embeddings.to(self.device)
        targets = targets.to(self.device)

        pos_thresh = 1e-5
        batch_size = embeddings.size(0)
        
        # normalization
        normalized_embds = F.normalize(embeddings, p=2, dim=1) # [B, D]
        normalized_protos = F.normalize(self.prototypes, p=2, dim=2) # [C, K, D]
        # weights = F.softmax(self.weights, dim=1) # [C, K]
        weights = self.weights - (self.weights.sum(dim=1, keepdim=True) - 1) / self.prototype_per_class

        # similarity matrices
        embd_embd_sim = torch.matmul(normalized_embds, normalized_embds.t())
        proto_embd_sim = torch.matmul(normalized_embds, normalized_protos.view(-1, self.embed_dim).t())
        proto_embd_sim = proto_embd_sim.view(
            batch_size, self.num_classes, self.prototype_per_class
        )

        # regularization
        regularization_term = list()
        for i in range(batch_size):
            # computing the regularization term
            pos_pair_ = embd_embd_sim[i][targets == targets[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - pos_thresh]
            neg_pair_ = embd_embd_sim[i][targets != targets[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            mean_ = self.hyper_weight * torch.mean(pos_pair_) + (1 - self.hyper_weight) * torch.mean(neg_pair_)
            # sigma_ = torch.mean(torch.sum(torch.pow(neg_pair_ - mean_, 2)))
            sigma_ = torch.mean(torch.pow(neg_pair_ - mean_, 2))
            regularization_term.append((sigma_))
        
        if len(regularization_term) > 0:
            regularization_term = torch.stack(regularization_term).mean()
        else:
            regularization_term = torch.tensor(0.0, device=self.device)
        

        total_loss = 0.0
        # for each sample, find pos and neg prototype and compute its loss
        for i in range(batch_size):
            x = normalized_embds[i] # [D]
            y = targets[i].item() # true class

            # extract the [k] similarities for the true class y
            pos_sim_protos = proto_embd_sim[i, y, :]

            # find the index of the best positive prototype
            best_pos_proto_idx = torch.argmax(pos_sim_protos).item() # python int

            # grab the its similarity, weight, and prior
            pos_sim = pos_sim_protos[best_pos_proto_idx]
            pos_proto = normalized_protos[y, best_pos_proto_idx]
            w_pos = weights[y, best_pos_proto_idx]
            prior_pos = self.class_priors[y]

            # for negatives, mask out the true class
            # take all classes except y, flatten them along K
            all_sim = proto_embd_sim[i] # [C, K]
            mask = torch.ones_like(all_sim, dtype=torch.bool)
            mask[y, :] = False # zero out true class row

            # apply the mask and find the flat argmax
            neg_sim_flat = all_sim.masked_fill(~mask, float('-inf')).view(-1) # [C * K]
            neg_idx_flat = torch.argmax(neg_sim_flat).item() # int in [0 .. C * K)

            # convert that flat index back to (class, prototype) via divmod
            best_neg_class, best_neg_proto_idx = divmod(neg_idx_flat, self.prototype_per_class)

            # grab its similarity, weight, and prior
            neg_sim = all_sim[best_neg_class, best_neg_proto_idx]
            neg_proto = normalized_protos[best_neg_class, best_neg_proto_idx]
            w_neg = weights[best_neg_class, best_neg_proto_idx]
            prior_neg = self.class_priors[best_neg_class]

            # build loss terms
            # similarity
            sim_term = torch.exp(self.theta) * (pos_sim - neg_sim)

            # bias
            eps = 1e-9
            # assume prior_pos, w_pos, prior_neg, w_neg are PyTorch scalars/tensors on the right device
            bias_term = (torch.log(prior_pos + eps) + torch.log(w_pos + eps)
                         - torch.log(prior_neg + eps) - torch.log(w_neg + eps))
            total_loss += (sim_term + bias_term)

        # average and add regularization
        loss = -total_loss / batch_size
        loss = loss + self.reg_weight * regularization_term
        return loss
