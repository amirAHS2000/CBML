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
        self.sigma_sq = cfg.LOSSES.MULTI_PROTOTYPE_CBML.SIGMA_SQ
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.hyper_weight = cfg.LOSSES.MULTI_PROTOTYPE_CBML.HYPER_WEIGHT
        self.reg_weight = cfg.LOSSES.MULTI_PROTOTYPE_CBML.REG_WEIGHT

        self.margin = cfg.LOSSES.MULTI_PROTOTYPE_CBML.MARGIN

        # initializing parameters
        # prototypes: [num_classes, prototype_per_class, embed_dim]
        self.prototypes = nn.Parameter(
            torch.randn(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=2)

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

        # pos_thresh = 1e-5
        batch_size = embeddings.size(0)
        
        # normalization
        normalized_embds = F.normalize(embeddings, p=2, dim=1).to(self.device) # [B, D]
        normalized_protos = F.normalize(self.prototypes, p=2, dim=2).to(self.device) # [C, K, D]
        weights = F.softmax(self.weights, dim=1).to(self.device) # [C, K]
        
        # similarity matrices
        # embd_embd_sim = torch.matmul(normalized_embds, normalized_embds.t())
        # proto_embd_sim = torch.matmul(normalized_embds, normalized_protos.view(-1, self.embed_dim).t())
        # proto_embd_sim = proto_embd_sim.view(
        #     batch_size, self.num_classes, self.prototype_per_class
        # )

        # # regularization
        # regularization_term = list()
        # for i in range(batch_size):
        #     # computing the regularization term
        #     pos_pair_ = embd_embd_sim[i][targets == targets[i]]
        #     pos_pair_ = pos_pair_[pos_pair_ < 1 - pos_thresh]
        #     neg_pair_ = embd_embd_sim[i][targets != targets[i]]

        #     if len(neg_pair_) < 1 or len(pos_pair_) < 1:
        #         continue

        #     mean_ = self.hyper_weight * torch.mean(pos_pair_) + (1 - self.hyper_weight) * torch.mean(neg_pair_)
        #     # sigma_ = torch.mean(torch.sum(torch.pow(neg_pair_ - mean_, 2)))
        #     sigma_ = torch.mean(torch.pow(neg_pair_ - mean_, 2))
        #     regularization_term.append((sigma_))
        
        # if len(regularization_term) > 0:
        #     regularization_term = torch.stack(regularization_term).mean()
        # else:
        #     regularization_term = torch.tensor(0.0, device=self.device)
        

        total_loss = 0.0
        # for each sample, find pos and neg prototype and compute its loss
        for i in range(batch_size):
            x = normalized_embds[i] # [D]
            y = targets[i].item() # true class

            # positive prototype selection
            best_pos_sim = None
            best_pos_proto = 0
            for k in range(self.prototype_per_class):
                proto = normalized_protos[y][k] # [D]
                sim = torch.dot(x, proto)

                if best_pos_sim is None or sim.item() > best_pos_sim.item():
                    best_pos_sim = sim
                    best_pos_proto = k

            mu_pos = normalized_protos[y][best_pos_proto] # [D]
            w_pos = weights[y, best_pos_proto]
            prior_pos = self.class_priors[y]
            pos_sim = best_pos_sim

            # negative prototype selection
            best_neg_sim = None
            best_neg_class = None
            best_neg_proto = 0
            for c in range(self.num_classes):
                if c == y:
                    continue
                for k in range(self.prototype_per_class):
                    proto = normalized_protos[c][k]
                    sim = torch.dot(x, proto)
                    if best_neg_sim is None or sim.item() > best_neg_sim.item():
                        best_neg_sim = sim
                        best_neg_class = c
                        best_neg_proto = k

            mu_neg = normalized_protos[best_neg_class][best_neg_proto] # [D]
            w_neg = weights[best_neg_class, best_neg_proto]
            prior_neg = self.class_priors[best_neg_class]
            neg_sim = best_neg_sim

            # build loss terms
            # similarity
            sim_term = (1.0 / self.sigma_sq) * (pos_sim - neg_sim)

            # bias
            eps = 1e-9
            # assume prior_pos, w_pos, prior_neg, w_neg are PyTorch scalars/tensors on the right device
            bias_term = (torch.log(prior_pos + eps) + torch.log(w_pos + eps)
                         - torch.log(prior_neg + eps) - torch.log(w_neg + eps))
            total_loss += (sim_term + bias_term)

        # average and add regularization
        loss = -total_loss / batch_size
        # loss = loss + self.reg_weight * regularization_term
        return loss

    def compute_accuracy(self, embeddings, targets):
        """
        Compute per-batch training accuracy.
        """
        # normalized embeddings [batch_size, embed_dim]
        normalized_embds = F.normalize(embeddings, p=2, dim=1)
        # normalized prototypes
        normalized_prototypes = F.normalize(self.prototypes, p=2, dim=2)
        flatten_prototypes = normalized_prototypes.view(-1, self.embed_dim) # [num_classes * prototype_per_class, embed_dim]

        # compute cosine similarities: [batch_size, num_classes * prototype_per_class]
        similarities = normalized_embds @ flatten_prototypes.t()
        # get prototypes with highest similarity (indices of prototypes)
        high_sim_prototype_idxs = similarities.argmax(dim=1)
        # map these indices to class indices (predicted classes)
        pred_classes = high_sim_prototype_idxs // self.prototype_per_class
        # compute accuracy
        correct = (pred_classes == targets).float().sum()
        accuracy = correct / embeddings.size(0)
        return accuracy
