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

        # initializing parameters

        # prototypes: [num_classes, prototype_per_class, embed_dim]
        self.prototypes = nn.Parameter(
            torch.randn(self.num_classes, self.prototype_per_class, self.embed_dim).to(self.device)
        )
        # self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=2)

        # weights: [num_classes, prototype_per_class]
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class).to(self.device) / self.prototype_per_class
        )
        
        # class priors: uniform for simplicity [num_classes]
        self.class_priors = nn.Parameter(
            torch.ones(self.num_classes).to(self.device) / self.num_classes, requires_grad=False
        )

    def compute_similarity(self, embeddings, prototypes):
        """
        compute cosine similarity between each embeddings and all prototypes.
        goal: used to find that one prototype which has most contribution
        (or similarity) with the input data.
        """
        # embeddings: [batch_size, embed_dim]
        # prototypes: [num_classes * prototype_per_class, embed_dim]
        similarities = embeddings @ prototypes.t() # [batch_size, num_classes * prototype_per_class]
        return similarities / self.sigma_sq
    
    def forward(self, embeddings, targets):
        # normalized embeddings: [batch_size, embed_dim]
        normalized_embds = F.normalize(embeddings, p=2, dim=1)
        # normalized prototypes: [num_classes, prototype_per_class, embed_dim]
        normalized_prototypes = F.normalize(self.prototypes, p=2, dim=2)
        # normalized weights: [num_classes, prototype_per_class]
        weights = F.softmax(self.weights, dim=1)

        batch_size = embeddings.size(0)

        # compute cosine similarity between each embeddings and all prototypes
        similarities = self.compute_similarity(
            normalized_embds, normalized_prototypes.view(-1, self.embed_dim)
        ) # [batch_size, num_classes * prototype_per_class]
        similarities = similarities.view(
            batch_size, self.num_classes, self.prototype_per_class
        ) # [batch_size, num_classes, prototype_per_class]

        # true class prototype
        true_labels = targets # [batch_size]
        # list of all prototype (similarity value) corresponding to each sample in batch
        pos_prototypes_indices = similarities[torch.arange(batch_size), true_labels] # [batch_size, prototype_per_class]
        # index of most contributed prototype for each sample in batch
        pos_max_prototypes_idx = torch.argmax(pos_prototypes_indices, dim=1) # [batch_size]
        # most contributed prototype (from the class of sample)
        pos_prototypes = normalized_prototypes[true_labels, pos_max_prototypes_idx] # [batch_size, embed_dim]
        # weights corresponding to each positive prototypes
        pos_weights = weights[true_labels, pos_max_prototypes_idx] # [batch_size]
        # positive class priors
        pos_priors = self.class_priors[true_labels] # [batch_size]

        # negative prototypes (excluding positive class)
        all_similarities = similarities.view(batch_size, -1) # [batch_size, num_classes * prototype_per_class]
        mask = torch.ones_like(all_similarities, dtype=torch.bool)
        # compute indices for positive class prototypes for each sample
        pos_class_prototypes_idx = (true_labels.view(-1, 1) * self.prototype_per_class +
                                    torch.arange(self.prototype_per_class, device=self.device).view(1, -1)) # [batch_size, prototype_per_class]
        flatten_idxs = pos_class_prototypes_idx.view(-1) # [batch_size * prototype_per_class]
        batch_idxs = torch.arange(batch_size, device=self.device).repeat_interleave(self.prototype_per_class)
        mask[batch_idxs, flatten_idxs] = False
        masked_similarities = all_similarities.masked_fill(~mask, float('-inf'))
        neg_class_idxs = torch.argmax(masked_similarities, dim=1) # [batch_size]
        neg_class = neg_class_idxs // self.prototype_per_class # [batch_size]
        neg_class_prototype_idx = neg_class % self.prototype_per_class # [batch_size]
        neg_prototypes = normalized_prototypes[neg_class, neg_class_prototype_idx] # [batch_size, embed_dim]
        neg_weights = weights[neg_class, neg_class_prototype_idx] # [batch_size]
        neg_priors = self.class_priors[neg_class] # [batch_size]

        # similarity term
        pos_sim = (normalized_embds * pos_prototypes).sum(dim=1) # [batch_size]
        neg_sim = (normalized_embds * neg_prototypes).sum(dim=1) # [batch_size]
        sim_term = -1 * (1 / self.sigma_sq) * (pos_sim - neg_sim) # [batch_size]

        bias_term = -torch.log(
            (pos_priors * pos_weights) / (neg_priors * neg_weights + 1e-8)
        ) # [batch_size]

        loss = (sim_term + bias_term).mean()

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
