import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationAccuracyMetric(object):
    def __init__(self, feats, labels, prototypes):
        # features: [batch_size, embed_dim]
        self.normalized_feats = F.normalize(feats, p=2, dim=1)
        # prototypes: [num_classes, prototype_per_class, embed_dim]
        self.normalized_protos = F.normalize(prototypes, p=2, dim=2)
        
        self.labels = labels

        self.embed_dim = feats.size(1)
        self.prototype_per_class = prototypes.size(1)
        self.batch_size = feats.size(0)

        # similarity matrix: [batch_size, num_classes * prototype_per_class]
        self.similarities = self.normalized_feats @ self.normalized_protos.view(-1, self.embed_dim)

    def classification_acc(self):
        # get the prototypes with highest similarity (indices of prototypes)
        high_sim_prototype_idxs = self.similarities.argmax(dim=1)
        predicted_class = high_sim_prototype_idxs // self.prototype_per_class
        
        # compute accuracy
        correct = (predicted_class == self.labels).float().sum()
        accuracy = correct / self.batch_size
        return accuracy
