import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


def compute_nmi(feats, labels, seed=0):
    """
    NMI between K-means clusters of the embeddings and ground-truth labels.
    feats:  (N, D) array or tensor
    labels: (N,) array of ints
    """
    if torch.is_tensor(feats):
        feats = feats.detach().cpu().numpy()
    labels = np.asarray(labels)

    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = kmeans.fit_predict(feats)
    return normalized_mutual_info_score(labels, pred)