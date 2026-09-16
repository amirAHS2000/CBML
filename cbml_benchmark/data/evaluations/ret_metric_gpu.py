import numpy as np
import torch


class RetMetricGPU(object):
    """
    GPU-vectorized drop-in replacement for RetMetric.
    Computes recall@k for multiple k values in a single pass, entirely on GPU,
    with no Python-level loop over queries.
    """

    def __init__(self, feats, labels, device='cuda'):
        if isinstance(feats, list) and len(feats) == 2:
            self.is_equal_query = False
            gallery_feats, query_feats = feats
            gallery_labels, query_labels = labels
        else:
            self.is_equal_query = True
            gallery_feats = query_feats = feats
            gallery_labels = query_labels = labels

        self.gallery_feats = self._to_tensor(gallery_feats, device, torch.float32)
        self.query_feats = self._to_tensor(query_feats, device, torch.float32)
        self.gallery_labels = self._to_tensor(gallery_labels, device, torch.long)
        self.query_labels = self._to_tensor(query_labels, device, torch.long)

        # [num_queries, num_gallery]
        self.sim_mat = self.query_feats @ self.gallery_feats.t()

    @staticmethod
    def _to_tensor(x, device, dtype):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not torch.is_tensor(x):
            x = torch.tensor(x)
        return x.to(device=device, dtype=dtype)

    def recall_at_ks(self, ks=(1, 2, 4, 8)):
        """
        Computes recall@k for all k in `ks` in one pass.
        Returns: dict {k: recall_value}
        """
        pos_mask = self.query_labels.unsqueeze(1) == self.gallery_labels.unsqueeze(0)
        neg_mask = ~pos_mask

        pos_sim = self.sim_mat.masked_fill(~pos_mask, float('-inf'))

        if self.is_equal_query:
            # second-highest positive similarity (matches np.sort(pos_sim)[-2])
            thresh = torch.topk(pos_sim, k=2, dim=1).values[:, 1]
        else:
            thresh = pos_sim.max(dim=1).values

        above_thresh = (self.sim_mat > thresh.unsqueeze(1)) & neg_mask
        neg_count = above_thresh.sum(dim=1)

        m = self.sim_mat.size(0)
        results = {}
        for k in ks:
            results[k] = (neg_count < k).float().sum().item() / m
        return results

    def recall_k(self, k=1):
        """Backward-compatible single-k interface."""
        return self.recall_at_ks(ks=(k,))[k]