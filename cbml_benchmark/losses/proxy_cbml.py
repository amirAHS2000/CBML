import torch
from torch import nn
import torch.nn.functional as F

from .registry import LOSS


@LOSS.register("proxy_cbml_loss")
class ProxyCBMLLoss(nn.Module):
    """Two selected instance-to-proxy relationships per anchor."""

    def __init__(self, cfg):
        super().__init__()
        opt = cfg.LOSSES.PROXY_CBML_LOSS
        self.num_classes = opt.NUM_CLASSES
        self.k = opt.K
        self.dim = cfg.MODEL.HEAD.DIM
        self.pos_a, self.pos_b = opt.POS_A, opt.POS_B
        self.neg_a, self.neg_b = opt.NEG_A, opt.NEG_B
        self.detach_negative = opt.DETACH_NEGATIVE
        if self.num_classes < 2 or self.k < 1 or self.dim < 1:
            raise ValueError("Require C >= 2, K >= 1 and D >= 1.")
        if self.pos_b <= 0 or self.neg_b <= 0:
            raise ValueError("POS_B and NEG_B must be positive.")
        self.proxies = nn.Parameter(F.normalize(
            torch.randn(self.num_classes, self.k, self.dim), dim=-1
        ))

    @torch.no_grad()
    def select_proxies(self, feats, proxies, labels):
        # Inputs are normalized. Mining is global, not batch-restricted.
        b = feats.shape[0]
        c, k = self.num_classes, self.k
        scores = (feats @ proxies.reshape(c * k, self.dim).t()).reshape(b, c, k)
        rows = torch.arange(b, device=feats.device)
        pos_k = scores[rows, labels].argmax(dim=1)
        same_class = torch.arange(c, device=feats.device)[None, :] == labels[:, None]
        neg_scores = scores.masked_fill(same_class.unsqueeze(-1), float('-inf'))
        neg_flat = neg_scores.reshape(b, c * k).argmax(dim=1)
        return pos_k, neg_flat // k, neg_flat % k

    def loss_components(self, feats, labels, return_details=False):
        """Return the two mean NLL terms separately for tests/diagnostics."""
        labels = labels.long().reshape(-1)
        if feats.ndim != 2 or feats.shape[1] != self.dim or feats.shape[0] == 0:
            raise ValueError("Expected nonempty embeddings of shape [B, D].")
        if labels.numel() != feats.shape[0]:
            raise ValueError("Expected one label per embedding.")
        if (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError("Training labels must be in [0, C-1].")
        x = F.normalize(feats, dim=-1)
        p = F.normalize(self.proxies, dim=-1)
        pos_k, neg_c, neg_k = self.select_proxies(x, p, labels)
        p_pos = p[labels, pos_k]
        p_neg = p[neg_c, neg_k]
        if self.detach_negative:
            p_neg = p_neg.detach()
        s_pos = (x * p_pos).sum(dim=-1)
        s_neg = (x * p_neg).sum(dim=-1)
        loss_pos = F.softplus((self.pos_a - s_pos) / self.pos_b).mean()
        loss_neg = F.softplus((s_neg - self.neg_a) / self.neg_b).mean()
        if return_details:
            return loss_pos, loss_neg, {
                "s_pos": s_pos.detach(), "s_neg": s_neg.detach(),
                "labels": labels.detach(), "pos_k": pos_k,
                "neg_c": neg_c, "neg_k": neg_k,
            }
        return loss_pos, loss_neg

    def forward(self, feats, labels):
        loss_pos, loss_neg = self.loss_components(feats, labels)
        return loss_pos + loss_neg
