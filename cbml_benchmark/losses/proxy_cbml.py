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
        self.count_aware = getattr(opt, "COUNT_AWARE", False)
        self.count_damping = getattr(opt, "COUNT_DAMPING", 10.0)
        self.count_min_assignments = getattr(opt, "COUNT_MIN_ASSIGNMENTS", 3)
        if self.count_aware:
            if not self.detach_negative:
                raise ValueError("Count-aware updates require DETACH_NEGATIVE=True.")
            if not (0 < self.count_damping < float('inf')) or self.count_min_assignments < 1:
                raise ValueError("Require finite COUNT_DAMPING > 0 and COUNT_MIN_ASSIGNMENTS >= 1.")
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

    @torch.no_grad()
    def scale_proxy_gradients(self, details, collect_stats=False):
        """E3: scale raw-parameter gradients from the SAME forward assignments.

        Call once after backward and before the proxy SGD step. This does not
        modify network gradients, mining, or the forward scalar objective.
        """
        if not self.count_aware:
            return {}
        grad = self.proxies.grad
        if grad is None:
            raise RuntimeError("Call scale_proxy_gradients after loss.backward().")
        ids = details['labels'] * self.k + details['pos_k']
        counts = torch.bincount(ids, minlength=self.num_classes * self.k)
        active = counts > 0
        factors = ids.numel() / (self.count_damping * counts.clamp_min(self.count_min_assignments).to(grad.dtype))
        factors = torch.where(active, factors, torch.zeros_like(factors))
        before = float(grad.norm()) if collect_stats else None
        grad.mul_(factors.reshape(self.num_classes, self.k, 1))
        if collect_stats:
            selected = factors[active]
            return {
                'proxy_grad_norm_before_scaling': before,
                'proxy_grad_norm_after_scaling': float(grad.norm()),
                'proxy_scale_min': float(selected.min()),
                'proxy_scale_mean': float(selected.mean()),
                'proxy_scale_max': float(selected.max()),
            }
        return {}

    def forward(self, feats, labels):
        loss_pos, loss_neg = self.loss_components(feats, labels)
        return loss_pos + loss_neg
