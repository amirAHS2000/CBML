import torch
from torch import nn
import torch.nn.functional as F

from .registry import LOSS


@LOSS.register("proxy_cbml_loss")
class ProxyCBMLLoss(nn.Module):
    """CBML-inspired instance-to-proxy loss with optional proxy-level MVC.

    The base objective still selects one hardest positive and one hardest
    negative proxy per anchor. The MVC branch considers every proxy belonging
    to every non-target class. All proxies are detached in the MVC branch, so
    MVC gradients update the embedding network but never the proxy bank.
    """

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

        self.mvc_enabled = bool(getattr(opt, "MVC_ENABLED", False))
        self.mvc_weight = float(getattr(opt, "MVC_WEIGHT", 1.0))
        self.mvc_gamma = float(getattr(opt, "MVC_GAMMA", 0.0))
        if self.mvc_weight < 0:
            raise ValueError("MVC_WEIGHT must be non-negative.")
        if not 0.0 <= self.mvc_gamma <= 1.0:
            raise ValueError("MVC_GAMMA must be in [0, 1].")
        if self.mvc_enabled and not self.detach_negative:
            raise ValueError("Proxy-MVC requires DETACH_NEGATIVE=True to preserve proxy-update constraints.")

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
        """Return the original positive and negative NLL terms.

        This API is retained for backwards compatibility with E0-E3 tests and
        analysis. The MVC term is returned separately by ``loss_terms``.
        """
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

    def mvc_component(self, feats, labels, return_details=False, diagnostic_only=False):
        """Proxy-level MVC over all (C-1)*K foreign-class proxies per anchor.

        For anchor i, q_i,c,k = cosine(z_i, p_c,k). The target is
            xi_i = gamma * max_k q_i,y_i,k
                   + (1-gamma) * mean_{c != y_i,k}(q_i,c,k).
        The regularizer is the mean squared deviation of every negative-proxy
        score from xi_i. Proxy parameters are detached; gradients flow only
        through the normalized embeddings.
        """
        if not self.mvc_enabled and not diagnostic_only:
            zero = feats.new_zeros(())
            return (zero, {}) if return_details else zero

        labels = labels.long().reshape(-1)
        x = F.normalize(feats, dim=-1)
        # Detach the complete proxy bank in MVC: this branch must not move
        # positive or foreign-class representatives.
        p = F.normalize(self.proxies.detach(), dim=-1)
        scores = torch.einsum("bd,ckd->bck", x, p)
        rows = torch.arange(feats.shape[0], device=feats.device)
        positive_scores = scores[rows, labels]
        positive_hard = positive_scores.max(dim=1).values

        class_ids = torch.arange(self.num_classes, device=feats.device)[None, :]
        negative_class_mask = class_ids != labels[:, None]
        negative_scores = scores.masked_select(
            negative_class_mask.unsqueeze(-1).expand_as(scores)
        ).reshape(feats.shape[0], (self.num_classes - 1) * self.k)

        negative_mean = negative_scores.mean(dim=1)
        target = self.mvc_gamma * positive_hard + (1.0 - self.mvc_gamma) * negative_mean
        residual = negative_scores - target.unsqueeze(1)
        per_anchor = residual.square().mean(dim=1)
        loss_mvc = per_anchor.mean()

        if not return_details:
            return loss_mvc
        stats = {
            "mvc_neg_score_mean": negative_scores.detach().mean(),
            "mvc_neg_score_std": negative_scores.detach().std(unbiased=False),
            "mvc_within_anchor_neg_std": negative_scores.detach().std(dim=1, unbiased=False).mean(),
            "mvc_positive_score_mean": positive_hard.detach().mean(),
            "mvc_target_mean": target.detach().mean(),
            "mvc_residual_std": residual.detach().std(unbiased=False),
            "mvc_variance_mean": negative_scores.detach().var(dim=1, unbiased=False).mean(),
        }
        return loss_mvc, stats

    def loss_terms(self, feats, labels, return_details=False):
        """Return positive, negative, MVC and optional detached diagnostics."""
        if return_details:
            loss_pos, loss_neg, details = self.loss_components(feats, labels, return_details=True)
            loss_mvc, mvc_stats = self.mvc_component(feats, labels, return_details=True)
            details.update(mvc_stats)
            return loss_pos, loss_neg, loss_mvc, details
        loss_pos, loss_neg = self.loss_components(feats, labels)
        loss_mvc = self.mvc_component(feats, labels, return_details=False)
        return loss_pos, loss_neg, loss_mvc, None

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
        loss_pos, loss_neg, loss_mvc, _ = self.loss_terms(feats, labels, return_details=False)
        return loss_pos + loss_neg + self.mvc_weight * loss_mvc
