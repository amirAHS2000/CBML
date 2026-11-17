"""Multi-prototype CBML from previous_works.pdf, Sections 1--7.

One selected positive and negative component are used in the two-class
posterior approximation. All negative component parameters are detached:
foreign-class examples update the embedding network, not their negative means
or mixture weights. Class priors are fixed; covariance is shared isotropic.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .registry import LOSS


def project_simplex_with_floor(values, floor):
    """Euclidean projection of each row onto w_j >= floor, sum_j w_j = 1.

    The PDF's mean-subtracted Lagrange step preserves only the equality.
    Projection additionally enforces strict positivity for log(w).
    """
    if values.ndim != 2:
        raise ValueError("Expected a [classes, components] tensor.")
    k = values.shape[1]
    if not (0 < floor < 1 / k):
        raise ValueError("Weight floor must be strictly between 0 and 1/K.")
    # Projection onto a nonnegative simplex of mass 1-K*floor.
    target = 1.0 - k * floor
    z = values - floor
    sorted_z = torch.sort(z, dim=1, descending=True).values
    cssv = sorted_z.cumsum(1) - target
    ind = torch.arange(1, k + 1, device=z.device, dtype=z.dtype).unsqueeze(0)
    valid = sorted_z - cssv / ind > 0
    rho = valid.sum(1).clamp_min(1) - 1
    theta = cssv.gather(1, rho[:, None]) / (rho + 1).to(z.dtype).unsqueeze(1)
    return torch.clamp_min(z - theta, 0) + floor


@LOSS.register('gmm_proxy_loss')
class GaussianMixtureProxyLoss(nn.Module):
    """Hard-component approximation of a Gaussian-mixture class posterior.

    PDF Eq. (1)--(26): argmax weighted log Gaussian per class, select the
    nearest competing *class* under its best component, then minimize
    softplus(log S_negative - log S_positive). All class priors uniform here.

    PDF Eq. (29)--(35): classwise mean-subtracted mixture-weight gradient.
    We follow it with a simplex projection to enforce strict positivity.
    """
    is_gmm_proxy_loss = True

    def __init__(self, cfg):
        super().__init__()
        opt = cfg.LOSSES.GMM_PROXY_LOSS
        self.num_classes = int(opt.NUM_CLASSES)
        self.k = int(opt.K)
        self.dim = int(cfg.MODEL.HEAD.DIM)
        self.weight_lr = float(opt.WEIGHT_LR)
        self.weight_floor = float(opt.WEIGHT_FLOOR)
        # 0.0 disables the additional near-uniform constraint (legacy behavior).
        self.weight_max_deviation = float(getattr(opt, 'WEIGHT_MAX_DEVIATION', 0.0))
        self.beta_lr = float(opt.BETA_LR)
        self.proxy_lr = float(opt.PROXY_LR)
        self.init_method = str(opt.INIT_METHOD)
        # THETA_INIT sets theta = log(beta) directly.  If omitted, preserve
        # backward compatibility with the former BETA_INIT option.
        theta_override = getattr(opt, 'THETA_INIT', None)
        if theta_override is None:
            beta_initial = float(opt.BETA_INIT)
            if not (math.isfinite(beta_initial) and beta_initial > 0):
                raise ValueError('BETA_INIT must be finite and positive.')
            theta_initial = math.log(beta_initial)
        else:
            theta_initial = float(theta_override)
            if not math.isfinite(theta_initial):
                raise ValueError('THETA_INIT must be finite.')
            beta_initial = math.exp(theta_initial)
        if self.num_classes < 2 or self.k < 1 or self.dim < 1:
            raise ValueError('Require C>=2, K>=1 and D>=1.')
        if not (math.isfinite(beta_initial) and beta_initial > 0):
            raise ValueError('Initial precision exp(theta) must be finite and positive.')
        if not (math.isfinite(self.weight_max_deviation) and self.weight_max_deviation >= 0):
            raise ValueError('WEIGHT_MAX_DEVIATION must be finite and nonnegative.')
        if not (math.isfinite(self.beta_lr) and self.beta_lr > 0):
            raise ValueError('BETA_LR must be finite and positive.')
        if not (math.isfinite(self.proxy_lr) and self.proxy_lr > 0):
            raise ValueError('PROXY_LR must be finite and positive.')
        if not (math.isfinite(self.weight_lr) and self.weight_lr > 0):
            raise ValueError('WEIGHT_LR must be finite and positive.')
        if not (0 < self.weight_floor < 1 / self.k):
            raise ValueError('WEIGHT_FLOOR must lie in (0, 1/K).')
        if self.init_method not in ('random', 'kmeans'):
            raise ValueError('INIT_METHOD must be random or kmeans.')
        # Unlike the cosine-only loss, Gaussian means are unconstrained.
        # L2-normalized initial means are provided by spherical KMeans.
        self.proxies = nn.Parameter(F.normalize(
            torch.randn(self.num_classes, self.k, self.dim), dim=-1
        ))
        self.mixture_weights = nn.Parameter(torch.full(
            (self.num_classes, self.k), 1.0 / self.k
        ))
        self.log_beta = nn.Parameter(
            torch.tensor(theta_initial),
            requires_grad=False
        )
        # Fixed, equal class priors: this is a BUFFER, not an optimizer parameter.
        # This preserves the existing checkpoint key and all negative-grad rules.
        self.register_buffer('class_log_priors', torch.full(
            (self.num_classes,), -math.log(self.num_classes)
        ))

    @property
    def beta(self):
        return self.log_beta.exp()

    def component_log_scores(self, x):
        """Log of p(c)*w[c,k]*N(x; mean[c,k], beta^-1 I), up to common constant."""
        x = F.normalize(x, dim=1)
        mu = self.proxies
        d2 = (x.square().sum(1, keepdim=True)
              + mu.square().sum(-1).reshape(1, -1)
              - 2.0 * x @ mu.reshape(-1, self.dim).T).clamp_min(0)
        scores = (self.class_log_priors[:, None] + self.mixture_weights.log()).reshape(1, -1)
        scores = scores - 0.5 * self.beta * d2
        return scores.reshape(x.shape[0], self.num_classes, self.k)

    @torch.no_grad()
    def select_components(self, x, labels):
        scores = self.component_log_scores(x)
        rows = torch.arange(x.shape[0], device=x.device)
        pos_k = scores[rows, labels].argmax(dim=1)
        negative_scores = scores.masked_fill(
            (torch.arange(self.num_classes, device=x.device)[None, :] == labels[:, None]).unsqueeze(-1),
            -torch.inf
        )
        neg_flat = negative_scores.flatten(1).argmax(dim=1)
        return pos_k, neg_flat // self.k, neg_flat % self.k

    def loss_and_details(self, feats, labels, return_details=False):
        labels = labels.long().reshape(-1)
        if feats.ndim != 2 or not len(feats) or feats.shape[1] != self.dim:
            raise ValueError('Expected nonempty [B,D] embeddings.')
        if len(labels) != len(feats) or (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError('Expected labels of shape [B], 0 <= labels < C.')
        x = F.normalize(feats, dim=1)
        pos_k, neg_c, neg_k = self.select_components(x, labels)
        # Important: negative means AND negative mixture weights are detached.
        pos_mu = self.proxies[labels, pos_k]
        neg_mu = self.proxies[neg_c, neg_k].detach()
        w_pos = self.mixture_weights[labels, pos_k]
        w_neg = self.mixture_weights[neg_c, neg_k].detach()
        log_positive = (self.class_log_priors[labels] + w_pos.log()
                        - 0.5 * self.beta * (x - pos_mu).square().sum(1))
        log_negative = (self.class_log_priors[neg_c] + w_neg.log()
                        - 0.5 * self.beta * (x - neg_mu).square().sum(1))
        log_odds = log_negative - log_positive
        loss = F.softplus(log_odds).mean()
        if not return_details:
            return loss
        return loss, {
            'labels': labels.detach(), 'pos_k': pos_k, 'neg_c': neg_c, 'neg_k': neg_k,
            'positive_log_score': log_positive.detach(),
            'negative_log_score': log_negative.detach(),
            'log_odds': log_odds.detach(),
            'posterior': torch.sigmoid(-log_odds.detach()),
            's_pos': F.cosine_similarity(x.detach(), pos_mu.detach()),
            's_neg': F.cosine_similarity(x.detach(), neg_mu.detach()),
            'positive_distance2': (x.detach() - pos_mu.detach()).square().sum(1),
            'negative_distance2': (x.detach() - neg_mu.detach()).square().sum(1),
            'selected_positive_weight': w_pos.detach(),
            'selected_negative_weight': w_neg.detach(),
        }

    @torch.no_grad()
    def step_mixture_weights(self, collect_stats=False):
        """PDF Eq. (32)--(35), followed by positive-simplex projection.

        Called once *after* backward. Class rows without positive examples
        have no gradients and remain bitwise unchanged.
        """
        grad = self.mixture_weights.grad
        if grad is None:
            raise RuntimeError('No mixture-weight gradients: call backward first.')
        active = grad.abs().sum(dim=1) > 0
        updates = {}
        if collect_stats:
            updates = {
                'mixture_weight_gradient_norm': float(grad.norm()),
                'mixture_weight_active_classes': int(active.sum()),
            }
        if active.any():
            centered = grad[active] - grad[active].mean(dim=1, keepdim=True)
            tentative = self.mixture_weights[active] - self.weight_lr * centered
            projected = project_simplex_with_floor(tentative, self.weight_floor)
            if self.weight_max_deviation > 0 and self.k > 1:
                # Contract toward the uniform distribution until every weight
                # lies within +/- WEIGHT_MAX_DEVIATION of 1/K.  This operation
                # preserves sum=1 and the strictly-positive floor; it is an
                # additional modeling constraint, not PDF Eq. (33) alone.
                uniform = 1.0 / self.k
                offset = projected - uniform
                largest = offset.abs().amax(dim=1, keepdim=True)
                fraction = (self.weight_max_deviation /
                            largest.clamp_min(1e-12)).clamp(max=1.0)
                projected = uniform + fraction * offset
            self.mixture_weights[active] = projected
        if collect_stats:
            weights = self.mixture_weights.detach()
            updates.update(
                mixture_weight_min=float(weights.min()),
                mixture_weight_max=float(weights.max()),
                mixture_weight_sum_error=float((weights.sum(1) - 1).abs().max()),
                mixture_weight_entropy=float((-(weights * weights.log()).sum(1)).mean()),
                mixture_weight_max_uniform_deviation=float(
                    (weights - 1.0 / self.k).abs().max()),
            )
        return updates

    def forward(self, feats, labels):
        return self.loss_and_details(feats, labels)
