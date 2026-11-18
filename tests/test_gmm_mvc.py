"""Synthetic tests for the proxy analogue of CBML MVC, Eqs. (28)--(30)."""
from types import SimpleNamespace as NS
import sys
import types
from pathlib import Path

# The released project snapshot intentionally omits several comparison losses;
# bypass its eager losses/__init__.py imports so this focused test can load GMM.
losses_pkg = types.ModuleType('cbml_benchmark.losses')
losses_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'cbml_benchmark' / 'losses')]
sys.modules.setdefault('cbml_benchmark.losses', losses_pkg)

import torch
from torch.nn import functional as F

from cbml_benchmark.losses.gmm_proxy import GaussianMixtureProxyLoss


def make_cfg(mvc_weight=1.0, gamma=0.2, detach=True):
    return NS(
        MODEL=NS(HEAD=NS(DIM=5)),
        LOSSES=NS(GMM_PROXY_LOSS=NS(
            NUM_CLASSES=4, K=3, BETA_INIT=4.481689,
            THETA_INIT=1.5, THETA_LEARNABLE=False,
            BETA_LR=1e-4, PROXY_LR=1e-2, WEIGHT_LR=1e-3,
            WEIGHT_FLOOR=1e-4, WEIGHT_MAX_DEVIATION=0.02,
            INIT_METHOD='kmeans', MVC_WEIGHT=mvc_weight,
            MVC_GAMMA=gamma, MVC_DETACH_PROXIES=detach,
        )),
    )


def test_mvc_scores_are_class_level_log_mixture_likelihoods():
    torch.manual_seed(7)
    loss = GaussianMixtureProxyLoss(make_cfg())
    x = F.normalize(torch.randn(6, 5), dim=1)
    scores = loss.class_gmm_similarity_scores(x)
    assert scores.shape == (6, 4)
    assert torch.isfinite(scores).all()


def test_mvc_uses_all_negative_classes_and_exact_cbml_target_form():
    torch.manual_seed(8)
    loss = GaussianMixtureProxyLoss(make_cfg(mvc_weight=0.7, gamma=0.2))
    x = torch.randn(5, 5)
    y = torch.tensor([0, 1, 2, 3, 0])
    value, details = loss.mvc_loss(x, y, return_details=True)
    scores = loss.class_gmm_similarity_scores(x)
    rows = torch.arange(len(y))
    positive = scores[rows, y]
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[rows, y] = False
    negative = scores[mask].reshape(len(y), loss.num_classes - 1)
    target = 0.2 * positive + 0.8 * negative.mean(1)
    expected = ((negative - target[:, None]) ** 2).mean()
    assert torch.allclose(value, expected, atol=1e-7)
    assert torch.allclose(details['mvc_target_mean'], target.mean())


def test_mvc_updates_embeddings_but_not_any_proxy_or_weight():
    torch.manual_seed(9)
    loss = GaussianMixtureProxyLoss(make_cfg(mvc_weight=1.0, detach=True))
    x = torch.randn(8, 5, requires_grad=True)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    mvc = loss.mvc_loss(x, y)
    mvc.backward()
    assert x.grad is not None and x.grad.norm() > 0
    assert loss.proxies.grad is None
    assert loss.mixture_weights.grad is None
    assert loss.log_beta.grad is None


def test_total_loss_is_gmm_plus_lambda_times_mvc():
    torch.manual_seed(10)
    cfg = make_cfg(mvc_weight=0.35)
    loss = GaussianMixtureProxyLoss(cfg)
    x = torch.randn(12, 5)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    total, details = loss.loss_and_details(x, y, return_details=True)
    expected = details['gmm_loss'] + 0.35 * details['mvc_loss']
    assert torch.allclose(total.detach(), expected, atol=1e-6)
    assert abs(float(loss.log_beta.detach()) - 1.5) < 1e-7
    assert not loss.log_beta.requires_grad


def test_zero_mvc_weight_preserves_previous_gmm_objective():
    torch.manual_seed(11)
    cfg0 = make_cfg(mvc_weight=0.0)
    cfg1 = make_cfg(mvc_weight=1.0)
    a = GaussianMixtureProxyLoss(cfg0)
    b = GaussianMixtureProxyLoss(cfg1)
    b.load_state_dict(a.state_dict())
    x = torch.randn(10, 5)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1])
    la = a(x, y)
    lb, db = b.loss_and_details(x, y, return_details=True)
    assert torch.allclose(la, db['gmm_loss'], atol=1e-6)
