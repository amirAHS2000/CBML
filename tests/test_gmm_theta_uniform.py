"""Regression tests for fixed uniform priors and near-uniform GMM mixture weights."""
import math
from types import SimpleNamespace as NS

import torch

from cbml_benchmark.losses.gmm_proxy import GaussianMixtureProxyLoss


def config(classes=3, components=3):
    return NS(
        MODEL=NS(HEAD=NS(DIM=4)),
        LOSSES=NS(GMM_PROXY_LOSS=NS(
            NUM_CLASSES=classes, K=components, BETA_INIT=10.0,
            THETA_INIT=1.5, BETA_LR=0.0001, PROXY_LR=0.01,
            WEIGHT_LR=0.1, WEIGHT_FLOOR=0.0001,
            WEIGHT_MAX_DEVIATION=0.02, INIT_METHOD='random',
        )),
    )


def test_theta_is_1_5_and_beta_is_exponential():
    loss = GaussianMixtureProxyLoss(config())
    assert abs(loss.log_beta.item() - 1.5) < 1e-6
    assert abs(loss.beta.item() - math.exp(1.5)) < 1e-6
    assert loss.log_beta.requires_grad  # only the INITIAL value was requested


def test_priors_are_equal_nonlearnable_and_saved_in_state():
    loss = GaussianMixtureProxyLoss(config(classes=4))
    assert torch.allclose(loss.class_log_priors.exp(), torch.full((4,), 0.25))
    assert 'class_log_priors' not in dict(loss.named_parameters())
    assert 'class_log_priors' in dict(loss.named_buffers())
    assert 'class_log_priors' in loss.state_dict()


def test_weight_step_keeps_weights_close_to_uniform_and_valid():
    obj = GaussianMixtureProxyLoss(config())
    before = obj.mixture_weights.detach().clone()
    # Only class 0 has positive assignments; the other classes must not move.
    obj.mixture_weights.grad = torch.tensor([
        [-1000., 500., 500.], [0., 0., 0.], [0., 0., 0.]
    ])
    obj.step_mixture_weights()
    weights = obj.mixture_weights.detach()
    assert torch.equal(weights[1:], before[1:])
    assert torch.allclose(weights.sum(1), torch.ones(3), atol=2e-6)
    assert bool((weights > 0).all())
    assert float((weights - 1./3).abs().max()) <= 0.020001
    assert weights[0].max() > 1. / 3  # not forcibly frozen uniform


def test_negative_classes_keep_zero_mean_and_weight_gradients():
    obj = GaussianMixtureProxyLoss(config())
    x = torch.randn(6, 4, requires_grad=True)
    labels = torch.zeros(6, dtype=torch.long)
    obj(x, labels).backward()
    assert torch.equal(obj.proxies.grad[1:], torch.zeros_like(obj.proxies.grad[1:]))
    assert torch.equal(obj.mixture_weights.grad[1:],
                       torch.zeros_like(obj.mixture_weights.grad[1:]))
    assert x.grad is not None and x.grad.norm() > 0
