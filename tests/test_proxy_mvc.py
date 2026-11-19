"""Synthetic tests for the direct proxy-level MVC regularizer."""
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F

from cbml_benchmark.losses import build_loss


def make_cfg(mvc_enabled=True, gamma=0.0, weight=1.0, classes=3, k=2, dim=4):
    opt = NS(NUM_CLASSES=classes, K=k, POS_A=0.5, POS_B=0.1,
             NEG_A=0.2, NEG_B=0.5, DETACH_NEGATIVE=True, PROXY_LR=0.01,
             COUNT_AWARE=False, COUNT_DAMPING=10.0, COUNT_MIN_ASSIGNMENTS=3,
             MVC_ENABLED=mvc_enabled, MVC_WEIGHT=weight, MVC_GAMMA=gamma)
    return NS(MODEL=NS(HEAD=NS(DIM=dim)),
              LOSSES=NS(NAME='proxy_cbml_loss', NAME_AUX='', PROXY_CBML_LOSS=opt))


def fixed_criterion(gamma=0.0, weight=1.0):
    torch.manual_seed(4)
    criterion = build_loss(make_cfg(True, gamma, weight))
    with torch.no_grad():
        criterion.proxies.copy_(F.normalize(torch.tensor([
            [[1., 0., 0., 0.], [.8, .2, .1, 0.]],
            [[0., 1., 0., 0.], [.1, .8, .3, 0.]],
            [[0., 0., 1., 0.], [.2, .1, .8, 0.]],
        ]), dim=-1))
    return criterion


def manual_proxy_mvc(feats, proxies, labels, gamma):
    x = F.normalize(feats, dim=-1)
    p = F.normalize(proxies.detach(), dim=-1)
    scores = torch.einsum('bd,ckd->bck', x, p)
    rows = torch.arange(len(labels))
    positive = scores[rows, labels].max(dim=1).values
    class_ids = torch.arange(p.shape[0])[None, :]
    mask = (class_ids != labels[:, None]).unsqueeze(-1).expand_as(scores)
    negative = scores.masked_select(mask).reshape(len(labels), -1)
    target = gamma * positive + (1-gamma) * negative.mean(dim=1)
    return (negative - target[:, None]).square().mean()


def test_gamma_zero_is_mean_variance_over_all_negative_proxies():
    c = fixed_criterion(gamma=0.0)
    x = torch.tensor([[.7, .4, .2, .1], [.1, .6, .7, .2]], requires_grad=True)
    y = torch.tensor([0, 2])
    _, _, mvc, _ = c.loss_terms(x, y)
    expected = manual_proxy_mvc(x, c.proxies, y, 0.0)
    assert torch.allclose(mvc, expected, atol=1e-7)


def test_gamma_target_matches_formula_for_nonzero_gamma():
    c = fixed_criterion(gamma=0.2)
    x = torch.tensor([[.7, .4, .2, .1], [.1, .6, .7, .2]], requires_grad=True)
    y = torch.tensor([0, 2])
    _, _, mvc, _ = c.loss_terms(x, y)
    expected = manual_proxy_mvc(x, c.proxies, y, 0.2)
    assert torch.allclose(mvc, expected, atol=1e-7)


def test_mvc_updates_embeddings_but_never_any_proxy():
    c = fixed_criterion(gamma=0.0)
    x = torch.tensor([[.7, .4, .2, .1], [.1, .6, .7, .2]], requires_grad=True)
    y = torch.tensor([0, 2])
    _, _, mvc, _ = c.loss_terms(x, y)
    mvc.backward()
    assert x.grad is not None and x.grad.norm() > 0
    assert c.proxies.grad is None


def test_combined_loss_preserves_base_when_mvc_disabled():
    c = fixed_criterion(gamma=0.0)
    disabled = build_loss(make_cfg(False, 0.0, 1.0))
    with torch.no_grad():
        disabled.proxies.copy_(c.proxies)
    x = torch.tensor([[.7, .4, .2, .1], [.1, .6, .7, .2]])
    y = torch.tensor([0, 2])
    base = sum(c.loss_components(x, y))
    assert torch.allclose(disabled(x, y), base, atol=1e-7)


def test_forward_adds_weighted_mvc_without_changing_proxy_gradient_rule():
    c = fixed_criterion(gamma=0.0, weight=0.7)
    x = torch.tensor([[.7, .4, .2, .1], [.1, .6, .7, .2]], requires_grad=True)
    y = torch.tensor([0, 2])
    pos, neg = c.loss_components(x, y)
    mvc = c.mvc_component(x, y)
    total = c(x, y)
    assert torch.allclose(total, pos + neg + 0.7 * mvc, atol=1e-7)
    proxy_grad_total = torch.autograd.grad(total, c.proxies, retain_graph=True)[0]
    proxy_grad_pos = torch.autograd.grad(pos, c.proxies)[0]
    assert torch.allclose(proxy_grad_total, proxy_grad_pos, atol=1e-7)
