"""Regression checks for opt-in upper-tail MVC on E3 Proxy-CBML."""
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from cbml_benchmark.losses import build_loss
from cbml_benchmark.utils.mvc_extended_diagnostics import component_gradients


def cfg(mode='upper_tail', enabled=True, weight=10., delta=0., gamma=0.):
    opt = NS(NUM_CLASSES=3, K=2, POS_A=.5, POS_B=.1, NEG_A=.2, NEG_B=.5,
             DETACH_NEGATIVE=True, PROXY_LR=.01,
             COUNT_AWARE=True, COUNT_DAMPING=10., COUNT_MIN_ASSIGNMENTS=3,
             MVC_ENABLED=enabled, MVC_WEIGHT=weight, MVC_GAMMA=gamma,
             MVC_MODE=mode, MVC_TAIL_DELTA=delta)
    return NS(MODEL=NS(HEAD=NS(DIM=4)),
              LOSSES=NS(NAME='proxy_cbml_loss', NAME_AUX='', PROXY_CBML_LOSS=opt))


def fixed_loss(**kwargs):
    loss = build_loss(cfg(**kwargs))
    with torch.no_grad():
        loss.proxies.copy_(F.normalize(torch.tensor([
            [[1., 0., 0., 0.], [.8, .2, .1, 0.]],
            [[0., 1., 0., 0.], [.1, .8, .3, 0.]],
            [[0., 0., 1., 0.], [.2, .1, .8, 0.]],
        ]), dim=-1))
    return loss


def inputs():
    return torch.tensor([[.7, .4, .2, .1], [.1, .6, .7, .2]], requires_grad=True), torch.tensor([0, 2])


def negative_scores(x, proxies, labels):
    norm_x = F.normalize(x, dim=-1)
    norm_p = F.normalize(proxies.detach(), dim=-1)
    scores = torch.einsum('bd,ckd->bck', norm_x, norm_p)
    c, k = proxies.shape[:2]
    mask = (torch.arange(c, device=x.device)[None, :] != labels[:, None]).unsqueeze(-1)
    return scores.masked_select(mask.expand_as(scores)).reshape(len(x), (c-1)*k)


@pytest.mark.parametrize('delta', [0., .05, .4])
def test_tail_matches_manual_all_foreign_proxies(delta):
    loss = fixed_loss(delta=delta)
    x,y = inputs()
    scores = negative_scores(x, loss.proxies, y)
    threshold = scores.mean(dim=1).detach()[:, None] + delta
    manual = F.relu(scores-threshold).square().mean()
    actual, stats = loss.mvc_component(x,y,return_details=True)
    assert torch.allclose(actual,manual,atol=1e-7)
    assert stats['mvc_tail_active_fraction'] == pytest.approx(float((scores > threshold).float().mean()))
    assert stats['mvc_variance_mean'] == pytest.approx(float(scores.detach().var(dim=1,unbiased=False).mean()))
    assert stats['mvc_tail_threshold_mean'] == pytest.approx(float(threshold.mean()))


def test_detached_threshold_gives_exact_upper_tail_gradient():
    loss = fixed_loss(delta=.03)
    x,y = inputs()
    actual = loss.mvc_component(x,y)
    gx = torch.autograd.grad(actual,x)[0]
    scores = negative_scores(x,loss.proxies,y)
    residual = scores-scores.mean(dim=1).detach()[:,None]-.03
    coefficients = 2*F.relu(residual.detach())/scores.numel()
    gx_manual = torch.autograd.grad(scores,x,grad_outputs=coefficients)[0]
    assert torch.allclose(gx,gx_manual,atol=1e-7)
    assert float(gx.norm())>0
    assert torch.autograd.grad(actual,loss.proxies,allow_unused=True)[0] is None


def test_no_penalty_on_below_threshold_scores():
    scores = torch.tensor([[-.4, .2, .6]],requires_grad=True)
    threshold=scores.mean(dim=1).detach()[:,None]
    penalty=F.relu(scores-threshold).square().mean()
    gradient=torch.autograd.grad(penalty,scores)[0]
    assert gradient[0,0] == 0 and gradient[0,1] > 0 and gradient[0,2] > 0
    assert gradient[0,1] == pytest.approx(2*(.2-((-.4+.2+.6)/3))/3)


def test_weighted_loss_only_updates_selected_positive_proxy():
    loss=fixed_loss(weight=10.)
    x,y=inputs()
    pos,neg,mvc,details=loss.loss_terms(x,y,return_details=True)
    total=loss(x,y)
    assert torch.allclose(total,pos+neg+10*mvc)
    total_proxy_grad=torch.autograd.grad(total,loss.proxies,retain_graph=True)[0]
    pos_proxy_grad=torch.autograd.grad(pos,loss.proxies,retain_graph=True)[0]
    assert torch.allclose(total_proxy_grad,pos_proxy_grad,atol=1e-7)
    assert details['mvc_tail_active_fraction'] > 0
    expected_active=torch.zeros(3,2,dtype=torch.bool)
    expected_active[details['labels'],details['pos_k']]=True
    assert torch.count_nonzero(total_proxy_grad[~expected_active]) == 0


def test_baseline_and_symmetric_backward_compatibility():
    x,y=inputs()
    baseline=fixed_loss(enabled=False)
    pos,neg=baseline.loss_components(x,y)
    assert torch.allclose(baseline(x,y),pos+neg)
    assert baseline.mvc_component(x,y) == 0
    diagnostic=baseline.mvc_component(x,y,diagnostic_only=True)
    assert diagnostic>0  # upper-tail diagnostic even while disabled
    symmetric=fixed_loss(mode='symmetric', gamma=.2)
    neg_scores=negative_scores(x,symmetric.proxies,y)
    positive_scores=(F.normalize(x,dim=-1).unsqueeze(1)*F.normalize(symmetric.proxies.detach()[y],dim=-1)).sum(-1)
    target=.2*positive_scores.max(dim=1).values + .8*neg_scores.mean(dim=1)
    expected=(neg_scores-target[:,None]).square().mean()
    assert torch.allclose(symmetric.mvc_component(x,y),expected,atol=1e-7)


def test_network_gradient_instrumentation_is_readonly_and_correct():
    torch.manual_seed(6)
    model=torch.nn.Linear(4,4,bias=False)
    crit=fixed_loss(weight=10.)
    x,y=inputs()
    feats=model(x.detach())
    pos,neg,mvc,_=crit.loss_terms(feats,y,return_details=True)
    measured=component_gradients(model,pos,neg,mvc,10.)
    assert all(p.grad is None for p in model.parameters())
    assert measured['component_weighted_mvc_grad_norm'] == pytest.approx(10*measured['component_mvc_grad_norm'])
    (pos+neg+10*mvc).backward()
    actual=(sum(float(p.grad.square().sum()) for p in model.parameters()))**.5
    assert measured['component_total_grad_norm'] == pytest.approx(actual,rel=1e-5)
    assert crit.proxies.grad is not None
    assert measured['component_mvc_to_main_ratio'] is not None


@pytest.mark.parametrize('kwargs', [dict(mode='invalid'), dict(delta=-.1),
                                   dict(delta=float('nan')), dict(gamma=.2)])
def test_invalid_tail_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fixed_loss(**kwargs)


def test_full_dataset_upper_tail_diagnostics_correct():
    import numpy as np
    from cbml_benchmark.utils.mvc_extended_diagnostics import proxy_score_geometry
    x=np.array([[1.,0.],[0.,1.]])
    bank=np.array([[[1.,0.],[0.,1.]],[[-1.,0.],[0.,-1.]]])
    train=proxy_score_geometry(x,[0,1],bank,'train',tail_delta=.1)
    assert train['proxies_per_anchor']==2
    assert train['tail_delta']==pytest.approx(.1)
    assert train['tail_active_fraction']==pytest.approx(.5)
    assert train['tail_penalty_mean']==pytest.approx(.08)
    val=proxy_score_geometry(x,[0,999],bank,'validation',tail_delta=.1)
    assert val['proxies_per_anchor']==4
    assert val['tail_active_fraction']==pytest.approx(.25)
    assert val['tail_penalty_mean']==pytest.approx((.9**2)/4)


def test_one_training_step_keeps_unassigned_proxies_unchanged():
    torch.manual_seed(27)
    crit=fixed_loss(weight=10.)
    network=torch.nn.Linear(4,4,bias=False)
    netopt=torch.optim.SGD(network.parameters(), lr=.01)
    proxyopt=torch.optim.SGD([crit.proxies], lr=.01)
    x,y=inputs()
    feats=network(x.detach())
    lp,ln,lm,info=crit.loss_terms(feats,y,return_details=True)
    total=lp+ln+crit.mvc_weight*lm
    before=crit.proxies.detach().clone()
    net_before=network.weight.detach().clone()
    netopt.zero_grad(); proxyopt.zero_grad()
    total.backward()
    crit.scale_proxy_gradients(info)
    netopt.step(); proxyopt.step()
    ids=torch.zeros((crit.num_classes,crit.k),dtype=torch.bool)
    ids[info['labels'],info['pos_k']]=True
    assert torch.equal(before[~ids],crit.proxies.detach()[~ids])
    assert torch.any(before[ids]!=crit.proxies.detach()[ids])
    assert torch.any(net_before!=network.weight.detach())
