"""Synthetic regression tests; no images or pretrained weights required."""
import logging
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from cbml_benchmark.losses import build_loss
from cbml_benchmark.solver import build_proxy_optimizer, build_lr_scheduler
from cbml_benchmark.utils.checkpoint import Checkpointer
from cbml_benchmark.utils.proxy_init import configure_proxy_classes, spherical_kmeans
from cbml_benchmark.engine.trainer import do_train


def options(detach=True):
    return NS(
        MODEL=NS(HEAD=NS(DIM=2)),
        LOSSES=NS(NAME='proxy_cbml_loss', NAME_AUX='', PROXY_CBML_LOSS=NS(
            NUM_CLASSES=2, K=2, POS_A=.5, POS_B=.1, NEG_A=.2, NEG_B=.1,
            DETACH_NEGATIVE=detach, PROXY_LR=.01, INIT_METHOD='random')),
        SOLVER=NS(STEPS=[10], GAMMA=.1, WARMUP_FACTOR=.01,
                  WARMUP_ITERS=0, WARMUP_METHOD='linear'),
        VALIDATION=NS(VERBOSE=1),
    )


def criterion_and_batch(detach=True):
    cfg = options(detach)
    criterion = build_loss(cfg)
    with torch.no_grad():
        criterion.proxies.copy_(torch.tensor([
            [[1., 0.], [-1., 0.]], [[.8, .6], [0., -1.]],
        ]))
    return cfg, criterion, torch.tensor([[.9, .2]], requires_grad=True), torch.tensor([0])


def test_selection_and_value():
    cfg, c, x, y = criterion_and_batch()
    p = F.normalize(c.proxies, dim=-1)
    pos, neg_c, neg_k = c.select_proxies(F.normalize(x, dim=-1), p, y)
    assert (pos.item(), neg_c.item(), neg_k.item()) == (0, 1, 0)
    assert c.proxies.shape == (2, 2, 2)
    z = F.normalize(x, dim=-1)[0]
    expected = F.softplus((.5 - z.dot(p[0, 0])) / .1)
    expected += F.softplus((z.dot(p[1, 0]) - .2) / .1)
    assert torch.allclose(c(x, y), expected)


def test_proxy_immobility_after_step():
    cfg, c, x, y = criterion_and_batch()
    opt = build_proxy_optimizer(cfg, c)
    before = c.proxies.detach().clone()
    opt.zero_grad()
    c(x, y).backward()
    assert c.proxies.grad[0, 0].norm() > 0
    assert torch.count_nonzero(c.proxies.grad[0, 1:]) == 0
    assert torch.count_nonzero(c.proxies.grad[1]) == 0
    opt.step()
    assert not torch.equal(before[0, 0], c.proxies[0, 0])
    assert torch.equal(before[0, 1:], c.proxies[0, 1:])
    assert torch.equal(before[1], c.proxies[1])


def test_negative_only_gradient_reaches_embedding():
    _, c, x, y = criterion_and_batch()
    _, neg = c.loss_components(x, y)
    neg.backward()
    assert x.grad.norm() > 0
    assert c.proxies.grad is None


def test_detach_ablation():
    cfg, c, x, y = criterion_and_batch(False)
    before = c.proxies.detach().clone()
    opt = build_proxy_optimizer(cfg, c)
    c(x, y).backward()
    assert c.proxies.grad[1, 0].norm() > 0
    opt.step()
    assert not torch.equal(before[1, 0], c.proxies[1, 0])


def test_mixed_class_proxy_gradient_equals_own_class_positive():
    _, c, _, _ = criterion_and_batch()
    x = torch.tensor([[.9, .2], [.65, .76]], requires_grad=True)
    y = torch.tensor([0, 1])
    total = torch.autograd.grad(c(x, y), c.proxies)[0]
    pos, _ = c.loss_components(x, y)
    positive = torch.autograd.grad(pos, c.proxies)[0]
    assert torch.equal(total, positive)
    assert total[0, 0].norm() > 0 and total[1, 0].norm() > 0


def test_class_count_and_spherical_kmeans():
    cfg = options()
    cfg.LOSSES.PROXY_CBML_LOSS.NUM_CLASSES = 0
    assert configure_proxy_classes(cfg, NS(label_list=['0', '1', '0'])) == [0, 1]
    assert cfg.LOSSES.PROXY_CBML_LOSS.NUM_CLASSES == 2
    with pytest.raises(ValueError):
        configure_proxy_classes(cfg, NS(label_list=['0', '5']))
    x = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])
    p = spherical_kmeans(x, 2)
    assert p.shape == (2, 2)
    assert torch.allclose(p.norm(dim=1), torch.ones(2))


def test_checkpoint_round_trip(tmp_path):
    cfg, c, x, y = criterion_and_batch()
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    proxy_opt = build_proxy_optimizer(cfg, c)
    sched = build_lr_scheduler(cfg, opt)
    c(model(x), y).backward()
    opt.step()
    proxy_opt.step()
    sched.step()
    saved = c.proxies.detach().clone()
    ck = Checkpointer(model, opt, sched, str(tmp_path), criterion=c,
                      proxy_optimizer=proxy_opt)
    ck.save('model_000001', iteration=1, class_ids=[0, 1])
    with torch.no_grad():
        c.proxies.zero_()
    state = ck.load(str(tmp_path / 'model_000001.pth'), expected_classes=[0, 1],
                    require_proxy_state=True)
    assert torch.equal(saved, c.proxies)
    assert state['iteration'] == 1
    with pytest.raises(ValueError):
        ck.load(str(tmp_path / 'model_000001.pth'), expected_classes=[0, 2])


def test_trainer_smoke(tmp_path):
    cfg, c, _, _ = criterion_and_batch()
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    proxy_opt = build_proxy_optimizer(cfg, c)
    sched = build_lr_scheduler(cfg, opt)
    train = [(torch.tensor([[.9, .2]]), torch.tensor([0]))] * 2

    class ValidationLoader(list):
        dataset = NS(label_list=['0', '0', '1', '1'])

    val = ValidationLoader([(torch.tensor([[1., 0.], [.99, .01],
                                          [0., 1.], [.01, .99]]),
                             torch.tensor([0, 0, 1, 1]))])
    ck = Checkpointer(model, opt, sched, str(tmp_path), criterion=c,
                      proxy_optimizer=proxy_opt)
    before = c.proxies.detach().clone()
    args = {'iteration': 0, 'class_ids': [0, 1]}
    do_train(cfg, model, train, val, opt, sched, c, None, ck,
             torch.device('cpu'), 1, args, logging.getLogger('test'),
             proxy_optimizer=proxy_opt)
    assert args['iteration'] == 2
    assert (tmp_path / 'model_000002.pth').exists()
    assert torch.equal(before[1], c.proxies[1])
    assert not torch.equal(before[0, 0], c.proxies[0, 0])


def test_actual_config_and_baseline_registry():
    pytest.importorskip('yacs')
    from cbml_benchmark.config import cfg
    original = cfg.clone()
    assert build_loss(original).__class__.__name__ == 'CBMLLoss'
    new = cfg.clone()
    new.merge_from_file('configs/example_resnet50_proxy_cbml.yaml')
    new.LOSSES.PROXY_CBML_LOSS.NUM_CLASSES = 100
    assert build_loss(new).proxies.shape == (100, 3, 512)
