"""E3 regression tests: exact scaling, gradient isolation and read-only logging."""
import csv
import json
import logging
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cbml_benchmark.losses import build_loss
from cbml_benchmark.solver import build_proxy_optimizer, build_lr_scheduler
from cbml_benchmark.engine.trainer import do_train
from cbml_benchmark.utils.checkpoint import Checkpointer
from cbml_benchmark.utils.training_diagnostics import TrainingDiagnostics, instance_geometry


def options(count_aware=True, dim=2, classes=2, k=2):
    return NS(MODEL=NS(HEAD=NS(DIM=dim)), LOSSES=NS(NAME='proxy_cbml_loss', NAME_AUX='',
        PROXY_CBML_LOSS=NS(NUM_CLASSES=classes, K=k, POS_A=.5, POS_B=.1,
            NEG_A=.2, NEG_B=.5, DETACH_NEGATIVE=True, PROXY_LR=.01,
            COUNT_AWARE=count_aware, COUNT_DAMPING=10., COUNT_MIN_ASSIGNMENTS=3)),
        SOLVER=NS(STEPS=[10], GAMMA=.1, WARMUP_FACTOR=.01, WARMUP_ITERS=0, WARMUP_METHOD='linear'),
        VALIDATION=NS(VERBOSE=1))


def test_count_aware_exact_real_batch_and_network_gradient():
    # Real E3 batch size: 15 classes, 8 examples each; several assignment counts.
    cfg = options(dim=46, classes=15, k=3)
    criterion = build_loss(cfg)
    with torch.no_grad():
        criterion.proxies.copy_(torch.eye(46)[:45].reshape(15,3,46))
    counts = torch.tensor([[1,3,4],[2,6,0]]+[[8,0,0]]*13)
    ids = torch.repeat_interleave(torch.arange(45), counts.flatten())
    labels = ids//3
    x = criterion.proxies.detach().reshape(45,46)[ids].clone()
    x[:, -1] = .4
    x.requires_grad_()
    pos, neg, details = criterion.loss_components(x, labels, return_details=True)
    assert torch.equal(details['pos_k'], ids%3)
    # Same scalar and network gradients as E1 at this exact state.
    legacy = build_loss(options(False, dim=46, classes=15, k=3))
    legacy.load_state_dict(criterion.state_dict())
    ref_x = x.detach().clone().requires_grad_()
    ref_loss = legacy(ref_x, labels)
    ref_loss.backward()
    loss = pos+neg
    assert torch.equal(loss, ref_loss)
    loss.backward()
    before = criterion.proxies.grad.clone()
    assert torch.equal(before, legacy.proxies.grad)
    grad_x = x.grad.clone()
    stats = criterion.scale_proxy_gradients(details, collect_stats=True)
    factors = torch.where(counts>0, 120/(10*counts.clamp_min(3).float()), 0.)
    assert torch.allclose(criterion.proxies.grad, before*factors.unsqueeze(-1))
    assert torch.equal(x.grad, grad_x) and torch.equal(x.grad, ref_x.grad)
    assert stats['proxy_scale_max'] == 4.0
    assert stats['proxy_scale_min'] == 1.5
    assert stats['proxy_grad_norm_after_scaling'] > stats['proxy_grad_norm_before_scaling']
    old = criterion.proxies.detach().clone()
    build_proxy_optimizer(cfg, criterion).step()
    assert torch.allclose(criterion.proxies, old-.01*before*factors.unsqueeze(-1))
    assert torch.equal(criterion.proxies[counts==0], old[counts==0])


def test_detachment_required_and_negative_only_gradient():
    cfg = options()
    c = build_loss(cfg)
    x = torch.tensor([[.8,.2]], requires_grad=True)
    _, neg = c.loss_components(x, torch.tensor([0]))
    neg.backward()
    assert c.proxies.grad is None and x.grad.norm()>0
    cfg.LOSSES.PROXY_CBML_LOSS.DETACH_NEGATIVE = False
    with pytest.raises(ValueError, match='DETACH_NEGATIVE'):
        build_loss(cfg)


def test_instance_geometry_matches_explicit_pairs():
    x = F.normalize(torch.tensor([[1.,.1],[.8,.6],[.1,1.],[.5,.5]]), dim=1).numpy()
    labels = np.array([0,0,1,1]); sim = x@x.T
    before = sim.copy()
    got = instance_geometry(sim, labels)
    same = (labels[:,None]==labels[None,:]) & ~np.eye(4,dtype=bool)
    other = labels[:,None]!=labels[None,:]
    for key, mask in [('same',same),('different',other)]:
        assert got[key+'_pair_count']==mask.sum()
        assert got[key+'_pair_mean']==pytest.approx(sim[mask].mean())
        assert got[key+'_pair_std']==pytest.approx(sim[mask].std())
    gaps=np.where(same,sim,-np.inf).max(1)-np.where(other,sim,-np.inf).max(1)
    assert got['instance_gap_mean']==pytest.approx(gaps.mean())
    assert got['instance_gap_p10']==pytest.approx(np.percentile(gaps,10))
    assert np.array_equal(sim,before)


def diagnostic_config(tmp_path):
    cfg = options()
    cfg.SAVE_DIR = str(tmp_path)
    cfg.dump = lambda:'test: true\n'
    cfg.DATA = NS(NUM_WORKERS=0,TRAIN_BATCHSIZE=2,TEST_BATCHSIZE=4,TRAIN_IMG_SOURCE='synthetic',TEST_IMG_SOURCE='synthetic')
    cfg.DIAGNOSTICS = NS(STATS_EVERY=2,TRAIN_RECALL_EVERY=1,FAST_RECALL=False,FINAL_EVAL=True,INSTANCE_GEOMETRY=True)
    return cfg


def test_negative_snapshot_counts_and_weights(tmp_path):
    cfg = diagnostic_config(tmp_path)
    c = build_loss(cfg)
    with torch.no_grad():
        # For each class, the unused proxy is the other class's exact direction.
        c.proxies.copy_(torch.tensor([[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]))
    x = np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],dtype=np.float32)
    y = np.array([0,0,1,1])
    diag = TrainingDiagnostics(cfg,torch.device('cpu'),logging.getLogger('test_e3_snap'))
    diag.set_initial_proxies(c)
    diag.proxy_snapshot(0,x,y,c); diag.close()
    with (diag.path/'proxy_snapshots.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['assigned_count']) for r in rows] == [2,0,2,0]
    assert [int(r['negative_selected_count']) for r in rows] == [0,2,0,2]
    expected = 2*(1/(1+np.exp(-(1-.2)/.5)))/.5
    assert float(rows[1]['negative_score_gradient_weight'])==pytest.approx(expected)
    event = [json.loads(s) for s in (diag.path/'events.jsonl').read_text().splitlines()][-1]
    assert event['unused_negative_fraction']==1.0
    assert event['unused_negative_weight_fraction']==1.0


def test_e3_trainer_and_diagnostics(tmp_path):
    cfg=diagnostic_config(tmp_path)
    logger=logging.getLogger('test_e3_train')
    class Loader(list):
        dataset=NS(label_list=['0','0','1','1'])
    x=torch.tensor([[1.,.1],[.9,.2],[.1,1.],[.2,.9]])
    y=torch.tensor([0,0,1,1]); val=Loader([(x,y)])
    train=[(x[:2],y[:2]),(x[2:],y[2:]),(x[:2],y[:2])]
    def run(diag):
        torch.manual_seed(123)
        model=torch.nn.Linear(2,2,bias=False)
        c=build_loss(cfg)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        po=build_proxy_optimizer(cfg,c); sch=build_lr_scheduler(cfg,opt)
        ck=Checkpointer(model,opt,sch,criterion=c,proxy_optimizer=po)
        if diag: diag.set_initial_proxies(c)
        do_train(cfg,model,train,val,opt,sch,c,None,ck,torch.device('cpu'),1,{'iteration':0},logger,
            proxy_optimizer=po,diagnostics=diag,train_eval_loader=val if diag else None)
        return model.weight.detach().clone(),c.proxies.detach().clone()
    a=run(None)
    diag=TrainingDiagnostics(cfg,torch.device('cpu'),logger); b=run(diag); diag.close()
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    steps=[json.loads(s) for s in (diag.path/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==3
    for s in steps[:2]:
        assert s['proxy_grad_norm_after_scaling']==pytest.approx(s['proxy_grad_norm_before_scaling']*2/30)
        assert s['unassigned_proxy_update_norm']==0.0
    assert not steps[2]['profiled']
    ev=[json.loads(s) for s in (diag.path/'events.jsonl').read_text().splitlines()]
    assert {(e['iteration'],e['split']) for e in ev if e['event']=='instance_geometry'} == {(i,s) for i in range(4) for s in ('train','validation')}
