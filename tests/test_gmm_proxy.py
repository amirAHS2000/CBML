"""Synthetic CPU invariants of previous_works.pdf Eqs. 1--35 and no foreign updates."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from cbml_benchmark.losses.gmm_proxy import (
    GaussianMixtureProxyLoss,
    project_simplex_with_floor,
)
from cbml_benchmark.utils.gmm_diagnostics import gmm_proxy_snapshot, sampled_gmm_stats


def config(c=3,k=2,dim=4,beta=5):
    return SimpleNamespace(
        MODEL=SimpleNamespace(HEAD=SimpleNamespace(DIM=dim)),
        LOSSES=SimpleNamespace(GMM_PROXY_LOSS=SimpleNamespace(
            NUM_CLASSES=c,K=k,BETA_INIT=beta,BETA_LR=.0001,PROXY_LR=.01,
            WEIGHT_LR=.001,WEIGHT_FLOOR=.0001,INIT_METHOD='kmeans')))


def make_loss():
    torch.manual_seed(23)
    return GaussianMixtureProxyLoss(config())


def test_log_odds_equals_hard_component_gaussian_posterior():
    obj=make_loss()
    x=F.normalize(torch.randn(8,4),dim=-1)
    labels=torch.tensor([0,0,0,0,1,1,1,1])
    with torch.no_grad():
        obj.mixture_weights[:] = torch.tensor([[.3,.7],[.6,.4],[.9,.1]])
    loss,details=obj.loss_and_details(x,labels,return_details=True)
    beta=float(obj.beta.detach())
    scores=[]
    for i,xi in enumerate(x):
        values=[]
        for c in range(obj.num_classes):
            values.append([math.log(1/obj.num_classes)+math.log(float(obj.mixture_weights[c,k].detach()))
                         - .5*beta*float((xi-obj.proxies[c,k]).square().sum().detach())
                         for k in range(obj.k)])
        p=max(values[int(labels[i])]);n=max(max(v) for c,v in enumerate(values) if c!=labels[i])
        scores.append((p,n))
    expected=torch.tensor([math.log1p(math.exp(n-p)) for p,n in scores])
    assert torch.allclose(loss,expected.mean(),atol=1e-6)
    assert torch.allclose(details['posterior'],torch.sigmoid(-details['log_odds']))


def test_negative_only_class_means_and_weights_do_not_receive_gradients():
    obj=make_loss()
    x=F.normalize(torch.randn(9,4),dim=1).requires_grad_()
    labels=torch.zeros(9,dtype=torch.long)
    loss,details=obj.loss_and_details(x,labels,return_details=True)
    loss.backward()
    assert x.grad.norm()>0
    assert obj.log_beta.grad is not None
    assert torch.equal(obj.proxies.grad[1:],torch.zeros_like(obj.proxies.grad[1:]))
    assert torch.equal(obj.mixture_weights.grad[1:],torch.zeros_like(obj.mixture_weights.grad[1:]))
    old_means=obj.proxies.detach().clone()
    old_weights=obj.mixture_weights.detach().clone()
    optimizer=torch.optim.SGD([obj.proxies,obj.log_beta],lr=.01,weight_decay=0.,momentum=0.)
    optimizer.step()
    obj.step_mixture_weights()
    assert torch.equal(obj.proxies[1:].detach(),old_means[1:])
    assert torch.equal(obj.mixture_weights[1:].detach(),old_weights[1:])


def test_weight_tangent_gradient_and_simplex_projection():
    w=torch.tensor([[.99,.005,.005],[.2,.3,.5],[1.,0.,0.]])
    proj=project_simplex_with_floor(w,1e-4)
    assert torch.allclose(proj.sum(1),torch.ones(3),atol=1e-6)
    assert bool((proj>=1e-4-1e-8).all())
    assert torch.isfinite(proj.log()).all()
    obj=make_loss()
    obj.mixture_weights.grad=torch.tensor([[1.,-1.],[0.,0.],[0.,0.]])
    before=obj.mixture_weights.detach().clone()
    stats=obj.step_mixture_weights(collect_stats=True)
    assert torch.equal(obj.mixture_weights.detach()[1:],before[1:])
    assert torch.allclose(obj.mixture_weights.detach()[0], before[0]-obj.weight_lr*torch.tensor([1.,-1.]))
    assert stats['mixture_weight_sum_error']<1e-6


def test_selection_uses_weights_not_just_nearest_cosine():
    obj=GaussianMixtureProxyLoss(config(c=2,k=2,dim=2,beta=1.))
    with torch.no_grad():
        obj.proxies[:]=torch.tensor([[[1.,0.],[.99,.01]],[[-1.,0.],[-.9,.1]]])
        obj.mixture_weights[:]=torch.tensor([[.01,.99],[.5,.5]])
    k,nc,nk=obj.select_components(torch.tensor([[1.,0.]]),torch.tensor([0]))
    assert int(k[0])==1  # the slightly farther but 99x more probable component
    assert int(nc[0])==1


def test_one_class_k1_remains_valid():
    obj=GaussianMixtureProxyLoss(config(c=2,k=1,dim=3))
    x=torch.randn(4,3)
    loss=obj(x,torch.tensor([0,0,1,1]));loss.backward()
    obj.step_mixture_weights()
    assert torch.equal(obj.mixture_weights.detach(),torch.ones_like(obj.mixture_weights))
    assert torch.isfinite(loss)


def test_log_beta_is_learnable_and_finite():
    obj=make_loss()
    x=torch.randn(10,4)
    loss=obj(x,torch.tensor([0,0,0,1,1,1,2,2,2,2]))
    loss.backward()
    assert torch.isfinite(obj.log_beta.grad)
    assert torch.abs(obj.log_beta.grad)>0


def test_gmm_diagnostics_match_selected_component_scores(tmp_path):
    obj=make_loss()
    x=F.normalize(torch.randn(15,4),dim=1)
    labels=torch.tensor([0]*5+[1]*5+[2]*5)
    loss,details=obj.loss_and_details(x,labels,return_details=True)
    loss.backward()
    sampled=sampled_gmm_stats(details,obj)
    assert abs(sampled['posterior_mean']-float(details['posterior'].mean()))<1e-7
    assert sampled['foreign_only_proxy_grad_norm']==0.0  # all classes present, empty mask
    class Stub:
        def __init__(self):
            self.initial_proxies=None
            self.proxy_csv=SimpleNamespace(writerow=lambda row: self.rows.append(row))
            self.proxy_file=SimpleNamespace(flush=lambda: None)
            self.rows=[];self.events=[]
        def event(self,name,**data):self.events.append(dict(event=name,**data))
    diag=Stub()
    gmm_proxy_snapshot(diag,0,x.numpy(),labels.numpy(),obj)
    assert len(diag.rows)==obj.num_classes*obj.k
    assert all(len(row)==13 for row in diag.rows)
    assert sum(row[3] for row in diag.rows)==len(x)
    assert sum(row[10] for row in diag.rows)==len(x)
    assert diag.events[-1]['mixture_weight_sum_error'] < 1e-6


def test_posterior_loss_matches_two_class_softplus():
    obj=make_loss()
    x=torch.randn(12,4)
    labels=torch.tensor([0]*6+[1]*6)
    v,details=obj.loss_and_details(x,labels,True)
    expected=F.softplus(details['negative_log_score']-details['positive_log_score']).mean()
    assert torch.allclose(v.detach(),expected,atol=1e-6)


def test_gmm_trainer_end_to_end_and_logging(tmp_path):
    import csv
    import json
    import logging
    from cbml_benchmark.losses import build_loss
    from cbml_benchmark.solver import build_proxy_optimizer, build_lr_scheduler
    from cbml_benchmark.engine.trainer import do_train
    from cbml_benchmark.utils.checkpoint import Checkpointer
    from cbml_benchmark.utils.training_diagnostics import TrainingDiagnostics
    from types import SimpleNamespace as NS

    opts=NS(NUM_CLASSES=2,K=2,BETA_INIT=10.,BETA_LR=.0001,PROXY_LR=.01,
            WEIGHT_LR=.001,WEIGHT_FLOOR=.0001,INIT_METHOD='random')
    cfg=NS(SAVE_DIR=str(tmp_path),dump=lambda:'gmm_test: true\n',
        MODEL=NS(HEAD=NS(DIM=2)),LOSSES=NS(NAME='gmm_proxy_loss',NAME_AUX='',GMM_PROXY_LOSS=opts),
        SOLVER=NS(STEPS=[10],GAMMA=.1,WARMUP_FACTOR=.01,WARMUP_ITERS=0,WARMUP_METHOD='linear'),
        VALIDATION=NS(VERBOSE=1),
        DATA=NS(NUM_WORKERS=0,TRAIN_BATCHSIZE=2,TEST_BATCHSIZE=4,
                TRAIN_IMG_SOURCE='synthetic',TEST_IMG_SOURCE='synthetic'),
        DIAGNOSTICS=NS(STATS_EVERY=2,TRAIN_RECALL_EVERY=1,FAST_RECALL=False,
                       FINAL_EVAL=True,INSTANCE_GEOMETRY=True))
    class Loader(list):
        dataset=NS(label_list=['0','0','1','1'])
    images=torch.tensor([[1.,.1],[.9,.2],[.1,1.],[.2,.9]])
    targets=torch.tensor([0,0,1,1])
    val=Loader([(images,targets)])
    train=[(images[:2],targets[:2]),(images[2:],targets[2:]),(images[:2],targets[:2])]
    logger=logging.getLogger('test_gmm_trainer')
    def run(diag):
        torch.manual_seed(2026)
        model=torch.nn.Linear(2,2,bias=False)
        criterion=build_loss(cfg)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        proxy_optimizer=build_proxy_optimizer(cfg,criterion)
        scheduler=build_lr_scheduler(cfg,optimizer)
        checkpointer=Checkpointer(model,optimizer,scheduler,criterion=criterion,proxy_optimizer=proxy_optimizer)
        if diag:diag.set_initial_proxies(criterion)
        do_train(cfg,model,train,val,optimizer,scheduler,criterion,None,checkpointer,
                 torch.device('cpu'),1,{'iteration':0},logger,
                 proxy_optimizer=proxy_optimizer,diagnostics=diag,
                 train_eval_loader=val if diag else None)
        return [model.weight.detach().clone(),criterion.proxies.detach().clone(),
                criterion.mixture_weights.detach().clone(),criterion.log_beta.detach().clone()]
    normal=run(None)
    diag=TrainingDiagnostics(cfg,torch.device('cpu'),logger)
    with_diag=run(diag)
    diag.close()
    for a,b in zip(normal,with_diag):
        assert torch.equal(a,b)
    rows=list(csv.DictReader((diag.path/'recall.csv').open()))
    assert {(int(r['iteration']),r['split']) for r in rows}=={
        (i,s) for i in range(4) for s in ('train','validation')}
    proxy_rows=list(csv.DictReader((diag.path/'proxy_snapshots.csv').open()))
    assert all(row['mixture_weight']!='' for row in proxy_rows)
    assert all(float(row['mixture_weight'])>0 for row in proxy_rows)
    steps=[json.loads(line) for line in (diag.path/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==3
    assert all(step['loss_pos'] is None and step['loss_neg'] is None for step in steps)
    assert all(step['unassigned_proxy_update_norm']==0 for step in steps if step['profiled'])
    assert all(step['foreign_only_proxy_update_norm']==0 for step in steps if step['profiled'])
    assert all(step['foreign_only_weight_update_norm']==0 for step in steps if step['profiled'])
    assert steps[0]['posterior_mean'] is not None
    assert steps[0]['mixture_weight_sum_error']<1e-6
    assert len([e for e in (diag.path/'events.jsonl').read_text().splitlines() if '"event": "proxy_snapshot"' in e])==4
