"""Numerical and noninterference checks for the extended diagnostic patch."""
import json
import logging
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from cbml_benchmark.utils.mvc_extended_diagnostics import (
    component_gradients, embedding_geometry, proxy_score_geometry,
)
from cbml_benchmark.utils.training_diagnostics import TrainingDiagnostics
from cbml_benchmark.engine.trainer import do_train
from cbml_benchmark.losses import build_loss
from cbml_benchmark.solver import build_proxy_optimizer, build_lr_scheduler
from cbml_benchmark.utils.checkpoint import Checkpointer


def test_gradient_norm_and_cosine():
    model = torch.nn.Linear(2, 1, bias=False)
    p = model(torch.tensor([[1., 0.]])).sum()
    n = model(torch.tensor([[0., 2.]])).sum()
    v = model(torch.tensor([[-1., 1.]])).sum()
    stats = component_gradients(model,p,n,v,10.)
    assert model.weight.grad is None
    assert stats['component_pos_grad_norm'] == pytest.approx(1.)
    assert stats['component_neg_grad_norm'] == pytest.approx(2.)
    assert stats['component_main_grad_norm'] == pytest.approx(np.sqrt(5))
    assert stats['component_mvc_main_cosine'] == pytest.approx(1/np.sqrt(10))
    assert stats['component_weighted_mvc_grad_norm'] == pytest.approx(10*np.sqrt(2))
    (p+n+10*v).backward()
    assert stats['component_total_grad_norm'] == pytest.approx(float(model.weight.grad.norm()))
    stats = component_gradients(model,model.weight.sum(),model.weight.sum(),model.weight.sum()*0,0.)
    assert stats['component_mvc_main_cosine'] is None


def test_spectrum_and_scatter():
    x = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    s = embedding_geometry(x,[0,0,1,1])
    assert s['eigenvalues'] == pytest.approx([.5,.5])
    assert s['effective_rank'] == pytest.approx(2.)
    assert s['within_class_scatter'] + s['between_class_scatter'] == pytest.approx(s['covariance_trace'])
    assert s['between_class_scatter'] == pytest.approx(0.)
    s = embedding_geometry(x,[0,1,2,3])
    assert s['within_class_scatter'] == pytest.approx(0.)
    assert s['between_class_scatter'] == pytest.approx(1.)
    assert embedding_geometry(np.ones((3,2)),[0,0,0])['effective_rank'] is None


def test_proxy_scores_exclude_only_training_own_class():
    x = np.array([[1.,0.],[0.,1.]])
    bank = np.array([[[1.,0.],[0.,1.]],[[-1.,0.],[0.,-1.]]])
    train = proxy_score_geometry(x,[0,1],bank,'train')
    assert train['proxies_per_anchor'] == 2
    assert train['within_anchor_variance_mean'] == pytest.approx(.25)
    val = proxy_score_geometry(x,[0,999],bank,'validation')
    assert val['proxies_per_anchor'] == 4
    assert val['within_anchor_variance_mean'] == pytest.approx(.5)
    assert val['pooled_score_std'] == pytest.approx(np.sqrt(.5))


@pytest.mark.parametrize('weight', [0.,1.,10.])
def test_training_noninterference(tmp_path,weight):
    cfg = NS(SAVE_DIR=str(tmp_path), dump=lambda:'test: true\n',
             DATA=NS(NUM_WORKERS=0,TRAIN_BATCHSIZE=2,TEST_BATCHSIZE=4,TRAIN_IMG_SOURCE='synthetic',TEST_IMG_SOURCE='synthetic'),
             DIAGNOSTICS=NS(STATS_EVERY=1,TRAIN_RECALL_EVERY=1,FAST_RECALL=True,FINAL_EVAL=True,
                            INSTANCE_GEOMETRY=True,EXTENDED_GEOMETRY=True,COMPONENT_GRADIENTS=True),
             MODEL=NS(HEAD=NS(DIM=2)), LOSSES=NS(NAME='proxy_cbml_loss',NAME_AUX='',PROXY_CBML_LOSS=NS(
                 NUM_CLASSES=2,K=2,POS_A=.5,POS_B=.1,NEG_A=.2,NEG_B=.5,DETACH_NEGATIVE=True,PROXY_LR=.01,
                 COUNT_AWARE=True,COUNT_DAMPING=10.,COUNT_MIN_ASSIGNMENTS=3,MVC_ENABLED=weight>0,MVC_WEIGHT=weight,MVC_GAMMA=0.)),
             SOLVER=NS(STEPS=[10],GAMMA=.1,WARMUP_FACTOR=.01,WARMUP_ITERS=0,WARMUP_METHOD='linear'),VALIDATION=NS(VERBOSE=1))
    logger = logging.getLogger('extended_diag_test')
    class Loader(list):
        dataset = NS(label_list=['0','0','1','1'])
    images=torch.tensor([[1.,.1],[.9,.2],[.1,1.],[.2,.9]])
    val=Loader([(images,torch.tensor([0,0,1,1]))])
    train=[(images[:2],torch.tensor([0,0])),(images[2:],torch.tensor([1,1]))]
    def run(diag):
        torch.manual_seed(123)
        model=torch.nn.Linear(2,2,bias=False)
        crit=build_loss(cfg)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        po=build_proxy_optimizer(cfg,crit)
        sch=build_lr_scheduler(cfg,opt)
        ck=Checkpointer(model,opt,sch,criterion=crit,proxy_optimizer=po)
        if diag:
            diag.set_initial_proxies(crit)
        do_train(cfg,model,train,val,opt,sch,crit,None,ck,torch.device('cpu'),1,{'iteration':0},logger,
                 proxy_optimizer=po,diagnostics=diag,train_eval_loader=val if diag else None)
        return model.weight.detach().clone(),crit.proxies.detach().clone(),torch.random.get_rng_state()
    a=run(None)
    diag=TrainingDiagnostics(cfg,torch.device('cpu'),logger)
    b=run(diag)
    diag.close()
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    steps=[json.loads(s) for s in (diag.path/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==2 and all(s['unassigned_proxy_update_norm']==0 for s in steps)
    for s in steps:
        assert s['component_weighted_mvc_grad_norm'] == pytest.approx(weight*s['component_mvc_grad_norm'])
        assert s['component_total_grad_norm'] == pytest.approx(s['other_grad_norm'],rel=1e-5)
        assert s['mvc_variance_mean'] is not None
        if weight==0:
            assert s['loss_mvc']==0 and s['component_mvc_grad_norm']>0
    events=[json.loads(s) for s in (diag.path/'events.jsonl').read_text().splitlines()]
    for event in ('embedding_spectrum','proxy_score_geometry'):
        assert {(e['iteration'],e['split']) for e in events if e['event']==event} == {(i,s) for i in range(3) for s in ('train','validation')}
    assert sum(e['event']=='proxy_spectrum' for e in events)==3
