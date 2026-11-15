import csv
import json
import logging
import random
from types import SimpleNamespace as NS

import numpy as np
import torch

from cbml_benchmark.data.evaluations import RetMetric
from cbml_benchmark.utils.training_diagnostics import (TrainingDiagnostics, preserve_rng,
    recall_from_matrix)
from cbml_benchmark.engine.trainer import do_train
from cbml_benchmark.solver import build_proxy_optimizer, build_lr_scheduler
from cbml_benchmark.utils.checkpoint import Checkpointer
from cbml_benchmark.losses import build_loss


def test_fast_recall_exact():
    rng = np.random.RandomState(9)
    x = rng.normal(size=(32, 8)).astype(np.float32)
    x /= np.linalg.norm(x,axis=1,keepdims=True)
    x[1] = x[0]  # ties
    labels = np.repeat(np.arange(8),4)
    metric = RetMetric(x,labels)
    assert recall_from_matrix(metric.sim_mat,labels) == [metric.recall_k(k) for k in (1,2,4,8)]


def test_rng_preservation():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    with preserve_rng(torch.device('cpu')):
        expected = random.random(), np.random.rand(), torch.rand(2)
    actual = random.random(), np.random.rand(), torch.rand(2)
    assert expected[:2] == actual[:2] and torch.equal(expected[2],actual[2])


def test_diagnostics_no_parameter_change_and_logs(tmp_path):
    cfg = NS(SAVE_DIR=str(tmp_path), dump=lambda:'test: true\n',
             DATA=NS(NUM_WORKERS=0,TRAIN_BATCHSIZE=2,TEST_BATCHSIZE=4,TRAIN_IMG_SOURCE='synthetic',TEST_IMG_SOURCE='synthetic'),
             DIAGNOSTICS=NS(STATS_EVERY=1,TRAIN_RECALL_EVERY=1,FAST_RECALL=True,FINAL_EVAL=True),
             MODEL=NS(HEAD=NS(DIM=2)), LOSSES=NS(NAME='proxy_cbml_loss',NAME_AUX='',PROXY_CBML_LOSS=NS(
                 NUM_CLASSES=2,K=2,POS_A=.5,POS_B=.1,NEG_A=.2,NEG_B=.1,DETACH_NEGATIVE=True,PROXY_LR=.01)),
             SOLVER=NS(STEPS=[10],GAMMA=.1,WARMUP_FACTOR=.01,WARMUP_ITERS=0,WARMUP_METHOD='linear'),VALIDATION=NS(VERBOSE=1))
    logger = logging.getLogger('test_diag')
    class Loader(list):
        dataset = NS(label_list=['0','0','1','1'])
    images=torch.tensor([[1.,.1],[.9,.2],[.1,1.],[.2,.9]])
    val=Loader([(images,torch.tensor([0,0,1,1]))])
    train=[(images[:2],torch.tensor([0,0])),(images[2:],torch.tensor([1,1]))]
    def run(diagnostic):
        torch.manual_seed(123)
        model=torch.nn.Linear(2,2,bias=False)
        criterion=build_loss(cfg)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        po=build_proxy_optimizer(cfg,criterion)
        sch=build_lr_scheduler(cfg,opt)
        ck=Checkpointer(model,opt,sch,criterion=criterion,proxy_optimizer=po)
        if diagnostic:
            diagnostic.set_initial_proxies(criterion)
        do_train(cfg,model,train,val,opt,sch,criterion,None,ck,torch.device('cpu'),1,{'iteration':0},logger,
                 proxy_optimizer=po,diagnostics=diagnostic,train_eval_loader=val if diagnostic else None)
        return model.weight.detach().clone(),criterion.proxies.detach().clone()
    a=run(None)
    diag=TrainingDiagnostics(cfg,torch.device('cpu'),logger)
    b=run(diag)
    diag.close()
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    with (diag.path/'recall.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert {(int(r['iteration']),r['split']) for r in rows} == {(i,s) for i in range(3) for s in ('train','validation')}
    steps=[json.loads(s) for s in (diag.path/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==2 and all(s['unassigned_proxy_update_norm']==0 for s in steps)
    assert all(s['loss_pos'] is not None and s['loss_neg'] is not None for s in steps)
    assert (diag.path/'proxy_snapshots.csv').exists()
