import argparse
import time
import random
import numpy as np
import torch

from cbml_benchmark.config import cfg
from cbml_benchmark.data import build_data
from cbml_benchmark.data.build import build_train_eval_data
from cbml_benchmark.engine.trainer import do_train, do_test
from cbml_benchmark.losses import build_loss, build_aux_loss
from cbml_benchmark.modeling import build_model
from cbml_benchmark.solver import (
    build_lr_scheduler, build_optimizer, build_proxy_optimizer,
)
from cbml_benchmark.utils.logger import setup_logger
from cbml_benchmark.utils.checkpoint import Checkpointer
from cbml_benchmark.utils.proxy_init import configure_proxy_classes, initialize_proxies
from cbml_benchmark.utils.training_diagnostics import TrainingDiagnostics, sync


def train(cfg, resume=None, seed=None):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    device = torch.device(cfg.MODEL.DEVICE)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    diagnostics = TrainingDiagnostics(cfg, device, logger) if cfg.DIAGNOSTICS.ENABLED else None
    try:
        if diagnostics:
            diagnostics.event("seed", explicit_seed=seed, note="Seed is optional; does not guarantee GPU determinism or exact resume.")
        _train(cfg, resume, logger, device, diagnostics)
    except BaseException as exc:
        if diagnostics:
            diagnostics.event("training_error", error=repr(exc))
        raise
    finally:
        if diagnostics:
            diagnostics.close()


def _train(cfg, resume, logger, device, diagnostics):
    start = time.perf_counter()
    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)
    train_eval_loader = None
    if diagnostics and cfg.DIAGNOSTICS.TRAIN_RECALL:
        train_eval_loader = build_train_eval_data(cfg)
    if diagnostics:
        diagnostics.event("data_setup", total_s=time.perf_counter()-start,
                          train_images=len(train_loader.dataset), val_images=len(val_loader.dataset),
                          train_classes=len(set(train_loader.dataset.label_list)),
                          val_classes=len(set(val_loader.dataset.label_list)))
    use_proxies = cfg.LOSSES.NAME == 'proxy_cbml_loss'
    class_ids = None
    if use_proxies:
        if cfg.LOSSES.NAME_AUX:
            raise ValueError("ProxyCBMLLoss must be the sole loss for this integration.")
        class_ids = configure_proxy_classes(cfg, train_loader.dataset)
    elif cfg.LOSSES.NAME_AUX == 'proxy_cbml_loss':
        raise ValueError("ProxyCBMLLoss is supported as the primary loss only.")

    logger.info(cfg)
    model_cfg = cfg.clone()
    if resume:
        model_cfg.MODEL.PRETRAIN = resume
    start = time.perf_counter()
    model = build_model(model_cfg).to(device)
    if diagnostics:
        sync(device)
        diagnostics.event("model_setup", total_s=time.perf_counter()-start)
    criterion = build_loss(cfg).to(device)
    criterion_aux = build_aux_loss(cfg).to(device) if cfg.LOSSES.NAME_AUX else None

    proxy_optimizer = None
    loss_param = None
    if use_proxies:
        if not resume:
            initialize_proxies(cfg, model, criterion, device, diagnostics=diagnostics)
        proxy_optimizer = build_proxy_optimizer(cfg, criterion)
    else:
        # Preserve the original optimizer path for legacy registered losses.
        learnable = ('softtriple_loss', 'proxynca_loss', 'center_loss', 'adv_loss')
        if cfg.LOSSES.NAME in learnable:
            loss_param = criterion
        if cfg.LOSSES.NAME_AUX in learnable:
            loss_param = criterion_aux
    optimizer = build_optimizer(cfg, model, loss_param=loss_param)
    scheduler = build_lr_scheduler(cfg, optimizer)

    logger.info(train_loader.dataset)
    logger.info(val_loader.dataset)
    arguments = {"iteration": 0}
    if use_proxies:
        arguments["class_ids"] = class_ids
        arguments["proxy_config"] = dict(cfg.LOSSES.PROXY_CBML_LOSS)
        arguments["proxy_config"]["DIM"] = cfg.MODEL.HEAD.DIM
    checkpointer = Checkpointer(
        model, optimizer, scheduler, cfg.SAVE_DIR,
        criterion=criterion if use_proxies else None,
        proxy_optimizer=proxy_optimizer,
    )
    if resume:
        state = checkpointer.load(resume, expected_classes=class_ids,
                                  require_proxy_state=use_proxies)
        if use_proxies:
            saved_config = dict(state.get("proxy_config") or {})
            # Pre-E3 checkpoints used the original batch-averaged rule.
            saved_config.setdefault("COUNT_AWARE", False)
            saved_config.setdefault("COUNT_DAMPING", 10.0)
            saved_config.setdefault("COUNT_MIN_ASSIGNMENTS", 3)
            # Checkpoint compatibility for runs created before Proxy-MVC fields existed.
            saved_config.setdefault("MVC_ENABLED", False)
            saved_config.setdefault("MVC_WEIGHT", 1.0)
            saved_config.setdefault("MVC_GAMMA", 0.0)
            saved_config.setdefault("MVC_MODE", "symmetric")
            saved_config.setdefault("MVC_TAIL_DELTA", 0.0)
            if saved_config != arguments["proxy_config"]:
                raise ValueError("Proxy configuration differs from the checkpoint; use the original training config.")
        for key in ('iteration', 'best_iteration', 'best_recall'):
            if key in state:
                arguments[key] = state[key]
        if not 0 <= arguments['iteration'] <= len(train_loader):
            raise ValueError("Checkpoint iteration is outside the training budget.")

    if diagnostics:
        diagnostics.set_initial_proxies(criterion)
        (diagnostics.path / "resolved_config.yaml").write_text(cfg.dump(), encoding="utf-8")
        diagnostics.event("training_ready", iteration=arguments["iteration"],
                          resume=resume, proxy_reference="resume" if resume else "initialization")
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler,
             criterion, criterion_aux, checkpointer, device,
             cfg.SOLVER.CHECKPOINT_PERIOD, arguments, logger,
             proxy_optimizer=proxy_optimizer, diagnostics=diagnostics,
             train_eval_loader=train_eval_loader)


def test(cfg):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = build_model(cfg).to(torch.device(cfg.MODEL.DEVICE))
    val_loader = build_data(cfg, is_train=False)
    logger.info(val_loader.dataset)
    do_test(model, val_loader, logger)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a retrieval network')
    parser.add_argument('--cfg', dest='cfg_file', required=True, type=str)
    parser.add_argument('--phase', dest='train_test', choices=['train', 'test'], default='train')
    parser.add_argument('--resume', default=None, help='Training checkpoint path')
    parser.add_argument('--seed', type=int, default=None, help='Optional Python, NumPy and PyTorch seed (fresh comparison runs)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    if args.train_test == 'train':
        train(cfg, resume=args.resume, seed=args.seed)
    else:
        if args.resume:
            raise ValueError('--resume is for training; use MODEL.PRETRAIN for testing.')
        test(cfg)
