import argparse
import torch

from cbml_benchmark.config import cfg
from cbml_benchmark.data import build_data
from cbml_benchmark.engine.trainer import do_train, do_test
from cbml_benchmark.losses import build_loss, build_aux_loss
from cbml_benchmark.modeling import build_model
from cbml_benchmark.solver import (
    build_lr_scheduler, build_optimizer, build_proxy_optimizer,
)
from cbml_benchmark.utils.logger import setup_logger
from cbml_benchmark.utils.checkpoint import Checkpointer
from cbml_benchmark.utils.proxy_init import configure_proxy_classes, initialize_proxies


def train(cfg, resume=None):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    device = torch.device(cfg.MODEL.DEVICE)
    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)
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
    model = build_model(model_cfg).to(device)
    criterion = build_loss(cfg).to(device)
    criterion_aux = build_aux_loss(cfg).to(device) if cfg.LOSSES.NAME_AUX else None

    proxy_optimizer = None
    loss_param = None
    if use_proxies:
        if not resume:
            initialize_proxies(cfg, model, criterion, device)
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
        if use_proxies and state.get("proxy_config") != arguments["proxy_config"]:
            raise ValueError("Proxy configuration differs from the checkpoint; use the original training config.")
        for key in ('iteration', 'best_iteration', 'best_recall'):
            if key in state:
                arguments[key] = state[key]
        if not 0 <= arguments['iteration'] <= len(train_loader):
            raise ValueError("Checkpoint iteration is outside the training budget.")

    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler,
             criterion, criterion_aux, checkpointer, device,
             cfg.SOLVER.CHECKPOINT_PERIOD, arguments, logger,
             proxy_optimizer=proxy_optimizer)


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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    if args.train_test == 'train':
        train(cfg, resume=args.resume)
    else:
        if args.resume:
            raise ValueError('--resume is for training; use MODEL.PRETRAIN for testing.')
        test(cfg)
