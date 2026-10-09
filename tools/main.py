import gc
import os
import random
import argparse

import numpy as np
import torch

from cbml_benchmark.config import cfg
from cbml_benchmark.data import build_data
from cbml_benchmark.engine.trainer import do_train, do_test
from cbml_benchmark.losses import build_loss, build_aux_loss
from cbml_benchmark.modeling import build_model
from cbml_benchmark.solver import build_lr_scheduler, build_optimizer
from cbml_benchmark.utils.logger import setup_logger
from cbml_benchmark.utils.checkpoint import Checkpointer
from cbml_benchmark.utils.prototype_initializer import (
    initialize_prototypes_random,
    initialize_prototypes_mean,
    initialize_prototypes_kmeans,
)
from cbml_benchmark.utils.cache_prototypes import (
    _get_prototype_cache_path,
    load_cached_prototypes,
    save_prototype_cache,
)


# The prototype cache is keyed by the config, not by the seed / head weights, so with
# several seeds a later run could silently load k-means prototypes that were computed
# from ANOTHER run's random head.  It is therefore disabled (k-means takes seconds).
USE_PROTOTYPE_CACHE = False


def set_seed(seed, deterministic=False):
    """Seed python, numpy and torch (CPU + all GPUs).

    This fixes: the random head initialisation, the RandomIdentitySampler stream
    (it uses python `random` / `np.random` in the main process) and the
    augmentation streams of the DataLoader workers (derived from the torch seed).
    It does NOT make GPU training bit-identical unless `deterministic=True`
    (slower; some ops may be unavailable) -- with it off, expect tiny run-to-run
    differences that are far smaller than the seed-to-seed variance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def train(cfg):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)

    seed = int(cfg.SOLVER.RNG_SEED)
    deterministic = bool(cfg['SOLVER']['DETERMINISTIC']) if 'DETERMINISTIC' in cfg.SOLVER else False
    set_seed(seed, deterministic)
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    with open(os.path.join(cfg.SAVE_DIR, 'config_resolved.yaml'), 'w') as f:
        f.write(cfg.dump())                       # exact config of this run (provenance)
    logger.info(f"seed = {seed}, deterministic = {deterministic}, SAVE_DIR = {cfg.SAVE_DIR}")
    logger.info(cfg)

    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    criterion = build_loss(cfg)
    criterion_aux = None
    if cfg.LOSSES.NAME_AUX != '':
        criterion_aux = build_aux_loss(cfg)

    # initializing prototypes if using mpcbml loss
    if cfg.LOSSES.NAME == 'cbml_loss':
        if USE_PROTOTYPE_CACHE:
            cached_prototypes, cached_cluster_sizes, cache_path = load_cached_prototypes(cfg)
        else:
            cached_prototypes, cached_cluster_sizes, cache_path = None, None, None

        if cached_prototypes is not None:
            logger.info(f"Loaded cached prototypes from {cache_path}, skipping recomputation.")
            prototypes = cached_prototypes
            cluster_sizes = cached_cluster_sizes
        else:
            logger.info(f"No cache found. Initializing prototypes using {cfg.LOSSES.CBML_LOSS.INIT_METHOD}...")

            if cfg.LOSSES.CBML_LOSS.INIT_METHOD == 'kmeans':
                # k-means seed follows the run seed (it was fixed to 0 before)
                prototypes, cluster_sizes = initialize_prototypes_kmeans(model=model, cfg=cfg, seed=seed)
            elif cfg.LOSSES.CBML_LOSS.INIT_METHOD == 'mean':
                prototypes = initialize_prototypes_mean(model=model, cfg=cfg)
                cluster_sizes = None
            elif cfg.LOSSES.CBML_LOSS.INIT_METHOD == 'random':
                prototypes = initialize_prototypes_random(
                    num_classes=cfg.LOSSES.CBML_LOSS.N_CLASSES,
                    prototype_per_class=cfg.LOSSES.CBML_LOSS.PROTOTYPE_PER_CLASS,
                    embed_dim=cfg.MODEL.HEAD.DIM,
                    device=device
                )
                cluster_sizes = None
            else:
                raise ValueError(f"Unknown initializing method: {cfg.LOSSES.CBML_LOSS.INIT_METHOD}")

            if USE_PROTOTYPE_CACHE:
                _, key_dict = _get_prototype_cache_path(cfg)
                save_prototype_cache(cache_path, key_dict, prototypes, cluster_sizes)
                logger.info(f"Saved prototype cache to {cache_path}")

        criterion.set_prototypes(prototypes)
        del prototypes
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Prototype initialization complete.")

    loss_param = None
    if cfg.LOSSES.NAME == 'softtriple_loss' or cfg.LOSSES.NAME == 'proxynca_loss' or cfg.LOSSES.NAME == 'center_loss' or cfg.LOSSES.NAME == 'adv_loss' or cfg.LOSSES.NAME == 'cbml_loss':
        loss_param = criterion
    if cfg.LOSSES.NAME_AUX == 'softtriple_loss' or cfg.LOSSES.NAME_AUX == 'proxynca_loss' or cfg.LOSSES.NAME_AUX == 'center_loss' or cfg.LOSSES.NAME_AUX == 'adv_loss':
        loss_param = criterion_aux

    optimizer = build_optimizer(cfg, model, loss_param=loss_param)
    scheduler = build_lr_scheduler(cfg, optimizer)

    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)
    eval_train_loader = build_data(cfg, is_train=True, is_eval=True)

    logger.info(train_loader.dataset)
    logger.info(val_loader.dataset)

    arguments = dict()
    arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    checkpointer = Checkpointer(model, optimizer, scheduler, cfg.SAVE_DIR)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        eval_train_loader,
        optimizer,
        scheduler,
        criterion,
        criterion_aux,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        logger
    )


def test(cfg):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    val_loader = build_data(cfg, is_train=False)
    logger.info(val_loader.dataset)

    do_test(
        model,
        val_loader,
        logger
    )


def parse_args():
    """
    Parse input arguments.  Any config key can be overridden from the command
    line after the known flags, e.g.
        python3 tools/main.py --cfg configs/x.yaml --phase train \
            SOLVER.RNG_SEED 2 SAVE_DIR output/seed2
    """
    parser = argparse.ArgumentParser(description='Train a retrieval network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='config file',
        default=None,
        type=str)
    parser.add_argument(
        '--phase',
        dest='train_test',
        help='train or test',
        default='train',
        type=str)
    parser.add_argument(
        'opts',
        help='Modify config options from the command line (KEY VALUE pairs)',
        default=None,
        nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.train_test == 'train':
        train(cfg)
    else:
        test(cfg)
