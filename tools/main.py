import gc
import json
import argparse
import torch

from cbml_benchmark.config import cfg
from cbml_benchmark.data import build_data
from cbml_benchmark.engine.trainer import do_train, do_test
from cbml_benchmark.losses import build_loss,build_aux_loss
from cbml_benchmark.modeling import build_model
from cbml_benchmark.solver import build_lr_scheduler, build_optimizer
from cbml_benchmark.utils.logger import setup_logger
from cbml_benchmark.utils.checkpoint import Checkpointer
from cbml_benchmark.utils.prototype_initializer import (
    initialize_prototypes_random,
    initialize_prototypes_mean,
    initialize_prototypes_kmeans,
)
from cbml_benchmark.utils.reproducibility import seed_everything
from cbml_benchmark.utils.cache_prototypes import (
    _get_prototype_cache_path,
    load_cached_prototypes,
    save_prototype_cache,
)


def train(cfg):
    seed_everything(cfg.SOLVER.RNG_SEED, deterministic=cfg.SOLVER.DETERMINISTIC)
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    logger.info(f'Reproducibility | seed={cfg.SOLVER.RNG_SEED} | deterministic={cfg.SOLVER.DETERMINISTIC}')
    logger.info(f'Loss configuration | lambda_reg={cfg.LOSSES.MPCBML_LOSS.LAMBDA_REG} | gamma_reg={cfg.LOSSES.MPCBML_LOSS.GAMMA_REG}')
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    criterion = build_loss(cfg)
    criterion_aux = None
    if cfg.LOSSES.NAME_AUX != '':
        criterion_aux = build_aux_loss(cfg)

    # initializing prototypes if using mpcbml loss
    if cfg.LOSSES.NAME == 'mpcbml_loss':
        cached_prototypes, cached_cluster_sizes, cache_path = load_cached_prototypes(cfg)

        if cached_prototypes is not None:
            logger.info(f"Loaded cached prototypes from {cache_path}, skipping recomputation.")
            prototypes = cached_prototypes
            cluster_sizes = cached_cluster_sizes
        else:
            logger.info(f"No cache found. Initializing prototypes using {cfg.LOSSES.MPCBML_LOSS.INIT_METHOD}...")

            if cfg.LOSSES.MPCBML_LOSS.INIT_METHOD == 'kmeans':
                prototypes, cluster_sizes = initialize_prototypes_kmeans(model=model, cfg=cfg)
            elif cfg.LOSSES.MPCBML_LOSS.INIT_METHOD == 'mean':
                prototypes = initialize_prototypes_mean(model=model, cfg=cfg)
                cluster_sizes = None
            elif cfg.LOSSES.MPCBML_LOSS.INIT_METHOD == 'random':
                prototypes = initialize_prototypes_random(
                    num_classes=cfg.LOSSES.MPCBML_LOSS.N_CLASSES,
                    prototype_per_class=cfg.LOSSES.MPCBML_LOSS.PROTOTYPE_PER_CLASS,
                    embed_dim=cfg.MODEL.HEAD.DIM,
                    device=device
                )
                cluster_sizes = None
            else:
                raise ValueError(f"Unknown initializing method: {cfg.LOSSES.MPCBML_LOSS.INIT_METHOD}")

            _, key_dict = _get_prototype_cache_path(cfg)
            save_prototype_cache(cache_path, key_dict, prototypes, cluster_sizes)
            logger.info(f"Saved prototype cache to {cache_path}")

        criterion.set_prototypes_and_weights(prototypes, cluster_sizes)
        del prototypes
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Prototype initialization complete.")

    loss_param = None
    if cfg.LOSSES.NAME in ['softtriple_loss', 'proxynca_loss', 'center_loss', 'adv_loss']:
        loss_param = criterion
    if cfg.LOSSES.NAME_AUX in ['softtriple_loss', 'proxynca_loss', 'center_loss', 'adv_loss']:
        loss_param = criterion_aux

    optimizer_main, optimizer_loss = build_optimizer(
        cfg,
        model,
        criterion=criterion,
        loss_param=loss_param,
    )

    scheduler_main, scheduler_loss = build_lr_scheduler(
        cfg,
        optimizer_main,
        optimizer_loss,
    )

    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)
    eval_train_loader = build_data(cfg, is_train=True, is_eval=True)

    logger.info(train_loader.dataset)
    logger.info(val_loader.dataset)

    arguments = dict()
    arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    checkpointer = Checkpointer(model, optimizer_main, scheduler_main, cfg.SAVE_DIR)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        eval_train_loader,
        optimizer_main,
        optimizer_loss,
        scheduler_main,
        scheduler_loss,
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
  Parse input arguments
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)

    if cfg.LOSSES.NAME == 'mpcbml_loss':
        with open(cfg.DATA.CLASS_COUNT_SOURCE, 'r') as fp:
            class_counts = json.load(fp)
        total = sum(class_counts.values())
        # assuming classes are stored as string keys "0", "1", ..., ensure correct order:
        priors = [class_counts.get(str(i), 0) / total for i in range(cfg.LOSSES.MPCBML_LOSS.N_CLASSES)]
        cfg.LOSSES.MPCBML_LOSS.CLASS_PRIORS = priors

    if args.train_test == 'train':
        train(cfg)
    else:
        test(cfg)
