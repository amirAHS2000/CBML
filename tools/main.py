import gc
import json
import argparse
import torch
import torch.nn.functional as F

from cbml_benchmark.config import cfg
from cbml_benchmark.data import build_data
from cbml_benchmark.engine.trainer import do_train, do_test
from cbml_benchmark.losses import build_loss,build_aux_loss
from cbml_benchmark.modeling import build_model
from cbml_benchmark.solver import build_lr_scheduler, build_optimizer
from cbml_benchmark.utils.logger import setup_logger
from cbml_benchmark.utils.checkpoint import Checkpointer

from cbml_benchmark.utils.prototype_initializer import initialize_prototypes_random, initialize_prototypes_mean, initialize_prototypes_kmeans


def train(cfg):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    criterion = build_loss(cfg)
    criterion_aux = None
    if cfg.LOSSES.NAME_AUX != '':
        criterion_aux = build_aux_loss(cfg)

    # initializing prototypes if using multi_prototype_cbml loss
    if cfg.LOSSES.NAME == 'multi_prototype_cbml':
        logger.info(f"Initializing prototypes using {cfg.LOSSES.MULTI_PROTOTYPE_CBML.INIT_METHOD}...")

        if cfg.LOSSES.MULTI_PROTOTYPE_CBML.INIT_METHOD == 'kmeans':
            prototypes = initialize_prototypes_kmeans(
                model=model,
                cfg=cfg
            )
        elif cfg.LOSSES.MULTI_PROTOTYPE_CBML.INIT_METHOD == 'mean':
            prototypes = initialize_prototypes_mean(
                model=model,
                cfg=cfg
            )
        elif cfg.LOSSES.MULTI_PROTOTYPE_CBML.INIT_METHOD == 'random':
            prototypes = initialize_prototypes_random(
                num_classes=cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES,
                prototype_per_class=cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
                embed_dim=cfg.MODEL.HEAD.DIM,
                device=device
            )
        else:
            raise ValueError(f"Unknown initializing method: {cfg.LOSSES.MULTI_PROTOTYPE_CBML.INIT_METHOD}")

        # normalize the prototypes
        prototypes = F.normalize(prototypes, p=2, dim=2)
        criterion.set_prototypes(prototypes)
        # clear the orginal prototypes tensor
        del prototypes
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Prototype initialization complete.")

    loss_param = None
    if cfg.LOSSES.NAME == 'softtriple_loss' or cfg.LOSSES.NAME == 'proxynca_loss' or cfg.LOSSES.NAME == 'center_loss' or cfg.LOSSES.NAME == 'adv_loss' or cfg.LOSSES.NAME == 'multi_prototype_cbml':
        loss_param = criterion
    if cfg.LOSSES.NAME_AUX == 'softtriple_loss' or cfg.LOSSES.NAME_AUX == 'proxynca_loss' or cfg.LOSSES.NAME_AUX == 'center_loss' or cfg.LOSSES.NAME_AUX == 'adv_loss':
        loss_param = criterion_aux

    optimizer = build_optimizer(cfg, model, loss_param=loss_param)
    scheduler = build_lr_scheduler(cfg, optimizer)

    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)

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
        optimizer,
        scheduler,
        criterion,
        criterion_aux,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        logger
        # dual_optimizer=dual_optimizer
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

    if cfg.LOSSES.NAME == 'multi_prototype_cbml':
        with open(cfg.DATA.CLASS_COUNT_SOURCE, 'r') as fp:
            class_counts = json.load(fp)
        total = sum(class_counts.values())
        # assuming classes are stored as string keys "0", "1", ..., ensure correct order:
        priors = [class_counts.get(str(i), 0) / total for i in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES)]
        cfg.LOSSES.MULTI_PROTOTYPE_CBML.CLASS_PRIORS = priors

    if args.train_test == 'train':
        train(cfg)
    else:
        test(cfg)
