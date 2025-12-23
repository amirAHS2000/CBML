import torch
from .lr_scheduler import WarmupMultiStepLR


def build_optimizer(cfg, model, criterion=None, loss_param=None):
    """
    Returns:
        optimizer_main: Adam (or cfg.SOLVER.OPTIMIZER_NAME) for model + most loss params
        optimizer_weights: SGD for MP-CBML weights, or None otherwise
    """
    adam_params = []

    # Model params with lr multiplier
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr_mul = 0.1 if "backbone" in key else 1.0
        adam_params.append({"params": [value], "lr_mul": lr_mul})

    is_mpcbml = (cfg.LOSSES.NAME == 'mpcbml_loss')

    if is_mpcbml and criterion is not None:
        # MP-CBML: add all loss params except weights to Adam
        for name, param in criterion.named_parameters():
            if not param.requires_grad:
                continue
            if name == 'weights':
                continue
            adam_params.append({"params": [param], "lr_mul": 1.0})
    elif loss_param is not None:
        # Other losses: add all loss parameters to Adam
        for p in loss_param.parameters():
            adam_params.append({"params": [p], "lr_mul": 1.0})

    # Build main optimizer with lr multipliers
    optimizer_main = _build_optimizer_with_lr_mul(
        adam_params,
        cfg.SOLVER.BASE_LR,
        cfg.SOLVER.WEIGHT_DECAY,
        optimizer_name=cfg.SOLVER.OPTIMIZER_NAME
    )

    optimizer_weights = None
    if is_mpcbml and criterion is not None and hasattr(criterion, 'weights'):
        optimizer_weights = torch.optim.SGD(
            [{"params": [criterion.weights]}],
            lr=getattr(cfg.SOLVER, 'WEIGHT_LR', 1e-4),
            momentum=getattr(cfg.SOLVER, 'WEIGHT_MOMENTUM', 0.0),
            weight_decay=0.0
        )

    return optimizer_main, optimizer_weights

def _build_optimizer_with_lr_mul(params, base_lr, weight_decay, optimizer_name='Adam'):
    param_groups = []
    for group in params:
        lr = base_lr * group.get("lr_mul", 1.0)
        param_groups.append({
            "params": group["params"],
            "lr": lr,
            "weight_decay": weight_decay,
        })
    optimizer_class = getattr(torch.optim, optimizer_name)
    return optimizer_class(param_groups)

def build_lr_scheduler(cfg, optimizer_main, optimizer_weights=None):
    scheduler_main = WarmupMultiStepLR(
        optimizer_main,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )

    scheduler_weights = None
    if optimizer_weights is not None:
        scheduler_weights = WarmupMultiStepLR(
            optimizer_weights,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )

    return scheduler_main, scheduler_weights
