import torch
from .lr_scheduler import WarmupMultiStepLR


def build_optimizer(cfg, model, criterion=None, loss_param=None):
    base_lr = getattr(cfg.SOLVER, 'BASE_LR', 0.0001)


    # ---------- ADAM for model ----------
    model_params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if key.startswith('backbone.'):
            lr_mul = 0.2
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
        elif key.startswith('headembedding.'):
            lr_mul = 1.0
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
        else:
            lr_mul = 1.0
            weight_decay = 0.0

        model_params.append({
            'params': [value],
            'lr': base_lr * lr_mul,
            'weight_decay': weight_decay
        })

    # Main optimizer (Adam) only for model
    optimizer_main = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
        model_params
    )

    # ---------- SGD for loss parameters ----------
    loss_params = []
    is_mpcbml = (cfg.LOSSES.NAME == 'mpcbml_loss')
   
    if is_mpcbml and criterion is not None:
        # Prototypes
        if hasattr(criterion, 'prototypes') and criterion.prototypes.requires_grad:
            loss_params.append({
                'params': [criterion.prototypes],
                'lr': base_lr * 1000.0,         # as before, high LR
                'momentum': 0.9,              # pure SGD, no momentum
                'weight_decay': 0.0
            })
        # Theta
        if hasattr(criterion, 'theta') and criterion.theta.requires_grad:
            loss_params.append({
                'params': [criterion.theta],
                'lr': base_lr * 3.0,
                'momentum': 0.0,
                'weight_decay': 0.0
            })
        # Weights (keep separate LR and momentum as per config)
        if hasattr(criterion, 'weights') and criterion.weights.requires_grad:
            weight_lr = getattr(cfg.SOLVER, 'WEIGHT_LR', 0.00003)
            weight_momentum = getattr(cfg.SOLVER, 'WEIGHT_MOMENTUM', 0.0)
            loss_params.append({
                'params': [criterion.weights],
                'lr': weight_lr,
                'momentum': weight_momentum,
                'weight_decay': 0.0
            })
    
    # Create a single SGD optimizer for all loss parameters
    if loss_params:
        optimizer_loss = torch.optim.SGD(loss_params)
    else:
        optimizer_loss = None

    for i, g in enumerate(optimizer_main.param_groups):
        print(
            f"group {i}: lr={g['lr']}, weight_decay={g.get('weight_decay', 'default')}"
        )
    
    for i, g in enumerate(optimizer_loss.param_groups):
        print(
            f"group {i}: lr={g['lr']}, weight_decay={g.get('weight_decay', 'default')}"
        )

    return optimizer_main, optimizer_loss

def build_lr_scheduler(cfg, optimizer_main, optimizer_loss=None):
    scheduler_main = WarmupMultiStepLR(
        optimizer_main,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
    
    scheduler_loss = None
    if optimizer_loss is not None:
        scheduler_loss = WarmupMultiStepLR(
            optimizer_loss,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    
    return scheduler_main, scheduler_loss

