import os
import json
import datetime
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from cbml_benchmark.data.evaluations import RetMetric
from cbml_benchmark.utils.feat_extractor import feat_extractor
from cbml_benchmark.utils.freeze_bn import set_bn_eval
from cbml_benchmark.utils.metric_logger import MetricLogger
from cbml_benchmark.utils.proto_stats import ProtoStatsLogger
from cbml_benchmark.utils.prototype_initializer import reinitialize_prototypes
from cbml_benchmark.utils.gen_diagnostics import GenDiagLogger, plot_gen_diag


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _centered_recalls(f, lab, mean):
    """R@1,2,4,8 after subtracting `mean` from L2-normalised features and re-normalising."""
    fc = _unit(f - mean).astype(np.float32)
    rm = RetMetric(feats=fc, labels=lab)
    return [rm.recall_k(k) for k in (1, 2, 4, 8)]


def do_train(
        cfg,               # Configuration object with training settings.
        model,             # Neural network model to train.
        train_loader,      # DataLoader for training data.
        val_loader,        # DataLoader for validation data.
        eval_train_loader,
        optimizer,         # Optimizer for updating model parameters.
        scheduler,         # Learning rate scheduler.
        criterion,         # Primary loss function.
        criterion_aux,     # Auxiliary loss function (if any).
        checkpointer,      # Object to save and load model checkpoints.
        device,            # Device for computation (e.g., "cuda" or "cpu").
        checkpoint_period, # Frequency (in iterations) to save checkpoints.
        arguments,         # Dictionary for tracking state (e.g., current iteration).
        logger             # Logger for printing training progress and metrics.
):
    """
    Main training loop.
    """
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")  # For tracking and logging training metrics.
    max_iter = len(train_loader)  # Total number of iterations.

    use_proxy = cfg.LOSSES.NAME == 'cbml_loss'
    if use_proxy:
        proto_logger = ProtoStatsLogger(criterion, out_dir=os.path.join(cfg.SAVE_DIR, 'proto_stats'),
                                        flush_every=20, reset_counts_every=1000)

    # ---- generalization diagnostics (V1, V2, V3, d', geometry); see gen_diagnostics.py ----
    gen_logger = None
    if 'DIAG' in cfg and cfg.DIAG.ENABLE:
        gen_logger = GenDiagLogger(
            out_dir=os.path.join(cfg.SAVE_DIR, 'gen_diag'),
            per_class=cfg.DIAG.PER_CLASS,
            seed=cfg.DATA.EVAL_TRAIN_SUBSAMPLE_SEED,   # same images in every eval and every run
            ramp_gamma=cfg.DIAG.RAMP_GAMMA,
            xi_gamma=cfg.DIAG.XI_GAMMA)
    else:
        logger.warning("cfg.DIAG missing or disabled: generalization diagnostics are OFF "
                       "(add the DIAG block to defaults.py).")

    # Initialize tracking variables for best model and time.
    start_iter = arguments["iteration"]
    state = {"best_iteration": -1, "best_recall": 0.0}

    # Store recalls for plotting / results.json
    train_recalls_over_iters = []
    val_recalls_over_iters = []
    val_centered_tm = []     # test feats centred with the TRAIN mean (deployable)
    val_centered_own = []    # test feats centred with their own mean (transductive reference)
    train_centered = []      # train feats centred with the train mean
    iters = []

    # Start timers for training.
    start_training_time = time.time()
    end = time.time()

    # ------------------------------------------------------------------------
    # Evaluation (val = test classes, train = train-eval subsample, diagnostics)
    # ------------------------------------------------------------------------
    def evaluate(it):
        model.eval()  # Set model to evaluation mode.
        logger.info('Validation')

        # Extract labels and features for validation set.
        labels = val_loader.dataset.label_list
        labels = np.array([int(k) for k in labels])
        feats = feat_extractor(model, val_loader, logger=logger)  # Feature extraction.

        # Compute retrieval metrics (e.g., recall at K).
        ret_metric = RetMetric(feats=feats, labels=labels)
        recall_curr = [ret_metric.recall_k(k) for k in (1, 2, 4, 8)]
        logger.info(f'Val Recalls: {recall_curr}')

        # Update best (recall@1 on the TEST classes -- kept for comparability only).
        if recall_curr[0] > state["best_recall"]:
            state["best_recall"] = recall_curr[0]
            state["best_iteration"] = it
            logger.info(f'Best iteration {it}: recall@1: {recall_curr[0]:.3f}')
        else:
            logger.info(f'Recall@1 at iteration {it:06d}: recall@1: {recall_curr[0]:.3f}')

        # Recalls on the training classes
        train_eval_labels = eval_train_loader.dataset.label_list
        train_eval_labels = np.array([int(k) for k in train_eval_labels])
        train_eval_feats = feat_extractor(model, eval_train_loader, logger=logger)

        ret_metric_train_eval = RetMetric(feats=train_eval_feats, labels=train_eval_labels)
        recall_curr_train_eval = [ret_metric_train_eval.recall_k(k) for k in (1, 2, 4, 8)]
        logger.info(f'Train Recalls: {recall_curr_train_eval}')

        # V1/V2/V3, d', Lemma-1 geometry on train classes vs. test classes
        if gen_logger is not None:
            gen_logger.log_pair(it, train_eval_feats, train_eval_labels, feats, labels, logger)

        # retrieval after removing the common direction (mean embedding): cheap check of
        # whether the shared component of the embedding hurts retrieval
        try:
            ft, fv = _unit(_np(train_eval_feats)), _unit(_np(feats))
            m_tr = ft.mean(0, keepdims=True)
            rc_tm = _centered_recalls(fv, labels, m_tr)
            rc_own = _centered_recalls(fv, labels, fv.mean(0, keepdims=True))
            rc_tr = _centered_recalls(ft, train_eval_labels, m_tr)
            logger.info(f'Centred Val Recalls (train mean): {rc_tm} | (own mean): {rc_own} | '
                        f'Centred Train Recalls: {rc_tr}')
        except Exception as e:                      # noqa
            logger.warning(f"[centred retrieval] skipped at iteration {it}: {e!r}")
            rc_tm = rc_own = rc_tr = None

        iters.append(it)
        train_recalls_over_iters.append(recall_curr_train_eval)
        val_recalls_over_iters.append(recall_curr)
        val_centered_tm.append(rc_tm)
        val_centered_own.append(rc_own)
        train_centered.append(rc_tr)

    for iteration, (images, targets) in enumerate(train_loader, start_iter):
        # ====================================================================
        # VALIDATION (before the update of this iteration, i.e. after `iteration` updates)
        # ====================================================================
        if iteration % cfg.VALIDATION.VERBOSE == 0 or iteration == max_iter:
            evaluate(iteration)

        if use_proxy and cfg.SOLVER.PROTO_REINIT_ITER > 0 and iteration == cfg.SOLVER.PROTO_REINIT_ITER:
            logger.info(f"Re-initializing prototypes at iteration {iteration}")
            # k-means seed follows the run seed
            reinitialize_prototypes(model, criterion, optimizer, cfg, seed=int(cfg.SOLVER.RNG_SEED))
            proto_logger.set_reference()

        # Switch back to training mode.
        model.train()
        model.apply(set_bn_eval)  # Freeze BatchNorm layers during training.

        # Measure data loading time.
        data_time = time.time() - end
        iteration = iteration + 1  # Increment iteration counter.
        arguments["iteration"] = iteration

        # Update learning rate scheduler.
        # (kept before optimizer.step() on purpose: moving it would shift the LR
        #  schedule by one step and make new runs incomparable with the old ones)
        scheduler.step()

        # Move data to the specified device.
        images = images.to(device)
        targets = torch.stack([target.to(device) for target in targets])

        # Forward pass through the model to get features.
        feats = model(images)
        if criterion_aux is not None:
            # Use auxiliary loss if provided.
            if cfg.LOSSES.NAME_AUX != 'adv_loss':
                loss = criterion(feats, targets)  # Primary loss.
                loss_aux = criterion_aux(feats, targets)  # Auxiliary loss.
                # Combine primary and auxiliary losses with a weight.
                loss = (1 - cfg.LOSSES.AUX_WEIGHT) * loss + cfg.LOSSES.AUX_WEIGHT * loss_aux
            else:
                # Special handling for adversarial loss.
                loss = criterion(feats, targets)
                feats = torch.split(feats, cfg.LOSSES.ADV_LOSS.CLASS_DIM, dim=1)
                loss_aux = criterion_aux(feats[0], feats[1])
                loss = (1 - cfg.LOSSES.AUX_WEIGHT) * loss + cfg.LOSSES.AUX_WEIGHT * loss_aux
        else:
            # Only use primary loss if no auxiliary loss is provided.
            loss = criterion(feats, targets)

        if use_proxy:
            proto_logger.update(feats, targets, step=iteration - 1)   # iteration was already incremented

        # Backward pass and optimization.
        optimizer.zero_grad()  # Clear previous gradients.
        loss.backward()        # Compute gradients.
        optimizer.step()       # Update model parameters.

        # Measure batch processing time.
        batch_time = time.time() - end
        end = time.time()

        # Update metrics and log.
        meters.update(time=batch_time, data=data_time, loss=loss.item())
        if use_proxy and getattr(criterion, "last_mvc", None) is not None:
            meters.update(mvc=criterion.last_mvc.item())
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # Log training progress every 20 iterations or at the end.
        if iteration % 20 == 0 or iteration == max_iter:
            mem_gb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0 if torch.cuda.is_available() else 0.0
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr}",
                        "max mem: {memory:.1f} GB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=", ".join(f"{v:.2e}" for v in sorted({g['lr'] for g in optimizer.param_groups})),
                    memory=mem_gb,
                )
            )

        # Save model checkpoint periodically.
        # if iteration % checkpoint_period == 0:
        #     checkpointer.save("model_{:06d}".format(iteration))

    # ====================================================================
    # FINAL EVALUATION: the loop above only evaluates BEFORE an update, so the
    # model after the very last update was never evaluated.  `iteration` now
    # equals the number of completed updates.
    # ====================================================================
    if not iters or iters[-1] != iteration:
        evaluate(iteration)

    if use_proxy:
        proto_logger.flush(iteration)

    # ====================================================================
    # POST-TRAINING: PLOTTING
    # ====================================================================
    for i, k in enumerate([1, 2, 4, 8]):
        plt.figure()
        plt.plot(iters, [r[i] for r in train_recalls_over_iters], label=f'Train R@{k}')
        plt.plot(iters, [r[i] for r in val_recalls_over_iters], label=f'Val R@{k}')
        plt.xlabel('Iteration')
        plt.ylabel(f'Recall@{k}')
        plt.legend()
        plt.title(f'Recall@K over Iterations (k={k})')
        plt.savefig(os.path.join(cfg.SAVE_DIR, f'recall_at_{k}_iter_{iteration}.png'))
        plt.close()  # Close to free memory

    if gen_logger is not None:
        try:
            plot_gen_diag(gen_logger.path, os.path.join(cfg.SAVE_DIR, 'gen_diag', 'gen_diag.png'))
        except Exception as e:  # diagnostics must never break the run
            logger.warning(f"[gen-diag] plotting failed: {e!r}")

    # Positive & Negative prototype usage heatmap (prototype losses only)
    if use_proxy and hasattr(criterion, "pos_proto_counts"):
        pos_proto_usage = criterion.pos_proto_counts.cpu().numpy()  # [C, K]
        neg_proto_usage = criterion.neg_proto_counts.cpu().numpy()  # [C, K]

        plt.figure(figsize=(10, 8))
        sns.heatmap(pos_proto_usage, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Selection count'})
        plt.xlabel('Positive Prototype index')
        plt.ylabel('Class index')
        plt.title('Positive Prototype selection heatmap')
        plt.savefig(os.path.join(cfg.SAVE_DIR, 'positive_prototype_selection.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(neg_proto_usage, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Selection count'})
        plt.xlabel('Negative Prototype index')
        plt.ylabel('Class index')
        plt.title('Negative Prototype selection heatmap')
        plt.savefig(os.path.join(cfg.SAVE_DIR, 'negative_prototype_selection.png'), dpi=150)
        plt.close()

    # Log total training time.
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    # Log the best iteration and recall achieved.
    logger.info(f"Best iteration: {state['best_iteration']:06d} | best recall {state['best_recall']} ")

    # ====================================================================
    # results.json: everything needed to aggregate several seeds later.
    # plateau = mean over all evaluations AFTER the last LR drop (iteration > STEPS[-1]);
    # it includes the final evaluation at the last iteration.
    # ====================================================================
    steps = list(cfg.SOLVER.STEPS)
    plateau_start = steps[-1] if steps else int(0.75 * max_iter)
    idx = [i for i, it in enumerate(iters) if it > plateau_start]
    plateau_val = np.mean([val_recalls_over_iters[i] for i in idx], axis=0).tolist() if idx else None
    plateau_train = np.mean([train_recalls_over_iters[i] for i in idx], axis=0).tolist() if idx else None
    def _plateau(lst):
        rows = [lst[i] for i in idx if lst[i] is not None]
        return np.mean(rows, axis=0).tolist() if rows else None

    results = {
        "seed": int(cfg.SOLVER.RNG_SEED),
        "save_dir": cfg.SAVE_DIR,
        "iters": iters,
        "val_recalls": val_recalls_over_iters,        # [n_eval, 4]  R@1,2,4,8
        "train_recalls": train_recalls_over_iters,    # [n_eval, 4]
        "final_iteration": int(iteration),
        "final_val": val_recalls_over_iters[-1],
        "final_train": train_recalls_over_iters[-1],
        "plateau_start": plateau_start,
        "plateau_val": plateau_val,
        "plateau_train": plateau_train,
        "val_centered_trainmean": val_centered_tm,
        "val_centered_own": val_centered_own,
        "train_centered": train_centered,
        "plateau_val_centered_trainmean": _plateau(val_centered_tm),
        "plateau_val_centered_own": _plateau(val_centered_own),
        "plateau_train_centered": _plateau(train_centered),
        "best_iteration_test_selected": state["best_iteration"],
        "best_recall_test_selected": state["best_recall"],
        "training_time_s": total_training_time,
        "config": cfg.dump(),
    }
    with open(os.path.join(cfg.SAVE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)
    logger.info(f"Final val R@1..8 (iteration {iteration}): {results['final_val']} | "
                f"plateau (> {plateau_start}) val: {plateau_val}")


def do_test(
        model,        # Neural network model to evaluate.
        val_loader,   # DataLoader for validation/test data.
        logger        # Logger for printing test progress and metrics.
):
    """
    Evaluate the model on the validation/test set.
    """
    logger.info("Start testing")
    model.eval()  # Set model to evaluation mode.
    logger.info('test')

    # Extract labels and features for the test set.
    labels = val_loader.dataset.label_list
    labels = np.array([int(k) for k in labels])
    feats = feat_extractor(model, val_loader, logger=logger)  # Feature extraction.

    # Compute retrieval metrics (e.g., recall at K).
    ret_metric = RetMetric(feats=feats, labels=labels)
    recall_curr = []
    recall_curr.append(ret_metric.recall_k(1))
    recall_curr.append(ret_metric.recall_k(2))
    recall_curr.append(ret_metric.recall_k(4))
    recall_curr.append(ret_metric.recall_k(8))

    # Log recall metrics.
    print(recall_curr)
