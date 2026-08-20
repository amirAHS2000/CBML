import os
import datetime
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from cbml_benchmark.data.evaluations import RetMetric
from cbml_benchmark.data.evaluations import RetMetricGPU
from cbml_benchmark.utils.feat_extractor import feat_extractor
from cbml_benchmark.utils.freeze_bn import set_bn_eval
from cbml_benchmark.utils.metric_logger import MetricLogger


def do_train(
        cfg,               # Configuration object with training settings.
        model,             # Neural network model to train.
        train_loader,      # DataLoader for training data.
        val_loader,        # DataLoader for validation data.
        eval_train_loader, # DataLoader for training data (without any augmentations or transformations just for recalls calculation)
        optimizer_main,    # Optimizer for updating model parameters.
        optimizer_loss,
        scheduler_main,    # Learning rate scheduler.
        scheduler_loss,
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

    # Initialize tracking variables for best model and time.
    start_iter = arguments["iteration"]
    best_iteration = -1
    best_recall = 0

    # store recalls for plotting
    train_recalls_over_iters = []
    val_recalls_over_iters = []
    iters = []

    # Regularization statistics collected at validation points. These values
    # describe the latest training batch seen by MpcbmlLoss before validation.
    reg_stats_history = {
        "xi": [],
        "ema_pos": [],
        "ema_neg": [],
        "current_pos_mean": [],
        "current_neg_mean": [],
        "neg_min": [],
        "neg_p10": [],
        "violation_rate": [],
        "mean_violation": [],
        "reg_loss_raw": [],
        "reg_loss_weighted": [],
    }
    
    # Start timers for training.
    start_training_time = time.time()
    end = time.time()

    for iteration, (images, targets) in enumerate(train_loader, start_iter):
        # ====================================================================
        # VALIDATION
        # ====================================================================
        # Perform validation periodically or at the end of training.
        if iteration % cfg.VALIDATION.VERBOSE == 0 or iteration == max_iter:
            model.eval()  # Set model to evaluation mode.
            logger.info('Validation')

            # Extract labels and features for validation set.
            labels = val_loader.dataset.label_list
            labels = np.array([int(k) for k in labels])
            feats = feat_extractor(model, val_loader, logger=logger, return_numpy=False)

            # Compute retrieval metrics (e.g., recall at K).
            ret_metric = RetMetricGPU(feats=feats, labels=labels, device=device)
            recall_curr = ret_metric.recall_at_ks(ks=(1, 2, 4, 8))
            recall_curr = [recall_curr[1], recall_curr[2], recall_curr[4], recall_curr[8]]

            # Log current recall metrics.
            logger.info(f'Val Recalls: {recall_curr}')

            # Update best model if recall@1 improves.
            if recall_curr[0] > best_recall:
                best_recall = recall_curr[0]
                best_iteration = iteration
                logger.info(f'Best iteration {iteration}: recall@1: {recall_curr[0]:.3f}')
                checkpointer.save(f"best_model")
            else:
                logger.info(f'Recall@1 at iteration {iteration:06d}: recall@1: {recall_curr[0]:.3f}')

            # compute recalls for training set (entire training set)
            train_eval_labels = eval_train_loader.dataset.label_list
            train_eval_labels = np.array([int(k) for k in train_eval_labels])
            train_eval_feats = feat_extractor(model, eval_train_loader, logger=logger, return_numpy=False)

            # compute retrieval metrics (e.g., recall at k) on training set
            ret_metric_train_eval = RetMetricGPU(feats=train_eval_feats, labels=train_eval_labels, device=device)
            recall_curr_train_eval_dict = ret_metric_train_eval.recall_at_ks(ks=(1, 2, 4, 8))
            recall_curr_train_eval = [recall_curr_train_eval_dict[k] for k in [1, 2, 4, 8]]
            
            logger.info(f'Train Recalls: {recall_curr_train_eval}')

            # store for plotting
            iters.append(iteration)
            train_recalls_over_iters.append(recall_curr_train_eval)
            val_recalls_over_iters.append(recall_curr)

            if hasattr(criterion, 'latest_xi'):
                def _scalar(value):
                    if value is None:
                        return None
                    if torch.is_tensor(value):
                        return float(value.detach().cpu().item())
                    return float(value)

                def _fmt(value, digits=4):
                    value = _scalar(value)
                    return f"{value:.{digits}f}" if value is not None else "N/A"

                # These are cached by MpcbmlLoss during the latest training
                # batch. In particular, latest_reg_loss is the actual
                # sample-wise squared-hinge term, not a reconstruction from
                # the old batch-mean formulation.
                xi = getattr(criterion, 'latest_xi', None)
                ema_pos = getattr(criterion, 'latest_ema_pos', None)
                ema_neg = getattr(criterion, 'latest_ema_neg', None)
                current_pos = getattr(criterion, 'latest_current_pos_mean', None)
                current_neg = getattr(criterion, 'latest_current_neg_mean', None)
                reg_loss = getattr(criterion, 'latest_reg_loss', None)
                weighted_reg_loss = getattr(criterion, 'latest_weighted_reg_loss', None)
                violation_rate = getattr(criterion, 'latest_neg_violation_rate', None)
                mean_violation = getattr(criterion, 'latest_mean_neg_violation', None)
                neg_min = getattr(criterion, 'latest_neg_min', None)
                neg_p10 = getattr(criterion, 'latest_neg_p10', None)

                logger.info(
                    f"Reg Stats (latest train batch) | "
                    f"EMA_pos: {_fmt(ema_pos)} | "
                    f"EMA_neg: {_fmt(ema_neg)} | "
                    f"xi: {_fmt(xi)} | "
                    f"current_pos_mean: {_fmt(current_pos)} | "
                    f"current_neg_mean: {_fmt(current_neg)} | "
                    f"neg_min: {_fmt(neg_min)} | "
                    f"neg_p10: {_fmt(neg_p10)} | "
                    f"violation_rate: {_fmt(_scalar(violation_rate) * 100 if violation_rate is not None else None, 2)}% | "
                    f"mean_violation: {_fmt(mean_violation)} | "
                    f"reg_loss_raw: {_fmt(reg_loss)} | "
                    f"reg_loss_weighted: {_fmt(weighted_reg_loss)}"
                )

                # Keep a history for post-training plots and a tab-separated
                # record that can be analyzed independently of the logger.
                reg_values = {
                    "xi": _scalar(xi),
                    "ema_pos": _scalar(ema_pos),
                    "ema_neg": _scalar(ema_neg),
                    "current_pos_mean": _scalar(current_pos),
                    "current_neg_mean": _scalar(current_neg),
                    "neg_min": _scalar(neg_min),
                    "neg_p10": _scalar(neg_p10),
                    "violation_rate": _scalar(violation_rate),
                    "mean_violation": _scalar(mean_violation),
                    "reg_loss_raw": _scalar(reg_loss),
                    "reg_loss_weighted": _scalar(weighted_reg_loss),
                }
                for key, value in reg_values.items():
                    reg_stats_history[key].append(value)

            if iteration in [0, 800, 1600, 2400, 3200, 4000, 4800]:
                with torch.no_grad():
                    # shape: [C, K] -> each entry is the L2 norm of that prototype
                    proto_norms_per_dim = criterion.prototypes.norm(p=2, dim=2)
                    np.savetxt(
                        os.path.join(cfg.SAVE_DIR, f'proto_norms_iter_{iteration:06d}.txt'),
                        proto_norms_per_dim.cpu().numpy(),
                        fmt='%.6f',
                        delimiter='\t',
                        header=f'Prototype L2 norms at iteration {iteration}'
                    )

                    # shape: [C, K] -> each entry is the weight's value corresponding to that prototype
                    weight_value_per_dim = criterion.weights
                    np.savetxt(
                        os.path.join(cfg.SAVE_DIR, f'weight_value_iter_{iteration:06d}.txt'),
                        weight_value_per_dim.cpu().numpy(),
                        fmt='%.6f',
                        delimiter='\t',
                        header=f"Weight's value at iteration {iteration}"
                    )
                    
                    logger.info(f"Saved prototype/weight snapshots at iteration {iteration}")

        # ====================================================================
        # TRAINING STEP
        # ====================================================================
        # Switch back to training mode.
        model.train()
        model.apply(set_bn_eval)  # Freeze BatchNorm layers during training.

        # Measure data loading time.
        data_time = time.time() - end
        iteration = iteration + 1  # Increment iteration counter.
        arguments["iteration"] = iteration

        # Update learning rate scheduler.
        scheduler_main.step()
        if scheduler_loss is not None:
            scheduler_loss.step()

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

        # Backward pass and optimization.
        optimizer_main.zero_grad()
        if optimizer_loss is not None:
            optimizer_loss.zero_grad()

        loss.backward()

        # If we have a separate weights optimizer (MP-CBML case)
        if optimizer_loss is not None and cfg.LOSSES.NAME == 'mpcbml_loss':
            # apply constrained gradient update before SGD step
            if hasattr(criterion, 'constrained_weight_update'):
                criterion.constrained_weight_update()
            optimizer_main.step()
            optimizer_loss.step()
        else:
            # single optimizer case
            optimizer_main.step()

        # Measure batch processing time.
        batch_time = time.time() - end
        end = time.time()

        # Update metrics and log.
        meters.update(time=batch_time, data=data_time, loss=loss.item())
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # Log training progress every 20 iterations or at the end.
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.1f} GB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_main.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )

        # Save model checkpoint periodically.
        # if iteration % checkpoint_period == 0:
        #     checkpointer.save("model_{:06d}".format(iteration))

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

    # Regularization diagnostics
    if reg_stats_history["xi"]:
        reg_path = os.path.join(cfg.SAVE_DIR, f'regularization_stats_iter_{iteration}.tsv')
        header = "iteration\t" + "\t".join(reg_stats_history.keys())
        rows = []
        for idx, logged_iter in enumerate(iters[-len(reg_stats_history["xi"]):]):
            rows.append([logged_iter] + [reg_stats_history[key][idx] for key in reg_stats_history])
        np.savetxt(reg_path, np.asarray(rows, dtype=float), delimiter='\t', fmt='%.8f', header=header, comments='')

        plt.figure()
        plt.plot(iters[-len(reg_stats_history["xi"]):], reg_stats_history["xi"], label='Threshold $\\xi$')
        plt.plot(iters[-len(reg_stats_history["xi"]):], reg_stats_history["current_neg_mean"], label='Mean dominant negative distance')
        plt.plot(iters[-len(reg_stats_history["xi"]):], reg_stats_history["neg_p10"], label='10th percentile negative distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.legend()
        plt.title('Negative-distance regularization diagnostics')
        plt.savefig(os.path.join(cfg.SAVE_DIR, f'reg_distance_stats_iter_{iteration}.png'))
        plt.close()

        plt.figure()
        plt.plot(iters[-len(reg_stats_history["xi"]):], [100.0 * x for x in reg_stats_history["violation_rate"]], label='Violation rate (%)')
        plt.xlabel('Iteration')
        plt.ylabel('Samples below threshold (%)')
        plt.legend()
        plt.title('Dominant negative threshold violation rate')
        plt.savefig(os.path.join(cfg.SAVE_DIR, f'reg_violation_rate_iter_{iteration}.png'))
        plt.close()

        plt.figure()
        plt.plot(iters[-len(reg_stats_history["xi"]):], reg_stats_history["reg_loss_raw"], label='Raw squared-hinge loss')
        plt.plot(iters[-len(reg_stats_history["xi"]):], reg_stats_history["reg_loss_weighted"], label='Weighted regularization contribution')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Regularization loss over iterations')
        plt.savefig(os.path.join(cfg.SAVE_DIR, f'reg_loss_stats_iter_{iteration}.png'))
        plt.close()

    # Positive & Negative prototype usage heatmap
    pos_proto_usage = criterion.pos_proto_counts.cpu().numpy() # [C, K]
    neg_proto_usage = criterion.neg_proto_counts.cpu().numpy() # [C, K]

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

    # ====================================================================    

    # Log total training time.
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    # Log the best iteration and recall achieved.
    logger.info(f"Best iteration: {best_iteration :06d} | best recall {best_recall} ")

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