import datetime
import time

import torch.nn.functional as F
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

            if hasattr(criterion, 'global_ma_pos'):
                with torch.no_grad():
                    xi = getattr(criterion, 'latest_xi', None)
                    current_neg = getattr(criterion, 'latest_current_neg_mean', None)

                    reg_term = 0.0
                    if xi is not None and current_neg is not None:
                        reg_term = (criterion.reg_weight * F.relu(xi - current_neg)).item()
                        # reg_term = (criterion.reg_weight * torch.pow(xi - current_neg, 2)).item()

                xi_str = f"{xi:.4f}" if xi is not None else "N/A"
                cneg_str = f"{current_neg:.4f}" if current_neg is not None else "N/A"

                logger.info(
                    f"Reg Stats | "
                    f"global_ma_pos: {criterion.global_ma_pos:.4f} | "
                    f"global_ma_neg: {criterion.global_ma_neg:.4f} | "
                    f"xi (threshold): {xi_str} | "
                    f"current_neg_mean: {cneg_str} | "
                    f"reg_loss: {reg_term:.4f}"
                )

            if iteration in [0, 800, 1600, 2400, 3200, 4000, 4800]:
                with torch.no_grad():
                    # shape: [C, K] -> each entry is the L2 norm of that prototype
                    proto_norms_per_dim = criterion.prototypes.norm(p=2, dim=2)
                    np.savetxt(
                        f'proto_norms_iter_{iteration:06d}.txt',
                        proto_norms_per_dim.cpu().numpy(),
                        fmt='%.6f',
                        delimiter='\t',
                        header=f'Prototype L2 norms at iteration {iteration}'
                    )

                    # shape: [C, K] -> each entry is the weight's value corresponding to that prototype
                    weight_value_per_dim = criterion.weights
                    np.savetxt(
                        f'weight_value_iter_{iteration:06d}.txt',
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
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:06d}".format(iteration))

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
        plt.savefig(f'recall_at_{k}_iter_{iteration}.png')
        plt.close()  # Close to free memory

    # Positive & Negative prototype usage heatmap
    pos_proto_usage = criterion.pos_proto_counts.cpu().numpy() # [C, K]
    neg_proto_usage = criterion.neg_proto_counts.cpu().numpy() # [C, K]

    plt.figure(figsize=(10, 8))
    sns.heatmap(pos_proto_usage, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Selection count'})
    plt.xlabel('Positive Prototype index')
    plt.ylabel('Class index')
    plt.title('Positive Prototype selection heatmap')
    plt.savefig('positive_prototype_selection.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(neg_proto_usage, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Selection count'})
    plt.xlabel('Negative Prototype index')
    plt.ylabel('Class index')
    plt.title('Negative Prototype selection heatmap')
    plt.savefig('negative_prototype_selection.png', dpi=150)
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