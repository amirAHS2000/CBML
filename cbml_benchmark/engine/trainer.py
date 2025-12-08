import os
import datetime
import time
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from cbml_benchmark.data.evaluations import RetMetric
from cbml_benchmark.utils.feat_extractor import feat_extractor, compute_similarity_stats
from cbml_benchmark.utils.freeze_bn import set_bn_eval
from cbml_benchmark.utils.metric_logger import MetricLogger
from cbml_benchmark.utils.visualization_utils import plot_distribution_figure


def update_ema_variables(model, ema_model):
    """
    Update the Exponential Moving Average (EMA) model parameters.
    Args:
    - model: The main model being trained.
    - ema_model: The model used to store EMA of the parameters.
    """
    alpha = 0.999  # EMA smoothing coefficient.
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # Update EMA parameters: new_ema = alpha * old_ema + (1 - alpha) * current_param
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def do_train(
        cfg,               # Configuration object with training settings.
        model,             # Neural network model to train.
        train_loader,      # DataLoader for training data.
        val_loader,        # DataLoader for validation data.
        eval_train_loader, # DataLoader for training data (without any augmentationsa or transformations just for recalls calculation)
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

    # define log file
    stats_log_path = os.path.join('outputs', 'statistics_log.csv')
    os.makedirs('outputs', exist_ok=True)

    # flag to track if header has been written
    header_written = os.path.exists(stats_log_path)

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
            feats = feat_extractor(model, val_loader, logger=logger)  # Feature extraction.

            # Compute retrieval metrics (e.g., recall at K).
            ret_metric = RetMetric(feats=feats, labels=labels)
            recall_curr = []
            recall_curr.append(ret_metric.recall_k(1))
            recall_curr.append(ret_metric.recall_k(2))
            recall_curr.append(ret_metric.recall_k(4))
            recall_curr.append(ret_metric.recall_k(8))

            # Log current recall metrics.
            logger.info(f'Val Recalls: {recall_curr}')

            # ================================================================
            # MP-CBML ENHANCED STATISTICS LOGGING
            # ================================================================
            if cfg.LOSSES.NAME == 'multi_prototype_cbml':
                plot_dir = os.path.join('outputs', 'dist_plots')
                os.makedirs(plot_dir, exist_ok=True)

                logger.info('Computing Similarity Distributions...')

                # validation set distribution (generalization check)
                # val_pos, val_neg = compute_similarity_stats(model, criterion, val_loader, device)
                # plot_distribution_figure(
                #     val_pos, val_neg,
                #     title=f'Val Distribution (Iter {iteration})',
                #     save_path=os.path.join(plot_dir, f'val_dist_{iteration:06d}.png')
                # )

                # training set distribution (overfitting check)
                # use eval_train_loader (no augmentation) to get clean stats
                train_pos, train_neg = compute_similarity_stats(model, criterion, eval_train_loader, device)
                plot_distribution_figure(
                    train_pos, train_neg,
                    title=f'Train Distribution (Iter {iteration})',
                    save_path=os.path.join(plot_dir, f'train_dist_{iteration:06d}.png')
                )

                # Weight sum verification (should all be ~1.0)
                weight_sums = criterion.weights.sum(dim=1)
                logger.info(f'Weight sum check - min: {weight_sums.min():.6f}, '
                            f'max: {weight_sums.max():.6f}, '
                            f'mean: {weight_sums.mean():.6f}')
                
                # Extract prototype and weight statistics
                proto_stats = criterion.show_prototype_stats()
                weight_stats = criterion.show_weight_stats()
                
                # Core loss components
                total_loss = getattr(criterion, 'current_total_loss', 0.0) or 0.0
                mpcbml_loss = getattr(criterion, 'mpcbml_total', 0.0) or 0.0
                sim_term = getattr(criterion, 'sim_mpcbml_total', 0.0) or 0.0
                bias_term = getattr(criterion, 'bias_mpcbml_total', 0.0) or 0.0
                
                # Bias breakdown
                prior_bias = getattr(criterion, 'prior_bias_total', 0.0) or 0.0
                weight_bias = getattr(criterion, 'weight_bias_total', 0.0) or 0.0
                
                # MVC components
                mvc_loss = getattr(criterion, 'current_mvc_value', 0.0) or 0.0
                mvc_contrib = getattr(criterion, 'current_mvc_contribution', 0.0) or 0.0
                pos_mean = getattr(criterion, 'current_positive_mean', 0.0) or 0.0
                neg_mean = getattr(criterion, 'current_negative_mean', 0.0) or 0.0
                xi_val = getattr(criterion, 'current_xi', 0.0) or 0.0
                
                # Selected similarities
                pos_sim = getattr(criterion, 'current_pos_sim', 0.0) or 0.0
                neg_sim = getattr(criterion, 'current_neg_sim', 0.0) or 0.0
                sim_margin = getattr(criterion, 'current_sim_margin', 0.0) or 0.0
                
                # Beta parameter (NEW - replaces log_weight stats)
                current_beta = getattr(criterion, 'current_beta', 1.0)
                theta_value = criterion.theta.item()  # log(beta)
                
                # Weight entropy
                weight_entropy_stats = criterion.show_weight_entropy()
                mean_entropy = weight_entropy_stats['mean_entropy']
                
                # Weight sum statistics (NEW - to monitor constraint preservation)
                weight_sum_min = weight_sums.min().item()
                weight_sum_max = weight_sums.max().item()
                weight_sum_std = weight_sums.std().item()
                
                # Write header if this is the first time
                if not header_written:
                    with open(stats_log_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            # Iteration & loss components
                            'iteration', 'total_loss', 'mpcbml_loss', 'sim_term', 'bias_term',
                            'prior_bias', 'weight_bias',
                            # MVC components
                            'mvc_loss', 'mvc_contribution', 'pos_mean', 'neg_mean', 'xi',
                            # Selected similarities
                            'pos_sim', 'neg_sim', 'sim_margin',
                            # Prototype statistics
                            'mean_intra_dist', 'mean_inter_dist', 'mean_displacement',
                            # Weight statistics
                            'mean_entropy', 'mean_max_weight',
                            # Beta parameter (temperature)
                            'beta', 'theta',
                            # Weight sum verification (constraint check)
                            'weight_sum_min', 'weight_sum_max', 'weight_sum_std'
                        ])
                        header_written = True
                
                # Append data
                with open(stats_log_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        # Iteration & loss components
                        iteration,
                        round(total_loss, 8),
                        round(mpcbml_loss, 8),
                        round(sim_term, 8),
                        round(bias_term, 8),
                        round(prior_bias, 8),
                        round(weight_bias, 8),
                        # MVC components
                        round(mvc_loss, 8),
                        round(mvc_contrib, 8),
                        round(pos_mean, 8),
                        round(neg_mean, 8),
                        round(xi_val, 8),
                        # Selected similarities
                        round(pos_sim, 8),
                        round(neg_sim, 8),
                        round(sim_margin, 8),
                        # Prototype statistics
                        round(proto_stats['mean_intra_dist'], 5),
                        round(proto_stats['mean_inter_dist'], 5),
                        round(proto_stats['mean_displacement'], 5),
                        # Weight statistics
                        round(mean_entropy, 5),
                        round(weight_stats['mean_max_weight'], 5),
                        # Beta parameter
                        round(current_beta, 5),
                        round(theta_value, 5),
                        # Weight sum verification
                        round(weight_sum_min, 6),
                        round(weight_sum_max, 6),
                        round(weight_sum_std, 6)
                    ])

            elif cfg.LOSSES.NAME == 'cbml_loss':
                # Get all logged values
                mvc_val = getattr(criterion, 'current_mvc_value', 0.0) or 0.0
                pos_mean = getattr(criterion, 'current_positive_mean', 0.0) or 0.0
                neg_mean = getattr(criterion, 'current_negative_mean', 0.0) or 0.0
                xi_val = getattr(criterion, 'current_xi', 0.0) or 0.0
                
                # Loss components (NEW)
                cbml_total = getattr(criterion, 'cbml_total', 0.0) or 0.0
                pos_loss = getattr(criterion, 'pos_loss_total', 0.0) or 0.0
                neg_loss = getattr(criterion, 'neg_loss_total', 0.0) or 0.0
                mvc_contrib = getattr(criterion, 'mvc_contribution', 0.0) or 0.0

                # Write header if file doesn't exist yet
                if not os.path.exists(stats_log_path):
                    with open(stats_log_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            'iteration',
                            # Statistics
                            'mvc_value', 'pos_mean', 'neg_mean', 'xi',
                            # Loss components
                            'cbml_total', 'pos_loss', 'neg_loss', 'mvc_contrib'
                        ])

                with open(stats_log_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        round(iteration, 5),
                        # Statistics
                        round(mvc_val, 8),
                        round(pos_mean, 8),
                        round(neg_mean, 8),
                        round(xi_val, 8),
                        # Loss components
                        round(cbml_total, 8),
                        round(pos_loss, 8),
                        round(neg_loss, 8),
                        round(mvc_contrib, 8)
                    ])

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
            train_eval_feats = feat_extractor(model, eval_train_loader, logger=logger)

            # compute retrieval metrics (e.g., recall at k) on training set
            ret_metric_train_eval = RetMetric(feats=train_eval_feats, labels=train_eval_labels)
            recall_curr_train_eval = [ret_metric_train_eval.recall_k(k) for k in [1, 2, 4, 8]]
            logger.info(f'Train Recalls: {recall_curr_train_eval}')

            # store for plotting
            iters.append(iteration)
            train_recalls_over_iters.append(recall_curr_train_eval)
            val_recalls_over_iters.append(recall_curr)

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

        # Backward pass and optimization.
        optimizer.zero_grad()   # Clear previous gradients.
        loss.backward()         # Compute gradients.

        # CRITICAL: Perform constrained weight update BEFORE optimizer.step()
        if cfg.LOSSES.NAME == 'multi_prototype_cbml':
            criterion.constrained_weight_update()

        optimizer.step()        # Update model parameters.

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
                    lr=optimizer.param_groups[0]["lr"],
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

    # visualization after training
    # plots_dir = 'outputs/plots'
    # os.makedirs(plots_dir, exist_ok=True)
    # plot scalar trends from logged CSV
    # plot_scalar_trends(log_path='outputs/statistics_log.csv', save_dir=plots_dir)

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