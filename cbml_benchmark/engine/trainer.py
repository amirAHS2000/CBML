import datetime
import time

import numpy as np
import torch
import random
from collections import defaultdict
import matplotlib.pyplot as plt

from cbml_benchmark.data.evaluations import RetMetric
from cbml_benchmark.utils.feat_extractor import feat_extractor
from cbml_benchmark.utils.freeze_bn import set_bn_eval
from cbml_benchmark.utils.metric_logger import MetricLogger


def feat_extractor_changed(model, data_loader, logger=None):
    model.eval()
    feats = []
    device = next(model.parameters()).device
    for i, batch in enumerate(data_loader):
        imgs = batch[0].to(device)
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if logger:
            logger.debug(f'Extract Features: [{i + 1}/{len(data_loader)}], Input shape: {imgs.shape}')
        with torch.no_grad():
            out = model(imgs).data.cpu().numpy()
            feats.append(out)
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    feats = np.vstack(feats)
    if logger:
        logger.debug(f'Extracted features shape: {feats.shape}')
    return feats

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

def compute_batched_train_recall(model, train_loader, cfg, iteration, logger, ks=[1,2,4,8]):
    """Compute recall on full train set in mini-batches."""
    model.eval()
    device = next(model.parameters()).device
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            embeddings = model(images)  # [batch_size, embed_dim]
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()  # [N_train, D]
    all_labels = torch.cat(all_labels, dim=0).numpy()  # [N_train]

    # Normalize embeddings
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix in chunks to avoid OOM
    batch_size = 1000  # Adjust based on memory; ~0.5GB per chunk for 5864 samples
    recalls = []
    for k in ks:
        match_counter = 0
        for i in range(0, len(all_embeddings), batch_size):
            query_embeds = all_embeddings[i:i + batch_size]
            query_labels = all_labels[i:i + batch_size]
            sim_matrix = np.dot(query_embeds, all_embeddings.T)  # [batch_size, N_train]

            # Mask self-similarities
            np.fill_diagonal(sim_matrix, -np.inf)

            # Find top-k indices
            topk_idx = np.argpartition(-sim_matrix, k-1, axis=1)[:, :k]
            pred_labels = all_labels[topk_idx]
            correct = np.any(pred_labels == query_labels[:, np.newaxis], axis=2).any(axis=1)
            match_counter += np.sum(correct)

        recall_k = match_counter / len(all_labels)
        recalls.append(recall_k)

    model.train()
    logger.info(f"Train recall at iteration {iteration}: {recalls}")
    return recalls

def do_train(
        cfg,               # Configuration object with training settings.
        model,             # Neural network model to train.
        train_loader,      # DataLoader for training data.
        val_loader,        # DataLoader for validation data.
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

    start_iter = arguments["iteration"]
    best_iteration = -1
    best_recall = 0

    # Store recalls for plotting
    train_recalls_over_iters = []
    val_recalls_over_iters = []
    iters = []

    start_training_time = time.time()
    end = time.time()

    for iteration, (images, targets) in enumerate(train_loader, start_iter):
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
            recall_curr = [ret_metric.recall_k(k) for k in [1, 2, 4, 8]]
            print(recall_curr)
            logger.info(f"The value of theta is: {criterion.show_theta()}")

            if recall_curr[0] > best_recall:
                best_recall = recall_curr[0]
                best_iteration = iteration
                logger.info(f'Best iteration {iteration}: recall@1: {recall_curr[0]:.3f}')
                checkpointer.save(f"best_model")
            else:
                logger.info(f'Recall@1 at iteration {iteration:06d}: recall@1: {recall_curr[0]:.3f}')

            # Compute full train recall every 200 iterations
            if iteration % 200 == 0 or iteration == max_iter:
                train_recalls = compute_batched_train_recall(model, train_loader, cfg, iteration, logger)
                logger.info(f"Overfit gap (train@1 - val@1): {train_recalls[0] - recall_curr[0]:.3f}")

                # Store for plotting
                iters.append(iteration)
                train_recalls_over_iters.append(train_recalls)
                val_recalls_over_iters.append(recall_curr)

                # Plot and save
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

            model.train()
            model.apply(set_bn_eval)

        data_time = time.time() - end
        iteration += 1
        arguments["iteration"] = iteration
        scheduler.step()
        images = images.to(device)
        targets = torch.stack([target.to(device) for target in targets])
        feats = model(images)
        if criterion_aux is not None:
            # Use auxiliary loss if provided.
            if cfg.LOSSES.NAME_AUX != 'adv_loss':
                loss = criterion(feats, targets)  # Primary loss.
                loss_aux = criterion_aux(feats, targets)  # Auxiliary loss.
                # Combine primary and auxiliary losses with a weight.
                loss = (1 - cfg.LOSSES.AUX_WEIGHT) * loss + cfg.LOSSES.AUX_WEIGHT * loss_aux
            else:
                loss = criterion(feats, targets)
                feats = torch.split(feats, cfg.LOSSES.ADV_LOSS.CLASS_DIM, dim=1)
                loss_aux = criterion_aux(feats[0], feats[1])
                loss = (1 - cfg.LOSSES.AUX_WEIGHT) * loss + cfg.LOSSES.AUX_WEIGHT * loss_aux
        else:
            loss = criterion(feats, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time, loss=loss.item())
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

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
