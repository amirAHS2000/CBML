import datetime
import itertools
import time

import numpy as np
import torch

from cbml_benchmark.data.evaluations import RetMetric
from cbml_benchmark.utils.feat_extractor import feat_extractor
from cbml_benchmark.utils.freeze_bn import set_bn_eval
from cbml_benchmark.utils.metric_logger import MetricLogger
from cbml_benchmark.utils.training_diagnostics import (
    gradient_norms, instance_geometry, preserve_rng, recall_from_matrix, sampled_proxy_stats, sync,
)


from cbml_benchmark.utils.mvc_extended_diagnostics import (
    component_gradients, embedding_geometry, proxy_score_geometry,
)


def update_ema_variables(model, ema_model):
    alpha = 0.999
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1-alpha,param.data)


def evaluate_recall(model, loader, logger, iteration=0, split='validation',
                    diagnostics=None, fast=False, criterion=None):
    """Preserve the released CBML retrieval definition and similarity matrix."""
    start = time.perf_counter()
    model.eval()
    labels = np.array([int(k) for k in loader.dataset.label_list])
    feats = feat_extractor(model, loader, logger=logger)
    extraction = time.perf_counter()-start
    t = time.perf_counter()
    metric = RetMetric(feats=feats, labels=labels)
    similarity = time.perf_counter()-t
    t = time.perf_counter()
    if fast:
        recalls = recall_from_matrix(metric.sim_mat, labels)
    else:
        recalls = [metric.recall_k(k) for k in (1, 2, 4, 8)]
    ranking = time.perf_counter()-t
    if diagnostics:
        diagnostics.record_recall(iteration, split, len(labels), recalls,
                                  extraction, similarity, ranking, time.perf_counter()-start)
    logger.info('%s iteration %d recall@1/2/4/8: %s', split, iteration, recalls)
    if diagnostics and getattr(diagnostics.cfg, 'INSTANCE_GEOMETRY', False):
        t = time.perf_counter()
        geometry = instance_geometry(metric.sim_mat, labels)
        diagnostics.event('instance_geometry', iteration=iteration, split=split,
                          total_s=time.perf_counter()-t, **geometry)
    del metric  # release the full similarity matrix before extended geometry
    if diagnostics and getattr(diagnostics.cfg, 'EXTENDED_GEOMETRY', False):
        t = time.perf_counter()
        diagnostics.event('embedding_spectrum', iteration=iteration, split=split,
                          **embedding_geometry(feats, labels))
        if criterion is not None and hasattr(criterion, 'proxies'):
            bank = criterion.proxies.detach().cpu().numpy()
            if split == 'train':
                c, k, d = bank.shape
                diagnostics.event('proxy_spectrum', iteration=iteration, split=split,
                                  **embedding_geometry(bank.reshape(c*k,d), np.repeat(np.arange(c), k)))
            diagnostics.event('proxy_score_geometry', iteration=iteration, split=split,
                              **proxy_score_geometry(
                                  feats, labels, bank, split,
                                  tail_delta=getattr(criterion, 'mvc_tail_delta', 0.0)
                                  if getattr(criterion, 'mvc_mode', 'symmetric') == 'upper_tail'
                                  else 0.0))
        diagnostics.event('extended_geometry_timing', iteration=iteration, split=split,
                          total_s=time.perf_counter()-t)
    if diagnostics and criterion is not None and split == 'train':
        diagnostics.proxy_snapshot(iteration, feats, labels, criterion)
    return recalls


def do_train(
        cfg, model, train_loader, val_loader, optimizer, scheduler,
        criterion, criterion_aux, checkpointer, device, checkpoint_period,
        arguments, logger, proxy_optimizer=None, diagnostics=None,
        train_eval_loader=None,
):
    logger.info('Start training')
    meters = MetricLogger(delimiter='  ')
    max_iter = len(train_loader)
    start_iter = arguments['iteration']
    best_iteration = arguments.get('best_iteration', -1)
    best_recall = arguments.get('best_recall', 0)
    start_training_time = time.perf_counter()
    fast = bool(diagnostics and diagnostics.cfg.FAST_RECALL)
    is_proxy = hasattr(criterion, 'proxies') and hasattr(criterion, 'loss_components')
    count_aware = bool(is_proxy and getattr(criterion, 'count_aware', False))
    if count_aware and (proxy_optimizer is None or criterion_aux is not None):
        raise ValueError('Count-aware updates require a dedicated proxy optimizer and no auxiliary loss.')

    def save(name, iteration):
        start = time.perf_counter()
        checkpointer.save(name, **arguments)
        if diagnostics:
            diagnostics.event('checkpoint', iteration=iteration, name=name,
                              total_s=time.perf_counter()-start)

    end = time.perf_counter()
    for iteration, (images, targets) in enumerate(
            itertools.islice(train_loader, max(0, max_iter-start_iter)), start_iter):
        data_wait = time.perf_counter()-end
        # The original validation schedule evaluates the model BEFORE the update.
        if iteration % cfg.VALIDATION.VERBOSE == 0 or iteration == max_iter:
            logger.info('Validation')
            recall_curr = evaluate_recall(model, val_loader, logger, iteration,
                                          diagnostics=diagnostics, fast=fast,
                                          criterion=criterion if is_proxy else None)
            print(recall_curr)
            if recall_curr[0] > best_recall:
                best_recall, best_iteration = recall_curr[0], iteration
                arguments.update(best_iteration=best_iteration, best_recall=best_recall)
                logger.info('Best iteration %d: recall@1: %.3f', iteration, best_recall)
                save('best_model', iteration)
            else:
                logger.info('Recall@1 at iteration %06d: recall@1: %.3f', iteration, recall_curr[0])
        # Extra train evaluation has its own RNG scope; no sampling or gradients.
        if diagnostics and train_eval_loader is not None and iteration % diagnostics.cfg.TRAIN_RECALL_EVERY == 0:
            with preserve_rng(device):
                evaluate_recall(model, train_eval_loader, logger, iteration, 'train',
                                diagnostics, fast=fast, criterion=criterion if is_proxy else None)

        model.train()
        model.apply(set_bn_eval)
        iteration += 1
        arguments['iteration'] = iteration
        profile = bool(diagnostics and (iteration == 1 or iteration % diagnostics.cfg.STATS_EVERY == 0))
        stages = {}
        if profile:
            sync(device)
        step_start = last = time.perf_counter()

        def mark(name):
            nonlocal last
            if profile:
                sync(device)
                now = time.perf_counter()
                stages[name] = now-last
                last = now

        scheduler.step()  # preserve released ordering for fair comparison
        images = images.to(device)
        targets = targets.to(device)  # collate_fn already returns a contiguous int64 tensor
        mark('transfer_and_scheduler_s')
        feats = model(images)
        mark('forward_s')
        details = None
        loss_pos = loss_neg = loss_mvc = None
        if criterion_aux is not None:
            if cfg.LOSSES.NAME_AUX != 'adv_loss':
                loss = criterion(feats, targets)
                loss_aux = criterion_aux(feats, targets)
            else:
                loss = criterion(feats, targets)
                split_feats = torch.split(feats, cfg.LOSSES.ADV_LOSS.CLASS_DIM, dim=1)
                loss_aux = criterion_aux(split_feats[0], split_feats[1])
            loss = (1-cfg.LOSSES.AUX_WEIGHT)*loss + cfg.LOSSES.AUX_WEIGHT*loss_aux
        elif is_proxy:
            if profile or count_aware:
                loss_pos, loss_neg, loss_mvc, details = criterion.loss_terms(feats, targets, return_details=True)
            else:
                loss_pos, loss_neg, loss_mvc, _ = criterion.loss_terms(feats, targets, return_details=False)
            loss = loss_pos + loss_neg + criterion.mvc_weight * loss_mvc
        else:
            loss = criterion(feats, targets)
        mark('loss_s')
        optimizer.zero_grad()
        if proxy_optimizer is not None:
            proxy_optimizer.zero_grad()
        component_stats = {}
        if (profile and is_proxy and criterion_aux is None
                and getattr(diagnostics.cfg, 'COMPONENT_GRADIENTS', False)):
            with preserve_rng(device):
                # Measure the same unweighted MVC even in the disabled baseline.
                # Never add this diagnostic-only term to the training loss.
                diagnostic_mvc = loss_mvc
                if not criterion.mvc_enabled:
                    diagnostic_mvc, mvc_stats = criterion.mvc_component(
                        feats, targets, return_details=True, diagnostic_only=True)
                    details.update(mvc_stats)
                weight = criterion.mvc_weight if criterion.mvc_enabled else 0.0
                component_stats = component_gradients(model, loss_pos, loss_neg,
                                                      diagnostic_mvc, weight)
                del diagnostic_mvc
            mark('component_gradients_s')
        loss.backward()
        mark('backward_s')

        extra = component_stats
        if profile:
            extra.update(gradient_norms(model))
            if details is not None:
                extra.update(sampled_proxy_stats(details, criterion))
                for key in ('mvc_neg_score_mean', 'mvc_neg_score_std',
                            'mvc_within_anchor_neg_std', 'mvc_positive_score_mean',
                            'mvc_target_mean', 'mvc_residual_std', 'mvc_variance_mean',
                            'mvc_tail_threshold_mean', 'mvc_tail_active_fraction',
                            'mvc_tail_hinge_mean', 'mvc_tail_hinge_max'):
                    if key in details:
                        extra[key] = float(details[key])
                before = criterion.proxies.detach().clone()
            mark('pre_update_diagnostics_s')
        if count_aware:
            extra.update(criterion.scale_proxy_gradients(details, collect_stats=profile))
            mark('proxy_gradient_scaling_s')
        elif profile and details is not None:
            # Identity scaling for the legacy rule; keep comparison fields explicit.
            extra['proxy_grad_norm_before_scaling'] = extra['proxy_grad_norm']
            extra['proxy_grad_norm_after_scaling'] = extra['proxy_grad_norm']
            extra.update(proxy_scale_min=1.0, proxy_scale_mean=1.0, proxy_scale_max=1.0)
        optimizer.step()
        if proxy_optimizer is not None:
            proxy_optimizer.step()
        mark('optimizer_s')
        if profile and details is not None:
            with torch.no_grad():
                delta = criterion.proxies.detach()-before
                extra['proxy_update_norm'] = float(delta.norm())
                assigned = torch.zeros(criterion.num_classes*criterion.k, dtype=torch.bool, device=device)
                assigned[details['labels']*criterion.k+details['pos_k']] = True
                extra['unassigned_proxy_update_norm'] = float(delta.reshape(-1, criterion.dim)[~assigned].norm())
            mark('post_update_diagnostics_s')
        if loss_pos is not None:
            loss_val, pos_val, neg_val, mvc_val = torch.stack([
                loss.detach(), loss_pos.detach(), loss_neg.detach(), loss_mvc.detach()
            ]).cpu().tolist()
            mvc_weighted_val = criterion.mvc_weight * mvc_val
        else:
            loss_val, pos_val, neg_val, mvc_val, mvc_weighted_val = loss.item(), None, None, None, None
        step_s = time.perf_counter()-step_start
        meters.update(time=step_s+data_wait, data=data_wait, loss=loss_val)
        if diagnostics:
            diagnostics.step(iteration=iteration, loss=loss_val, loss_pos=pos_val, loss_neg=neg_val,
                             loss_mvc=mvc_val, loss_mvc_weighted=mvc_weighted_val,
                             mvc_enabled=bool(is_proxy and criterion.mvc_enabled),
                             mvc_weight=criterion.mvc_weight if is_proxy else None,
                             mvc_gamma=criterion.mvc_gamma if is_proxy else None,
                             mvc_mode=criterion.mvc_mode if is_proxy else None,
                             mvc_tail_delta=criterion.mvc_tail_delta if is_proxy else None,
                             batch_size=len(images), data_wait_s=data_wait, step_wall_s=step_s,
                             network_lr=optimizer.param_groups[0]['lr'],
                             proxy_lr=proxy_optimizer.param_groups[0]['lr'] if proxy_optimizer else None,
                             profiled=profile, cuda_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                             **stages, **extra)
        eta_seconds = meters.time.global_avg*(max_iter-iteration)
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info('eta (training only): %s  iter: %d  %s  lr: %.6f  max mem: %.1f GB',
                        datetime.timedelta(seconds=int(eta_seconds)), iteration, meters,
                        optimizer.param_groups[0]['lr'],
                        torch.cuda.max_memory_allocated(device)/1024.0**3 if device.type=='cuda' else 0.)
        if iteration % checkpoint_period == 0 or iteration == max_iter:
            save('model_{:06d}'.format(iteration), iteration)
        end = time.perf_counter()

    # Optional observation of the final checkpoint. Do NOT change legacy best selection.
    if diagnostics and diagnostics.cfg.FINAL_EVAL and max_iter > start_iter:
        with preserve_rng(device):
            evaluate_recall(model, val_loader, logger, max_iter, diagnostics=diagnostics, fast=fast,
                                          criterion=criterion if is_proxy else None)
            if train_eval_loader is not None:
                evaluate_recall(model, train_eval_loader, logger, max_iter, 'train', diagnostics,
                                fast=fast, criterion=criterion if is_proxy else None)
    total = time.perf_counter()-start_training_time
    logger.info('Total training time: %s (%.4f s / completed update)',
                datetime.timedelta(seconds=total), total/max(1,max_iter-start_iter))
    logger.info('Best iteration: %06d | best recall %s', best_iteration, best_recall)
    if diagnostics:
        diagnostics.event('training_complete', completed_updates=max_iter-start_iter,
                          total_s=total, best_iteration=best_iteration, best_recall=best_recall)


def do_test(model, val_loader, logger):
    logger.info('Start testing')
    print(evaluate_recall(model, val_loader, logger))
