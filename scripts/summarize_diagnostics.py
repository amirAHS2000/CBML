"""Usage: python scripts/summarize_diagnostics.py OUTPUT/diagnostics/RUN"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(path):
    events = read_jsonl(path/'events.jsonl')
    steps = read_jsonl(path/'steps.jsonl')
    with (path/'recall.csv').open(newline='', encoding='utf-8') as f:
        recalls = list(csv.DictReader(f))
    lines = ['CBML diagnostic summary', str(path), '', 'INITIALIZATION']
    for e in events:
        if e['event'] in ('environment','seed','data_setup','model_setup','proxy_init','training_ready'):
            lines.append(json.dumps(e, sort_keys=True))
    lines += ['', 'TIMING (seconds; sampled stage medians, not summed wall time)']
    for key in ('data_wait_s','step_wall_s','transfer_and_scheduler_s','forward_s','loss_s','backward_s','optimizer_s','pre_update_diagnostics_s','proxy_gradient_scaling_s','post_update_diagnostics_s'):
        v = [r[key] for r in steps if r.get(key) is not None]
        if v:
            lines.append(f'{key:30s} median={median(v):.5f}  n={len(v)}')
    for name in ('evaluation','checkpoint','proxy_snapshot','instance_geometry'):
        e = [r for r in events if r['event']==name]
        if e:
            lines.append(f'{name:30s} total={sum(r.get("total_s",0) for r in e):.3f}  count={len(e)}')
    lines += ['', 'RECALL (fractions; same iteration gap = train minus validation)',
              'iteration  train_R1  val_R1   gap_pp  val_R2  val_R4  val_R8']
    by_iter = {}
    for row in recalls:
        by_iter.setdefault(int(row['iteration']), {})[row['split']] = row
    def val(row, key):
        return float(row[key]) if row else None
    def fmt(x):
        return f'{x:.4f}' if x is not None else '   -- '
    for i, row in sorted(by_iter.items()):
        train, validation = row.get('train'),row.get('validation')
        a,b = val(train,'recall_1'),val(validation,'recall_1')
        gap = 100*(a-b) if a is not None and b is not None else None
        lines.append(f'{i:9d}  {fmt(a)}  {fmt(b)}  {fmt(gap)}  '+ '  '.join(fmt(val(validation,f'recall_{k}')) for k in (2,4,8)))
    validation_rows = [r for r in recalls if r['split']=='validation']
    if validation_rows:
        best = max(validation_rows,key=lambda r:float(r['recall_1']))
        lines.append(f'Best observed validation R@1: {float(best["recall_1"]):.4f} at {best["iteration"]} (includes optional final observation)')
    lines += ['', 'PROXY SNAPSHOTS (full training set)']
    for e in events:
        if e['event']=='proxy_snapshot':
            lines.append(json.dumps(e, sort_keys=True))
    lines += ['', 'INSTANCE GEOMETRY (explicit self exclusion; original recall unchanged)']
    for e in events:
        if e['event']=='instance_geometry':
            lines.append(json.dumps(e, sort_keys=True))
    lines += ['', 'LOSS COMPONENTS AND PROXY-MVC (sampled training steps)']
    for key in ('loss', 'loss_pos', 'loss_neg', 'loss_mvc', 'loss_mvc_weighted',
                'mvc_neg_score_mean', 'mvc_neg_score_std', 'mvc_within_anchor_neg_std',
                'mvc_positive_score_mean', 'mvc_target_mean', 'mvc_residual_std',
                'mvc_variance_mean'):
        v = [s[key] for s in steps if s.get(key) is not None]
        if v:
            lines.append(f'{key:35s} median={median(v):.6f} n={len(v)}')
    mvc_steps = [s for s in steps if s.get('mvc_enabled')]
    if mvc_steps:
        lines.append(f"MVC enabled steps: {len(mvc_steps)}; weight={mvc_steps[-1].get('mvc_weight')}; gamma={mvc_steps[-1].get('mvc_gamma')}")
    lines += ['', 'PROXY GRADIENT SCALING (sampled training steps)']
    for key in ('proxy_grad_norm_before_scaling', 'proxy_grad_norm_after_scaling',
                'proxy_scale_min', 'proxy_scale_mean', 'proxy_scale_max'):
        v = [s[key] for s in steps if s.get(key) is not None]
        if v:
            lines.append(f'{key:35s} median={median(v):.6f} n={len(v)}')
    lines += ['', 'LAST SAMPLED TRAINING STEP']
    profiled = [s for s in steps if s.get('profiled')]
    if profiled:
        lines.append(json.dumps(profiled[-1], sort_keys=True))
    lines += ['', 'COMPLETION / ERRORS']
    lines.extend(json.dumps(e,sort_keys=True) for e in events if e['event'] in ('training_complete','training_error'))
    report = '\n'.join(lines)+'\n'
    (path/'summary.txt').write_text(report,encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    print(summarize(p.parse_args().run_dir))
