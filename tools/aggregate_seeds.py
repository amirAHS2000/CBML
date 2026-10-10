"""
Aggregate runs over seeds.

Usage (Colab):
  !python3 tools/aggregate_seeds.py \
      --group proto="output/proto2_s*" \
      --group orig="output/orig_noreg_s*" \
      --group origmvc="output/orig_mvc_s*"

Each matched directory must contain results.json (and, for the diagnostics,
gen_diag/gen_diag.jsonl).  Reports mean +- std (ddof=1) over seeds and, if two
groups are given, the paired difference (group1 - group2) matched on seed.
"""
import argparse
import glob
import json
import os

import numpy as np

DIAG_KEYS = ["V1", "V2", "V3", "mean_delta", "z_trip", "trip_err_emp", "V_cls",
             "radius_deg", "center_sep_nn_deg", "eff_rank_all", "mean_emb_norm_sq",
             "mu_pos", "mu_neg"]
SPLITS = ("train", "test", "train_c", "test_c", "test_c_own")


def load_run(d):
    r = json.load(open(os.path.join(d, "results.json")))
    out = {"seed": r["seed"], "dir": d}
    pv = np.array(r["plateau_val"]) * 100 if r["plateau_val"] else np.full(4, np.nan)
    pt = np.array(r["plateau_train"]) * 100 if r["plateau_train"] else np.full(4, np.nan)
    out.update(plateau_val=pv, plateau_train=pt, final_val=np.array(r["final_val"]) * 100,
               final_train=np.array(r["final_train"]) * 100)
    out["gap_R1"] = pt[0] - pv[0]
    for key in ("plateau_val_centered_trainmean", "plateau_val_centered_own"):
        out[key] = np.array(r[key]) * 100 if r.get(key) else None
    # diagnostics averaged over the same plateau evaluations
    p = os.path.join(d, "gen_diag", "gen_diag.jsonl")
    out["diag"] = {}
    if os.path.exists(p):
        recs = [json.loads(l) for l in open(p) if l.strip()]
        for split in SPLITS:
            rr = [x for x in recs if x["split"] == split and x["iter"] > r["plateau_start"]]
            for k in DIAG_KEYS:
                if rr:
                    out["diag"][f"{split}.{k}"] = float(np.mean([x[k] for x in rr]))
    return out


def ms(x):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.2f} +- {x.std(ddof=1) if len(x) > 1 else float('nan'):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help='name="glob"')
    args = ap.parse_args()
    groups = {}
    for g in args.group:
        name, pat = g.split("=", 1)
        dirs = sorted(d for d in glob.glob(pat.strip('"')) if os.path.exists(os.path.join(d, "results.json")))
        groups[name] = [load_run(d) for d in dirs]
        print(f"\n=== {name}: {len(dirs)} runs, seeds {[r['seed'] for r in groups[name]]}")
        if not dirs:
            continue
        for k in range(4):
            print(f"  plateau val R@{[1, 2, 4, 8][k]}: {ms([r['plateau_val'][k] for r in groups[name]])}")
        print(f"  final   val R@1: {ms([r['final_val'][0] for r in groups[name]])}")
        print(f"  plateau train R@1: {ms([r['plateau_train'][0] for r in groups[name]])}")
        print(f"  train-val gap (R@1): {ms([r['gap_R1'] for r in groups[name]])}")
        for key, lab in (("plateau_val_centered_trainmean", "val R@1 centred (train mean)"),
                         ("plateau_val_centered_own", "val R@1 centred (own mean)")):
            vals = [r[key][0] for r in groups[name] if r.get(key) is not None]
            if vals:
                print(f"  plateau {lab}: {ms(vals)}")
        keys = sorted({k for r in groups[name] for k in r["diag"]})
        for k in keys:
            vals = [r["diag"][k] for r in groups[name] if k in r["diag"]]
            print(f"  diag {k:<28s}: {np.mean(vals):.5f} +- {np.std(vals, ddof=1) if len(vals) > 1 else float('nan'):.5f}")

    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sa = {r["seed"]: r for r in groups[a]}
            sb = {r["seed"]: r for r in groups[b]}
            common = sorted(set(sa) & set(sb))
            if common:
                d = np.array([sa[s]["plateau_val"][0] - sb[s]["plateau_val"][0] for s in common])
                print(f"\n=== paired difference {a} - {b} (plateau val R@1) over seeds {common}: "
                      f"{d.mean():+.2f} +- {d.std(ddof=1) if len(d) > 1 else float('nan'):.2f}  "
                      f"(per seed: {np.round(d, 2).tolist()})")


if __name__ == "__main__":
    main()
