"""
run_mlp_baseline.py
───────────────────
Train and evaluate the supervised MLP baseline across 5 seeds, then
write results into all_results_main.json alongside contrastive model results.

Usage
─────
  python run_mlp_baseline.py --config config.yaml --data_dir data/ --n_epochs 200 \
      --results all_results_main.json
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import load_csv, to_tensors
from baseline import EngineeredBaseline, SupervisedMLPBaseline

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def encode_mlp_scores(mlp_bl, X_np):
    """Return (preds, confidence_scores) from mlp baseline."""
    return mlp_bl.predict(X_np)


def eval_mlp(mlp_bl, mlp_bl_ood, baseline, baseline_ood,
             X_train, y_train, X_val, y_val,
             X_test, y_test, is_hard_neg,
             X_ood_train, y_ood_train, X_ood_test, y_ood_test,
             noise_levels, rng_noise):
    """Evaluate a single MLP baseline across all 5 experiments."""
    X_tr_np  = X_train.cpu().numpy()
    X_te_np  = X_test.cpu().numpy()
    y_te_np  = y_test.cpu().numpy()

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    preds, scores = mlp_bl.predict(X_te_np)
    is_correct = (preds == y_te_np).astype(int)
    f1   = f1_score(y_te_np, preds, average="macro", zero_division=0)
    prec = precision_score(y_te_np, preds, average="macro", zero_division=0)
    rec  = recall_score(y_te_np, preds, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(is_correct, scores)
    except ValueError:
        auroc = float("nan")
    e1 = {"f1": round(f1,4), "precision": round(prec,4),
          "recall": round(rec,4), "auroc": round(auroc,4)}

    # ── Experiment 3: noise robustness ────────────────────────────────────────
    e3 = {}
    for noise in [0.0] + list(noise_levels):
        X_noisy = X_te_np.copy()
        if noise > 0:
            X_noisy = np.clip(X_noisy + rng_noise.normal(0, noise, X_noisy.shape), 0, 1)
        label = f"{int(noise * 100)}%"
        e3[label] = round(mlp_bl.score(X_noisy, y_te_np)["f1"], 4)

    # ── Experiment 4: OOD ─────────────────────────────────────────────────────
    y_ood_te_np = y_ood_test.cpu().numpy()
    ood_f1 = round(mlp_bl_ood.score(X_ood_test.cpu().numpy(), y_ood_te_np)["f1"], 4)

    # ── Experiment 5: hard negatives ──────────────────────────────────────────
    hn_idx = np.where(is_hard_neg)[0]
    hn_auroc = float("nan")
    if len(hn_idx) > 0:
        X_hn = X_te_np[hn_idx]
        y_hn = y_te_np[hn_idx]
        hn_preds, hn_scores = mlp_bl.predict(X_hn)
        is_correct_hn = (hn_preds == y_hn).astype(int)
        try:
            hn_auroc = roc_auc_score(is_correct_hn, hn_scores)
        except ValueError:
            hn_auroc = float("nan")
        hn_auroc = round(hn_auroc, 4)

    return {"exp1": e1, "exp3_f1": e3, "exp4_f1": ood_f1, "exp5_auroc": hn_auroc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--data_dir", default="data/")
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--seeds",    default="0,1,2,3,4")
    p.add_argument("--results",  default="all_results_main.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    print("Loading data …")
    df_train = load_csv(args.data_dir, "train.csv")
    df_val   = load_csv(args.data_dir, "val.csv")
    df_test  = load_csv(args.data_dir, "test.csv")
    df_ood_train = load_csv(args.data_dir, "ood_train.csv")
    df_ood_test  = load_csv(args.data_dir, "ood_test.csv")

    X_train, y_train = to_tensors(df_train, device)
    X_val,   y_val   = to_tensors(df_val,   device)
    X_test,  y_test  = to_tensors(df_test,  device)
    X_ood_train, y_ood_train = to_tensors(df_ood_train, device)
    X_ood_test,  y_ood_test  = to_tensors(df_ood_test,  device)
    is_hard_neg = df_test["is_hard_negative"].values.astype(bool)

    baseline = EngineeredBaseline()
    baseline.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    baseline.tune_threshold(X_val.cpu().numpy(), y_val.cpu().numpy())
    baseline_ood = EngineeredBaseline()
    baseline_ood.fit(X_ood_train.cpu().numpy(), y_ood_train.cpu().numpy())

    noise_levels = cfg["evaluation"]["exp3_noise_levels"]
    rng_noise = np.random.default_rng(0)

    mlp_results = {"seeds": []}
    for seed in seeds:
        t0 = time.time()
        print(f"\nTraining MLP baseline seed={seed} …")
        mlp_bl = SupervisedMLPBaseline(
            input_dim=cfg["data"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            n_layers=cfg["model"]["n_layers"],
            n_classes=cfg["data"]["n_classes"],
            n_epochs=args.n_epochs,
            batch_size=cfg["training"]["batch_size"],
            seed=seed,
        )
        mlp_bl.fit(X_train.cpu().numpy(), y_train.cpu().numpy(),
                   X_val.cpu().numpy(),   y_val.cpu().numpy())

        mlp_bl_ood = SupervisedMLPBaseline(
            input_dim=cfg["data"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            n_layers=cfg["model"]["n_layers"],
            n_classes=cfg["data"]["n_classes"],
            n_epochs=args.n_epochs,
            batch_size=cfg["training"]["batch_size"],
            seed=seed,
        )
        mlp_bl_ood.fit(X_ood_train.cpu().numpy(), y_ood_train.cpu().numpy())

        res = eval_mlp(mlp_bl, mlp_bl_ood, baseline, baseline_ood,
                       X_train, y_train, X_val, y_val,
                       X_test, y_test, is_hard_neg,
                       X_ood_train, y_ood_train, X_ood_test, y_ood_test,
                       noise_levels, rng_noise)
        mlp_results["seeds"].append(res)
        dt = time.time() - t0
        print(f"  seed={seed} F1={res['exp1']['f1']:.4f}  OOD={res['exp4_f1']:.4f}  "
              f"HN_AUROC={res['exp5_auroc']:.4f}  ({dt:.0f}s)")

    # Summary
    f1s = [s["exp1"]["f1"] for s in mlp_results["seeds"]]
    aurocs = [s["exp1"]["auroc"] for s in mlp_results["seeds"]]
    hn_aurocs = [s["exp5_auroc"] for s in mlp_results["seeds"]]
    ood_f1s = [s["exp4_f1"] for s in mlp_results["seeds"]]
    print(f"\n── MLP Baseline Summary (mean ± std over {len(seeds)} seeds) ──")
    print(f"  F1     : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUROC  : {np.nanmean(aurocs):.4f} ± {np.nanstd(aurocs):.4f}")
    print(f"  HN AUROC: {np.nanmean(hn_aurocs):.4f} ± {np.nanstd(hn_aurocs):.4f}")
    print(f"  OOD F1 : {np.mean(ood_f1s):.4f} ± {np.std(ood_f1s):.4f}")

    # Merge into existing results file
    if os.path.exists(args.results):
        with open(args.results) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results["mlp_baseline"] = mlp_results

    with open(args.results, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nMerged MLP baseline results → {args.results}")


if __name__ == "__main__":
    main()
