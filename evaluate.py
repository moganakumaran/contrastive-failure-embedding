"""
evaluate.py
───────────
Five-experiment evaluation harness for the contrastive failure embedding
and engineered baseline, corresponding to Sections 3.1 – 3.6 of the
manuscript.

Experiments
───────────
  1  Recurrence detection accuracy (F1, Precision, Recall, AUROC)
  2  Precision-recall sensitivity across similarity thresholds
  3  Robustness to telemetry noise injection (5%, 10%, 20%)
  4  Out-of-distribution workload scale generalization
  5  Hard negative discrimination (AUROC on hard-neg test subset)

Usage
─────
  # All experiments
  python evaluate.py --config config.yaml --checkpoint runs/nt_xent/best.pt --data_dir data/

  # Single experiment
  python evaluate.py --config config.yaml --checkpoint runs/nt_xent/best.pt --data_dir data/ --exp 3

  # Baseline only
  python evaluate.py --config config.yaml --data_dir data/ --baseline_only
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc,
)

sys.path.insert(0, os.path.dirname(__file__))
from encoder  import FailureEmbeddingEncoder, EmbeddingRegistry
from baseline import EngineeredBaseline, PercentileNormalizer


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_csv(data_dir: str, fname: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, fname))


def to_tensors(df: pd.DataFrame, device: torch.device):
    feat_cols = [c for c in df.columns if c not in ("label", "is_hard_negative")]
    X = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(device)
    y = torch.tensor(df["label"].values,   dtype=torch.long).to(device)
    return X, y


# ── Embedding inference ───────────────────────────────────────────────────────

@torch.no_grad()
def encode_all(model: FailureEmbeddingEncoder,
               X: torch.Tensor,
               batch_size: int = 1024) -> torch.Tensor:
    model.eval()
    parts = [model(X[i: i + batch_size]) for i in range(0, len(X), batch_size)]
    return torch.cat(parts, dim=0)


# ── Embedding-based prediction ─────────────────────────────────────────────────

def embedding_predict(
    model:     FailureEmbeddingEncoder,
    X_train:   torch.Tensor,
    y_train:   torch.Tensor,
    X_query:   torch.Tensor,
    threshold: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build registry from training embeddings; retrieve nearest for each query.

    Returns
    ───────
    pred_labels : (N,) int  — predicted class (-1 = below threshold)
    sim_scores  : (N,) float
    """
    registry = EmbeddingRegistry()
    emb_train = encode_all(model, X_train)
    registry.add(emb_train, y_train)

    emb_query = encode_all(model, X_query)
    sim_scores, pred_labels, _ = registry.query(emb_query, threshold=threshold)
    return pred_labels.cpu().numpy(), sim_scores.cpu().numpy()


def tune_embedding_threshold(
    model:   FailureEmbeddingEncoder,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val:   torch.Tensor,
    y_val:   torch.Tensor,
    n_steps: int = 100,
) -> float:
    """Sweep cosine similarity threshold on the validation set; return θ*."""
    preds_all, scores_all = embedding_predict(model, X_train, y_train, X_val, 0.0)
    # scores_all is max cosine sim regardless of threshold
    emb_train = encode_all(model, X_train)
    emb_val   = encode_all(model, X_val)
    registry  = EmbeddingRegistry()
    registry.add(emb_train, y_train)
    sim_scores, _, _ = registry.query(emb_val, threshold=0.0)
    sim_np = sim_scores.cpu().numpy()

    # Re-retrieve nearest labels unconditionally
    emb_val_n = F.normalize(emb_val, p=2, dim=1)
    emb_tr_n  = F.normalize(emb_train, p=2, dim=1)
    sim_mat   = (emb_val_n @ emb_tr_n.T).cpu().numpy()
    best_idx  = sim_mat.argmax(axis=1)
    nearest   = y_train.cpu().numpy()[best_idx]
    y_np      = y_val.cpu().numpy()

    best_f1, best_theta = -1.0, 0.75
    for theta in np.linspace(0.0, 1.0, n_steps):
        preds = np.where(sim_np >= theta, nearest, -1)
        f1    = f1_score(y_np, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_theta = f1, theta
    return best_theta


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    scores: np.ndarray) -> dict:
    is_correct = (y_pred == y_true).astype(int)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(is_correct, scores)
    except ValueError:
        auroc = float("nan")
    return {"f1": round(f1, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "auroc": round(auroc, 4)}


# ── Experiment 1: Recurrence Detection Accuracy ───────────────────────────────

def exp1_accuracy(model, baseline, X_train, y_train, X_test, y_test,
                  emb_threshold, device) -> dict:
    print("\n─── Experiment 1: Recurrence Detection Accuracy ───")

    # Embedding
    emb_preds, emb_scores = embedding_predict(model, X_train, y_train,
                                               X_test, emb_threshold)
    emb_metrics = compute_metrics(y_test.cpu().numpy(), emb_preds, emb_scores)

    # Baseline
    bl_preds, bl_scores = baseline.predict(X_test.cpu().numpy())
    bl_metrics = compute_metrics(y_test.cpu().numpy(), bl_preds, bl_scores)

    print(f"  Embedding : {emb_metrics}")
    print(f"  Baseline  : {bl_metrics}")
    return {"embedding": emb_metrics, "baseline": bl_metrics}


# ── Experiment 2: Precision-Recall Sensitivity ────────────────────────────────

def exp2_pr_sensitivity(model, baseline, X_train, y_train, X_test, y_test,
                        n_steps, device) -> dict:
    print("\n─── Experiment 2: Precision-Recall Sensitivity ───")

    def pr_auc_sweep(scores, y_true):
        thresholds = np.linspace(scores.min(), scores.max(), n_steps)
        precisions, recalls = [], []
        # Nearest-label lookup (no model needed; just threshold sweep)
        for t in thresholds:
            preds = np.where(scores >= t, -99, -1)   # placeholder; refine below
        # Use sklearn's PR curve on binary "correct recurrence" signal
        is_correct = np.zeros(len(y_true), dtype=int)   # will be set per method
        return None

    # Embedding: binary correct-recurrence signal at many thresholds
    emb_train_emb = encode_all(model, X_train)
    emb_test_emb  = encode_all(model, X_test)
    registry = EmbeddingRegistry()
    registry.add(emb_train_emb, y_train)
    sim_scores, _, _ = registry.query(emb_test_emb, threshold=0.0)
    sim_np    = sim_scores.cpu().numpy()
    # nearest labels
    sim_mat   = (F.normalize(emb_test_emb, p=2, dim=1) @
                 F.normalize(emb_train_emb, p=2, dim=1).T).cpu().numpy()
    best_idx  = sim_mat.argmax(axis=1)
    nearest   = y_train.cpu().numpy()[best_idx]
    y_np      = y_test.cpu().numpy()
    is_correct_emb = (nearest == y_np).astype(int)

    emb_prec, emb_rec, _ = precision_recall_curve(is_correct_emb, sim_np)
    emb_pr_auc = auc(emb_rec, emb_prec)

    # Baseline
    bl_preds_all, bl_scores = baseline.predict(X_test.cpu().numpy())
    is_correct_bl = (bl_preds_all == y_np).astype(int)
    bl_prec, bl_rec, _ = precision_recall_curve(is_correct_bl, bl_scores)
    bl_pr_auc = auc(bl_rec, bl_prec)

    print(f"  Embedding PR AUC : {emb_pr_auc:.4f}")
    print(f"  Baseline  PR AUC : {bl_pr_auc:.4f}")
    return {
        "embedding": {"pr_auc": round(emb_pr_auc, 4),
                      "precision": emb_prec.tolist(),
                      "recall": emb_rec.tolist()},
        "baseline":  {"pr_auc": round(bl_pr_auc, 4),
                      "precision": bl_prec.tolist(),
                      "recall": bl_rec.tolist()},
    }


# ── Experiment 3: Noise Robustness ────────────────────────────────────────────

def exp3_noise_robustness(model, baseline, X_train, y_train, X_test, y_test,
                          noise_levels, emb_threshold, device) -> dict:
    print("\n─── Experiment 3: Noise Robustness ───")
    rng = np.random.default_rng(0)
    results = {}

    for noise in [0.0] + list(noise_levels):
        # Add Gaussian noise to raw test features
        X_np = X_test.cpu().numpy()
        if noise > 0:
            X_noisy = np.clip(X_np + rng.normal(0, noise, X_np.shape), 0.0, 1.0)
        else:
            X_noisy = X_np
        X_noisy_t = torch.tensor(X_noisy, dtype=torch.float32).to(device)

        # Embedding
        emb_preds, emb_scores = embedding_predict(model, X_train, y_train,
                                                   X_noisy_t, emb_threshold)
        emb_f1 = f1_score(y_test.cpu().numpy(), emb_preds,
                          average="macro", zero_division=0)

        # Baseline
        bl_metrics = baseline.score(X_noisy, y_test.cpu().numpy())
        bl_f1 = bl_metrics["f1"]

        label = f"{int(noise * 100)}%"
        print(f"  Noise {label:>4s}  Emb F1={emb_f1:.4f}  Baseline F1={bl_f1:.4f}  "
              f"Δ={emb_f1 - bl_f1:+.4f}")
        results[label] = {"embedding_f1": round(emb_f1, 4),
                          "baseline_f1":  round(bl_f1,  4),
                          "advantage":    round(emb_f1 - bl_f1, 4)}

    return results


# ── Experiment 4: OOD Generalization ─────────────────────────────────────────

def exp4_ood(model, baseline_ood, X_ood_train, y_ood_train,
             X_ood_test, y_ood_test, emb_threshold, device) -> dict:
    print("\n─── Experiment 4: OOD Generalization ───")

    emb_preds, emb_scores = embedding_predict(model, X_ood_train, y_ood_train,
                                               X_ood_test, emb_threshold)
    emb_f1 = f1_score(y_ood_test.cpu().numpy(), emb_preds,
                      average="macro", zero_division=0)

    bl_metrics = baseline_ood.score(X_ood_test.cpu().numpy(),
                                    y_ood_test.cpu().numpy())
    bl_f1 = bl_metrics["f1"]

    print(f"  Embedding OOD F1 : {emb_f1:.4f}")
    print(f"  Baseline  OOD F1 : {bl_f1:.4f}  Δ={emb_f1 - bl_f1:+.4f}")
    return {"embedding_f1": round(emb_f1, 4),
            "baseline_f1":  round(bl_f1,  4),
            "advantage":    round(emb_f1 - bl_f1, 4)}


# ── Experiment 5: Hard Negative Discrimination ───────────────────────────────

def exp5_hard_neg(model, baseline, X_train, y_train, X_test, y_test,
                  is_hard_neg: np.ndarray, emb_threshold, device) -> dict:
    print("\n─── Experiment 5: Hard Negative Discrimination ───")

    # Select hard-negative test records only
    hn_idx = np.where(is_hard_neg)[0]
    if len(hn_idx) == 0:
        print("  No hard negatives found in test set. Check dataset.")
        return {}

    X_hn = X_test[hn_idx]
    y_hn = y_test[hn_idx]

    # Embedding AUROC
    emb_train_emb = encode_all(model, X_train)
    emb_hn        = encode_all(model, X_hn)
    sim_mat  = (F.normalize(emb_hn, p=2, dim=1) @
                F.normalize(emb_train_emb, p=2, dim=1).T).cpu().numpy()
    best_idx = sim_mat.argmax(axis=1)
    nearest  = y_train.cpu().numpy()[best_idx]
    sim_max  = sim_mat.max(axis=1)
    is_correct_emb = (nearest == y_hn.cpu().numpy()).astype(int)
    try:
        emb_auroc = roc_auc_score(is_correct_emb, sim_max)
    except ValueError:
        emb_auroc = float("nan")

    # Baseline AUROC
    X_hn_np = X_hn.cpu().numpy()
    y_hn_np = y_hn.cpu().numpy()
    bl_preds, bl_scores = baseline.predict(X_hn_np)
    is_correct_bl = (bl_preds == y_hn_np).astype(int)
    try:
        bl_auroc = roc_auc_score(is_correct_bl, bl_scores)
    except ValueError:
        bl_auroc = float("nan")

    print(f"  Hard negatives   : {len(hn_idx):,}")
    print(f"  Embedding AUROC  : {emb_auroc:.4f}")
    print(f"  Baseline  AUROC  : {bl_auroc:.4f}  Δ={emb_auroc - bl_auroc:+.4f}")
    return {"n_hard_negatives": len(hn_idx),
            "embedding_auroc": round(emb_auroc, 4),
            "baseline_auroc":  round(bl_auroc,  4),
            "advantage":       round(emb_auroc - bl_auroc, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run the five evaluation experiments.")
    p.add_argument("--config",       default="config.yaml")
    p.add_argument("--checkpoint",   default=None,
                   help="Path to model checkpoint (.pt). If None, skips embedding experiments.")
    p.add_argument("--data_dir",     default="data/")
    p.add_argument("--exp",          type=int, default=None,
                   choices=[1, 2, 3, 4, 5],
                   help="Run a single experiment. Omit to run all.")
    p.add_argument("--baseline_only", action="store_true")
    p.add_argument("--output",       default="evaluation_results.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ─────────────────────────────────────────────────────────────
    df_train  = load_csv(args.data_dir, "train.csv")
    df_val    = load_csv(args.data_dir, "val.csv")
    df_test   = load_csv(args.data_dir, "test.csv")

    X_train, y_train = to_tensors(df_train, device)
    X_val,   y_val   = to_tensors(df_val,   device)
    X_test,  y_test  = to_tensors(df_test,  device)
    is_hard_neg = df_test["is_hard_negative"].values.astype(bool)

    # ── Load OOD data (Experiment 4) ──────────────────────────────────────────
    try:
        df_ood_train = load_csv(args.data_dir, "ood_train.csv")
        df_ood_test  = load_csv(args.data_dir, "ood_test.csv")
        X_ood_train, y_ood_train = to_tensors(df_ood_train, device)
        X_ood_test,  y_ood_test  = to_tensors(df_ood_test,  device)
        has_ood = True
    except FileNotFoundError:
        has_ood = False
        print("OOD data not found; Experiment 4 will be skipped.")

    # ── Load model ────────────────────────────────────────────────────────────
    model = None
    emb_threshold = cfg["baseline"]["similarity_threshold"]

    if not args.baseline_only and args.checkpoint:
        ckpt  = torch.load(args.checkpoint, map_location=device)
        model = FailureEmbeddingEncoder(
            input_dim=cfg["data"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            embedding_dim=ckpt.get("embedding_dim", cfg["model"]["embedding_dim"]),
            n_layers=ckpt.get("n_layers", cfg["model"]["n_layers"]),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {args.checkpoint}  "
              f"(val_F1={ckpt.get('val_f1', '?'):.4f})")

        # Tune threshold on validation set
        if cfg["baseline"]["tune_threshold"]:
            emb_threshold = tune_embedding_threshold(
                model, X_train, y_train, X_val, y_val,
                n_steps=cfg["evaluation"]["exp2_threshold_steps"],
            )
            print(f"Tuned embedding threshold: {emb_threshold:.4f}")

    # ── Fit baseline ──────────────────────────────────────────────────────────
    baseline = EngineeredBaseline(threshold=cfg["baseline"]["similarity_threshold"])
    baseline.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    if cfg["baseline"]["tune_threshold"]:
        bl_threshold = baseline.tune_threshold(
            X_val.cpu().numpy(), y_val.cpu().numpy(),
            n_steps=cfg["evaluation"]["exp2_threshold_steps"],
        )
        print(f"Tuned baseline threshold  : {bl_threshold:.4f}")

    # ── OOD baseline ──────────────────────────────────────────────────────────
    baseline_ood = None
    if has_ood:
        baseline_ood = EngineeredBaseline()
        baseline_ood.fit(X_ood_train.cpu().numpy(), y_ood_train.cpu().numpy())

    # ── Run experiments ───────────────────────────────────────────────────────
    all_results: Dict = {}
    run_all = args.exp is None

    if (run_all or args.exp == 1) and model is not None:
        all_results["exp1"] = exp1_accuracy(
            model, baseline, X_train, y_train, X_test, y_test,
            emb_threshold, device,
        )

    if (run_all or args.exp == 2) and model is not None:
        all_results["exp2"] = exp2_pr_sensitivity(
            model, baseline, X_train, y_train, X_test, y_test,
            cfg["evaluation"]["exp2_threshold_steps"], device,
        )

    if (run_all or args.exp == 3) and model is not None:
        all_results["exp3"] = exp3_noise_robustness(
            model, baseline, X_train, y_train, X_test, y_test,
            noise_levels=cfg["evaluation"]["exp3_noise_levels"],
            emb_threshold=emb_threshold,
            device=device,
        )

    if (run_all or args.exp == 4) and has_ood and model is not None:
        all_results["exp4"] = exp4_ood(
            model, baseline_ood,
            X_ood_train, y_ood_train, X_ood_test, y_ood_test,
            emb_threshold, device,
        )

    if (run_all or args.exp == 5) and model is not None:
        all_results["exp5"] = exp5_hard_neg(
            model, baseline,
            X_train, y_train, X_test, y_test,
            is_hard_neg, emb_threshold, device,
        )

    # ── Save results ──────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
