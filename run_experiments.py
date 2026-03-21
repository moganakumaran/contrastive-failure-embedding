"""
run_experiments.py
──────────────────
Full 5-seed experimental pipeline for the manuscript.

Runs:
  1. Main comparison: SupCon / NT-Xent / Triplet × 5 seeds
  2. Architecture ablation: embedding_dim ∈ {8,16,32,64,128} × 5 seeds
  3. Depth ablation: n_layers ∈ {1,2,3,4} × 5 seeds
  4. Temperature ablation: τ ∈ {0.05,0.07,0.10,0.20} × 5 seeds
  5. All five evaluation experiments for each trained model

Usage
─────
  python run_experiments.py --config config.yaml --data_dir data/ --n_epochs 200
"""

import argparse
import os
import sys
import time
import json
import yaml
import numpy as np

import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import train as _train_model
from evaluate import load_csv, to_tensors, encode_all
from baseline import EngineeredBaseline, SupervisedMLPBaseline
from encoder import FailureEmbeddingEncoder


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_args(**kwargs):
    """Create a namespace with default training arg values."""
    defaults = dict(loss="supcon", embedding_dim=None, n_layers=None,
                    output_dir=None, seed=0, temperature=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def run_training(cfg, loss, seed, output_dir, embedding_dim=None, n_layers=None,
                 temperature=None):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "best.pt")
    if os.path.exists(ckpt_path):
        print(f"  [skip] checkpoint exists: {ckpt_path}")
        return ckpt_path
    args = make_args(loss=loss, seed=seed, output_dir=output_dir,
                     embedding_dim=embedding_dim, n_layers=n_layers,
                     temperature=temperature)
    _train_model(cfg, args)
    return ckpt_path


def load_model(ckpt_path, cfg, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FailureEmbeddingEncoder(
        input_dim=cfg["data"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        embedding_dim=ckpt.get("embedding_dim", cfg["model"]["embedding_dim"]),
        n_layers=ckpt.get("n_layers", cfg["model"]["n_layers"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt.get("val_f1", float("nan"))


def quick_eval(model, baseline, baseline_ood, mlp_baseline, mlp_baseline_ood,
               X_train, y_train, X_val, y_val,
               X_test, y_test, is_hard_neg,
               X_ood_train, y_ood_train, X_ood_test, y_ood_test,
               cfg, device):
    """
    Run all 5 experiments for a single model checkpoint.

    Embeddings are pre-computed once and reused across experiments to avoid
    redundant forward passes through the encoder.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score
    from encoder import EmbeddingRegistry

    # ── Pre-compute all embeddings once ──────────────────────────────────────
    model.eval()
    with torch.no_grad():
        emb_train     = encode_all(model, X_train)
        emb_val       = encode_all(model, X_val)
        emb_test      = encode_all(model, X_test)
        emb_ood_train = encode_all(model, X_ood_train)
        emb_ood_test  = encode_all(model, X_ood_test)

    # ── Tune threshold on validation set ─────────────────────────────────────
    # Build registry from train embeddings; query val set
    emb_tr_n = F.normalize(emb_train, p=2, dim=1).cpu().numpy()
    emb_vl_n = F.normalize(emb_val,   p=2, dim=1).cpu().numpy()
    y_tr_np  = y_train.cpu().numpy()
    y_vl_np  = y_val.cpu().numpy()

    sim_val = emb_vl_n @ emb_tr_n.T          # (N_val, N_train)
    best_idx_val  = sim_val.argmax(axis=1)
    nearest_val   = y_tr_np[best_idx_val]
    best_sim_val  = sim_val.max(axis=1)

    best_f1, emb_threshold = -1.0, 0.75
    for theta in np.linspace(0.0, 1.0, cfg["evaluation"]["exp2_threshold_steps"]):
        preds = np.where(best_sim_val >= theta, nearest_val, -1)
        f1    = f1_score(y_vl_np, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, emb_threshold = f1, theta

    # ── Experiment 1: Recurrence Detection Accuracy ───────────────────────────
    emb_te_n     = F.normalize(emb_test, p=2, dim=1).cpu().numpy()
    y_te_np      = y_test.cpu().numpy()
    sim_test     = emb_te_n @ emb_tr_n.T
    best_idx_te  = sim_test.argmax(axis=1)
    nearest_te   = y_tr_np[best_idx_te]
    best_sim_te  = sim_test.max(axis=1)

    emb_preds_e1 = np.where(best_sim_te >= emb_threshold, nearest_te, -1)
    is_correct   = (emb_preds_e1 == y_te_np).astype(int)
    emb_f1       = f1_score(y_te_np, emb_preds_e1, average="macro", zero_division=0)
    from sklearn.metrics import precision_score, recall_score
    emb_prec = precision_score(y_te_np, emb_preds_e1, average="macro", zero_division=0)
    emb_rec  = recall_score( y_te_np, emb_preds_e1, average="macro", zero_division=0)
    try:
        emb_auroc = roc_auc_score(is_correct, best_sim_te)
    except ValueError:
        emb_auroc = float("nan")

    bl_preds, bl_scores = baseline.predict(X_test.cpu().numpy())
    bl_is_correct = (bl_preds == y_te_np).astype(int)
    bl_f1    = f1_score(y_te_np, bl_preds, average="macro", zero_division=0)
    bl_prec  = precision_score(y_te_np, bl_preds, average="macro", zero_division=0)
    bl_rec   = recall_score(  y_te_np, bl_preds, average="macro", zero_division=0)
    try:
        bl_auroc = roc_auc_score(bl_is_correct, bl_scores)
    except ValueError:
        bl_auroc = float("nan")

    mlp_preds, mlp_scores = mlp_baseline.predict(X_test.cpu().numpy())
    mlp_is_correct = (mlp_preds == y_te_np).astype(int)
    mlp_f1   = f1_score(y_te_np, mlp_preds, average="macro", zero_division=0)
    mlp_prec = precision_score(y_te_np, mlp_preds, average="macro", zero_division=0)
    mlp_rec  = recall_score(  y_te_np, mlp_preds, average="macro", zero_division=0)
    try:
        mlp_auroc = roc_auc_score(mlp_is_correct, mlp_scores)
    except ValueError:
        mlp_auroc = float("nan")

    print(f"\n─── Experiment 1: Recurrence Detection Accuracy ───")
    print(f"  Embedding : f1={emb_f1:.4f}  prec={emb_prec:.4f}  "
          f"rec={emb_rec:.4f}  auroc={emb_auroc:.4f}")
    print(f"  MLP BL    : f1={mlp_f1:.4f}  prec={mlp_prec:.4f}  "
          f"rec={mlp_rec:.4f}  auroc={mlp_auroc:.4f}")
    print(f"  Eng BL    : f1={bl_f1:.4f}  prec={bl_prec:.4f}  "
          f"rec={bl_rec:.4f}  auroc={bl_auroc:.4f}")
    e1 = {
        "embedding":   {"f1": round(emb_f1,4), "precision": round(emb_prec,4),
                        "recall": round(emb_rec,4), "auroc": round(emb_auroc,4)},
        "mlp_baseline":{"f1": round(mlp_f1,4), "precision": round(mlp_prec,4),
                        "recall": round(mlp_rec,4), "auroc": round(mlp_auroc,4)},
        "baseline":    {"f1": round(bl_f1,4),  "precision": round(bl_prec,4),
                        "recall": round(bl_rec,4),  "auroc": round(bl_auroc,4)},
    }

    # ── Experiment 3: Noise Robustness ────────────────────────────────────────
    print("\n─── Experiment 3: Noise Robustness ───")
    e3 = {}
    rng_noise = np.random.default_rng(0)
    for noise in [0.0] + list(cfg["evaluation"]["exp3_noise_levels"]):
        X_noisy = X_test.cpu().numpy().copy()
        if noise > 0:
            X_noisy = np.clip(X_noisy + rng_noise.normal(0, noise, X_noisy.shape), 0, 1)
        X_noisy_t = torch.tensor(X_noisy, dtype=torch.float32).to(device)
        with torch.no_grad():
            emb_noisy   = encode_all(model, X_noisy_t)
        emb_noisy_n = F.normalize(emb_noisy, p=2, dim=1).cpu().numpy()
        sim_n       = emb_noisy_n @ emb_tr_n.T
        best_idx_n  = sim_n.argmax(axis=1)
        nearest_n   = y_tr_np[best_idx_n]
        best_sim_n  = sim_n.max(axis=1)
        emb_preds_n = np.where(best_sim_n >= emb_threshold, nearest_n, -1)
        emb_f1_n    = f1_score(y_te_np, emb_preds_n, average="macro", zero_division=0)

        bl_metrics = baseline.score(X_noisy, y_te_np)
        bl_f1_n    = bl_metrics["f1"]
        mlp_f1_n   = mlp_baseline.score(X_noisy, y_te_np)["f1"]
        label = f"{int(noise * 100)}%"
        print(f"  Noise {label:>4s}  Emb={emb_f1_n:.4f}  MLP={mlp_f1_n:.4f}  "
              f"Baseline={bl_f1_n:.4f}  Δ={emb_f1_n-bl_f1_n:+.4f}")
        e3[label] = {"embedding_f1": round(emb_f1_n,4),
                     "mlp_f1": round(mlp_f1_n,4),
                     "baseline_f1": round(bl_f1_n,4),
                     "advantage": round(emb_f1_n-bl_f1_n,4)}

    # ── Experiment 4: OOD Generalization ─────────────────────────────────────
    print("\n─── Experiment 4: OOD Generalization ───")
    emb_ood_tr_n = F.normalize(emb_ood_train, p=2, dim=1).cpu().numpy()
    emb_ood_te_n = F.normalize(emb_ood_test,  p=2, dim=1).cpu().numpy()
    y_ood_tr_np  = y_ood_train.cpu().numpy()
    y_ood_te_np  = y_ood_test.cpu().numpy()
    sim_ood      = emb_ood_te_n @ emb_ood_tr_n.T
    best_idx_ood = sim_ood.argmax(axis=1)
    nearest_ood  = y_ood_tr_np[best_idx_ood]
    best_sim_ood = sim_ood.max(axis=1)
    emb_ood_preds = np.where(best_sim_ood >= emb_threshold, nearest_ood, -1)
    emb_ood_f1    = f1_score(y_ood_te_np, emb_ood_preds, average="macro", zero_division=0)

    bl_ood_met  = baseline_ood.score(X_ood_test.cpu().numpy(), y_ood_te_np)
    bl_ood_f1   = bl_ood_met["f1"]
    mlp_ood_f1  = mlp_baseline_ood.score(X_ood_test.cpu().numpy(), y_ood_te_np)["f1"]
    print(f"  Emb OOD F1: {emb_ood_f1:.4f}  MLP: {mlp_ood_f1:.4f}  "
          f"Baseline: {bl_ood_f1:.4f}  Δ={emb_ood_f1-bl_ood_f1:+.4f}")
    e4 = {"embedding_f1": round(emb_ood_f1,4),
          "mlp_f1": round(mlp_ood_f1,4),
          "baseline_f1": round(bl_ood_f1,4),
          "advantage": round(emb_ood_f1-bl_ood_f1,4)}

    # ── Experiment 5: Hard Negative Discrimination ───────────────────────────
    print("\n─── Experiment 5: Hard Negative Discrimination ───")
    hn_idx = np.where(is_hard_neg)[0]
    e5 = {}
    if len(hn_idx) > 0:
        emb_hn_n = emb_te_n[hn_idx]
        y_hn_np  = y_te_np[hn_idx]
        sim_hn   = emb_hn_n @ emb_tr_n.T
        best_idx_hn = sim_hn.argmax(axis=1)
        nearest_hn  = y_tr_np[best_idx_hn]
        sim_hn_max  = sim_hn.max(axis=1)
        is_correct_hn = (nearest_hn == y_hn_np).astype(int)
        try:
            emb_hn_auroc = roc_auc_score(is_correct_hn, sim_hn_max)
        except ValueError:
            emb_hn_auroc = float("nan")

        X_hn_np  = X_test.cpu().numpy()[hn_idx]
        bl_hn_preds, bl_hn_scores = baseline.predict(X_hn_np)
        is_correct_bl_hn = (bl_hn_preds == y_hn_np).astype(int)
        try:
            bl_hn_auroc = roc_auc_score(is_correct_bl_hn, bl_hn_scores)
        except ValueError:
            bl_hn_auroc = float("nan")

        mlp_hn_preds, mlp_hn_scores = mlp_baseline.predict(X_hn_np)
        is_correct_mlp_hn = (mlp_hn_preds == y_hn_np).astype(int)
        try:
            mlp_hn_auroc = roc_auc_score(is_correct_mlp_hn, mlp_hn_scores)
        except ValueError:
            mlp_hn_auroc = float("nan")

        print(f"  Hard negatives: {len(hn_idx):,}")
        print(f"  Embedding AUROC: {emb_hn_auroc:.4f}  MLP: {mlp_hn_auroc:.4f}  "
              f"Baseline AUROC: {bl_hn_auroc:.4f}")
        e5 = {"n_hard_negatives": len(hn_idx),
              "embedding_auroc": round(emb_hn_auroc,4),
              "mlp_auroc":       round(mlp_hn_auroc,4),
              "baseline_auroc":  round(bl_hn_auroc,4),
              "advantage":       round(emb_hn_auroc - bl_hn_auroc,4)}

    return {"exp1": e1, "exp3": e3, "exp4": e4, "exp5": e5,
            "emb_threshold": emb_threshold}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--data_dir",  default="data/")
    p.add_argument("--n_epochs",  type=int, default=200)
    p.add_argument("--seeds",     default="0,1,2,3,4")
    p.add_argument("--runs_dir",  default="runs/")
    p.add_argument("--results",   default="all_results.json")
    p.add_argument("--skip_ablation", action="store_true",
                   help="Skip architecture/temperature ablation (faster)")
    return p.parse_args()


def main():
    args  = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["n_epochs"] = args.n_epochs
    cfg["data"]["output_dir"]   = args.data_dir
    cfg["hardware"]["device"]   = "cpu"   # always CPU (no GPU assumed)

    device = torch.device("cpu")

    # ── Load data ─────────────────────────────────────────────────────────────
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

    # ── Fit baseline ──────────────────────────────────────────────────────────
    print("Fitting baseline …")
    baseline = EngineeredBaseline()
    baseline.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    baseline.tune_threshold(X_val.cpu().numpy(), y_val.cpu().numpy())

    baseline_ood = EngineeredBaseline()
    baseline_ood.fit(X_ood_train.cpu().numpy(), y_ood_train.cpu().numpy())

    # ── Supervised MLP baseline (5 seeds) ─────────────────────────────────────
    print("Training supervised MLP baselines (5 seeds) …")
    mlp_baselines      = []
    mlp_baselines_ood  = []
    for seed in seeds:
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
                   X_val.cpu().numpy(), y_val.cpu().numpy())
        mlp_baselines.append(mlp_bl)
        print(f"  MLP baseline seed={seed} done")

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
        mlp_baselines_ood.append(mlp_bl_ood)

    # ── All results dict ──────────────────────────────────────────────────────
    all_results = {}

    # ── 1. Main comparison: SupCon / NT-Xent / Triplet × 5 seeds ─────────────
    print("\n" + "="*60)
    print("=== Main comparison: SupCon / NT-Xent / Triplet x 5 seeds")
    print("="*60)
    for loss in ["supcon", "nt_xent", "triplet"]:
        all_results[loss] = {"seeds": []}
        for seed in seeds:
            t0 = time.time()
            out_dir  = os.path.join(args.runs_dir, f"{loss}_seed{seed}")
            ckpt     = run_training(cfg, loss, seed, out_dir)
            model, val_f1 = load_model(ckpt, cfg, device)
            print(f"\n  Evaluating {loss} seed={seed} val_F1={val_f1:.4f} …")
            seed_idx = seeds.index(seed)
            res = quick_eval(model, baseline, baseline_ood,
                             mlp_baselines[seed_idx], mlp_baselines_ood[seed_idx],
                             X_train, y_train, X_val, y_val,
                             X_test, y_test, is_hard_neg,
                             X_ood_train, y_ood_train, X_ood_test, y_ood_test,
                             cfg, device)
            res["val_f1"] = val_f1
            all_results[loss]["seeds"].append(res)
            print(f"  Done in {time.time()-t0:.0f}s  "
                  f"test_F1={res['exp1']['embedding']['f1']:.4f}")

    # ── 2. Architecture ablation: embedding_dim ───────────────────────────────
    if not args.skip_ablation:
        print("\n" + "="*60 + "\n=== Architecture: embedding_dim ablation")
        all_results["ablation_dim"] = {}
        for dim in cfg["ablation"]["embedding_dims"]:
            all_results["ablation_dim"][str(dim)] = {"seeds": []}
            for seed in seeds:
                out_dir = os.path.join(args.runs_dir, f"dim{dim}_seed{seed}")
                ckpt    = run_training(cfg, "supcon", seed, out_dir, embedding_dim=dim)
                model, val_f1 = load_model(ckpt, cfg, device)
                seed_idx = seeds.index(seed)
                res = quick_eval(model, baseline, baseline_ood,
                                 mlp_baselines[seed_idx], mlp_baselines_ood[seed_idx],
                                 X_train, y_train, X_val, y_val,
                                 X_test, y_test, is_hard_neg,
                                 X_ood_train, y_ood_train, X_ood_test, y_ood_test,
                                 cfg, device)
                res["val_f1"] = val_f1
                all_results["ablation_dim"][str(dim)]["seeds"].append(res)
                print(f"  dim={dim} seed={seed} val_F1={val_f1:.4f}")

        # ── 3. Architecture ablation: n_layers ───────────────────────────────
        print("\n=== Architecture: n_layers ablation")
        all_results["ablation_depth"] = {}
        for depth in cfg["ablation"]["n_layers"]:
            all_results["ablation_depth"][str(depth)] = {"seeds": []}
            for seed in seeds:
                out_dir = os.path.join(args.runs_dir, f"depth{depth}_seed{seed}")
                ckpt    = run_training(cfg, "supcon", seed, out_dir, n_layers=depth)
                model, val_f1 = load_model(ckpt, cfg, device)
                seed_idx = seeds.index(seed)
                res = quick_eval(model, baseline, baseline_ood,
                                 mlp_baselines[seed_idx], mlp_baselines_ood[seed_idx],
                                 X_train, y_train, X_val, y_val,
                                 X_test, y_test, is_hard_neg,
                                 X_ood_train, y_ood_train, X_ood_test, y_ood_test,
                                 cfg, device)
                res["val_f1"] = val_f1
                all_results["ablation_depth"][str(depth)]["seeds"].append(res)
                print(f"  depth={depth} seed={seed} val_F1={val_f1:.4f}")

        # ── 4. Temperature ablation ───────────────────────────────────────────
        print("\n=== Temperature ablation: τ ∈ {0.05, 0.07, 0.10, 0.20}")
        temps = [0.05, 0.07, 0.10, 0.20]
        all_results["ablation_temp"] = {}
        for tau in temps:
            key = f"tau{tau}"
            all_results["ablation_temp"][key] = {"seeds": []}
            for seed in seeds:
                out_dir = os.path.join(args.runs_dir, f"tau{tau}_seed{seed}")
                ckpt = run_training(cfg, "supcon", seed, out_dir, temperature=tau)
                model, val_f1 = load_model(ckpt, cfg, device)
                seed_idx = seeds.index(seed)
                res = quick_eval(model, baseline, baseline_ood,
                                 mlp_baselines[seed_idx], mlp_baselines_ood[seed_idx],
                                 X_train, y_train, X_val, y_val,
                                 X_test, y_test, is_hard_neg,
                                 X_ood_train, y_ood_train, X_ood_test, y_ood_test,
                                 cfg, device)
                res["val_f1"] = val_f1
                all_results["ablation_temp"][key]["seeds"].append(res)
                print(f"  τ={tau} seed={seed} val_F1={val_f1:.4f}")

    # ── Save raw results ──────────────────────────────────────────────────────
    with open(args.results, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved → {args.results}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY — Main comparison (mean ± std over seeds)")
    print("="*60)
    for loss in ["supcon", "nt_xent", "triplet"]:
        if loss not in all_results:
            continue
        f1s     = [s["exp1"]["embedding"]["f1"] for s in all_results[loss]["seeds"]]
        aurocs  = [s["exp1"]["embedding"]["auroc"] for s in all_results[loss]["seeds"]]
        hn_aur  = [s["exp5"].get("embedding_auroc", float("nan"))
                   for s in all_results[loss]["seeds"]]
        ood_f1  = [s["exp4"]["embedding_f1"] for s in all_results[loss]["seeds"]]
        print(f"\n  {loss.upper()}")
        print(f"    F1        : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"    AUROC     : {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        print(f"    HN AUROC  : {np.nanmean(hn_aur):.4f} ± {np.nanstd(hn_aur):.4f}")
        print(f"    OOD F1    : {np.mean(ood_f1):.4f} ± {np.std(ood_f1):.4f}")

    # Baseline (standalone)
    bl_preds, bl_scores = baseline.predict(X_test.cpu().numpy())
    from sklearn.metrics import f1_score, roc_auc_score
    bl_f1 = f1_score(y_test.cpu().numpy(), bl_preds, average="macro", zero_division=0)
    is_correct = (bl_preds == y_test.cpu().numpy()).astype(int)
    try:
        bl_auroc = roc_auc_score(is_correct, bl_scores)
    except ValueError:
        bl_auroc = float("nan")
    print(f"\n  BASELINE")
    print(f"    F1    : {bl_f1:.4f}")
    print(f"    AUROC : {bl_auroc:.4f}")


if __name__ == "__main__":
    main()
