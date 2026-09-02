# Supervised Contrastive Embeddings for Automated Failure Recurrence Detection in Data Lakehouse Platforms

Supplemental code for the manuscript submitted to *Applied Sciences* (MDPI).

**Author portfolio:** https://moganakumaran.github.io/portfolio/

---

## Repository Structure

```
├── config.yaml                    # All hyperparameters and experiment settings
├── requirements.txt               # Python dependencies
│
├── synthetic_data_generator.py    # Step 1 — generate the 60,000-record benchmark dataset
├── encoder.py                     # Model: MLP backbone + ProjectionHead
├── losses.py                      # SupCon, NT-Xent, and Triplet loss implementations
├── baseline.py                    # Engineered cosine baseline + Supervised MLP baseline
├── train.py                       # Step 2 — training loop (warmup + cosine decay)
├── run_experiments.py             # Step 3 — run all 5 experiments across 5 seeds
├── run_mlp_baseline.py            # Step 3b — MLP baseline evaluation (called by run_experiments.py)
├── evaluate.py                    # Single-checkpoint evaluation harness
│
├── checkpoints/                   # Pre-trained SupCon checkpoints (5 seeds, ~172 KB each)
│   ├── supcon_seed0/best.pt
│   ├── supcon_seed1/best.pt
│   ├── supcon_seed2/best.pt
│   ├── supcon_seed3/best.pt
│   └── supcon_seed4/best.pt
│
├── results/
│   ├── all_results_main.json      # Pre-computed results (5 seeds, all models)
│   └── supplemental_metrics.json  # PR AUC and HN AUROC at 5%/10%/20% fractions
│
├── figures/                       # All paper figures as vector SVG
│   ├── fig01_architecture_pipeline.svg
│   ├── fig02_embedding_space.svg
│   ├── fig03_precision_recall_curves.svg
│   ├── fig04_training_convergence.svg
│   ├── fig05_confusion_matrix.svg
│   ├── fig06_comparative_performance.svg
│   ├── fig07_noise_robustness.svg
│   ├── fig08_ood_saturation.svg
│   ├── fig09_hard_negative_auroc.svg
│   ├── fig10_perclass_f1.svg
│   └── descriptions.md
│
└── docs/
    └── dataset_schema.md          # Feature schema and per-class distribution design
```

---

## Setup

**Python 3.10+ required.** All experiments run on CPU; no GPU required.

```bash
pip install -r requirements.txt
```

---

## Quick Verification (No Training Required)

The pre-computed results in `results/` and pre-trained checkpoints in `checkpoints/` allow
full verification of all paper tables without re-running training.

### Verify Table 3 numbers from JSON

```python
import json, numpy as np

with open("results/all_results_main.json") as f:
    r = json.load(f)

sc = r["supcon"]["seeds"]
f1s = [sc[i]["exp1"]["embedding"]["f1"] for i in range(5)]
print(f"SupCon F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
# => SupCon F1: 0.9384 ± 0.0019

ml = r["mlp_baseline"]["seeds"]
f1s_mlp = [ml[i]["exp1"]["f1"] for i in range(5)]
print(f"MLP F1:    {np.mean(f1s_mlp):.4f} ± {np.std(f1s_mlp):.4f}")
# => MLP F1: 0.9412 ± 0.0019
```

### Evaluate a pre-trained checkpoint directly

```bash
# First generate data (Step 1), then evaluate checkpoint
python synthetic_data_generator.py --config config.yaml

python evaluate.py --config config.yaml \
  --checkpoint checkpoints/supcon_seed0/best.pt \
  --data_dir data/ --exp 1
```

---

## Full Reproduction Pipeline

All scripts must be run from the repository root.

### Step 1 — Generate the synthetic dataset

```bash
python synthetic_data_generator.py --config config.yaml
```

Writes to `data/` (gitignored; reproducible from seed):

| File | Records | Description |
|------|---------|-------------|
| `train.csv` | 42,000 | Training split (70%) |
| `val.csv` | 9,000 | Validation split (15%) |
| `test.csv` | 9,000 | Test split (15%); 20% are hard negatives |
| `ood_train.csv` | 42,000 | OOD registry (large-cluster scale) |
| `ood_test.csv` | 9,000 | OOD queries (withheld scale range) |
| `full_dataset.csv` | 60,000 | Full dataset before splitting |

Each record is a 28-dimensional percentile-relative Apache Spark telemetry vector with a failure
class label (0–5). See `docs/dataset_schema.md` for full feature definitions.

---

### Step 2 — Train models (5 seeds each)

Train the primary SupCon model:

```bash
for seed in 0 1 2 3 4; do
  python train.py --config config.yaml --loss supcon \
    --output_dir runs/supcon_seed${seed} --seed ${seed}
done
```

Train ablation models (NT-Xent and Triplet):

```bash
for seed in 0 1 2 3 4; do
  python train.py --config config.yaml --loss nt_xent \
    --output_dir runs/nt_xent_seed${seed} --seed ${seed}
  python train.py --config config.yaml --loss triplet \
    --output_dir runs/triplet_seed${seed} --seed ${seed}
done
done
```

> The Supervised MLP Baseline is trained inline during Step 3 — no separate checkpoint is saved.

**Training configuration (from `config.yaml`):**

| Parameter | Value |
|-----------|-------|
| Architecture | 28 → 128 → 128 → 32 (hidden_dim=128, embedding_dim=32) |
| Projection head (training only) | 32 → 64 → 128 (discarded at inference) |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Batch size | 256 (class-stratified sampling) |
| Epochs | 200 (10-epoch linear warmup, cosine decay) |
| Loss temperature τ | 0.07 |

---

### Step 3 — Run all 5 experiments

```bash
python run_experiments.py --config config.yaml --results results/all_results_main.json
```

This trains the MLP baseline inline (5 seeds) and evaluates all models.

| Experiment | Description | Paper Table/Figure |
|------------|-------------|-------------------|
| 1 | Recurrence detection accuracy (F1, P, R, AUROC) | Table 3 |
| 2 | Precision-recall sensitivity | Figure 3 |
| 3 | Noise robustness at 0%, 5%, 10%, 20% Gaussian noise | Table 4 |
| 4 | OOD generalization under large-cluster scale shift | Table 7 |
| 5 | Hard negative discrimination AUROC at 20% fraction | Table 5 |

---

### Step 4 — Ablation studies (optional)

```bash
# Embedding dimensionality ablation
for dim in 8 16 32 64 128; do
  for seed in 0 1 2 3 4; do
    python train.py --config config.yaml --loss supcon \
      --embedding_dim ${dim} --output_dir runs/abl_dim${dim}_seed${seed} --seed ${seed}
  done
done

# Encoder depth ablation
for depth in 1 2 3 4; do
  for seed in 0 1 2 3 4; do
    python train.py --config config.yaml --loss supcon \
      --n_layers ${depth} --output_dir runs/abl_depth${depth}_seed${seed} --seed ${seed}
  done
done

# Temperature ablation
for tau in 0.05 0.07 0.10 0.20; do
  for seed in 0 1 2 3 4; do
    python train.py --config config.yaml --loss supcon \
      --temperature ${tau} --output_dir runs/abl_temp${tau}_seed${seed} --seed ${seed}
  done
done
```

Evaluate a single ablation checkpoint:

```bash
python evaluate.py --config config.yaml \
  --checkpoint runs/abl_dim8_seed0/best.pt --data_dir data/ --exp 1
```

---

## Pre-Computed Results Summary

| Method | F1 | Precision | Recall | AUROC |
|--------|----|-----------|--------|-------|
| SupCon (proposed) | 0.9384 ± 0.0019 | 0.9387 ± 0.0018 | 0.9384 ± 0.0018 | 0.8964 ± 0.0090 |
| Supervised MLP | 0.9412 ± 0.0019 | 0.9414 ± 0.0019 | 0.9411 ± 0.0020 | 0.9275 ± 0.0035 |
| Engineered Baseline | 0.8458 | 0.8515 | 0.8444 | 0.6244 |

Both learned methods significantly outperform the engineered baseline (~9 pp F1, p < 0.001).
The SupCon vs. MLP accuracy difference (0.003) is not statistically significant (p = 0.47).
SupCon's key advantage is open-set recognition: new failure classes can be registered at
inference time without retraining.

---

## Failure Class Index

| Label | Class | Spark Signature |
|-------|-------|-----------------|
| 0 | Data Skew | High `partition_size_std`, shuffle read imbalance |
| 1 | Memory Saturation | Peak JVM heap near limit, GC overhead |
| 2 | Shuffle Spill | High `shuffle_spill_bytes`, elevated sort buffer usage |
| 3 | Spot / Preemptible Interruption | Abrupt executor deregistration, TaskSetManager retries |
| 4 | Data Format / Cast Error | Exception classifier signals, low resource utilisation |
| 5 | API Rate Limiting | Throttle events, task retry with backoff pattern |

---

## Key Architecture Notes

**Encoder** (`encoder.py`): three-layer MLP, 28 → 128 → 128 → 32 (BatchNorm + ReLU per layer).
Output is L2-normalized to the unit hypersphere.

**Projection head** (`encoder.py`, `ProjectionHead`): two-layer head, 32 → 64 → 128, used only
during training. The contrastive loss is computed on the 128-dim projected space; the head is
discarded at inference.

- `model.forward(x)` → 32-dim embedding (inference / registry storage)
- `model.forward_with_projection(x)` → 128-dim projection (contrastive loss computation)

**EmbeddingRegistry** (`encoder.py`): stores resolved failure embeddings; supports cosine
similarity retrieval. In production, backed by an ANN index (e.g., FAISS). Here, operates
over dense tensors for evaluation.

**OOD normalization** (`synthetic_data_generator.py`): OOD splits are normalized with the
training-set percentile normalizer — not re-fitted on OOD data. This exposes the saturation
mechanism reported in §5.4: 68.4% of OOD features exceed the training 95th-percentile bound,
collapsing SupCon OOD F1 to 0.5926.

---

## Reproducibility Notes

- **SupCon checkpoints are deterministic**: inference from a saved checkpoint always produces
  the same embeddings and metrics.
- **Supervised MLP baseline**: retrained from scratch at evaluation time; no checkpoint is
  saved. Results match the paper to within ~1–2% across runs.
- **Engineered baseline**: fully deterministic; always matches reported values exactly.
- All experiments were conducted on CPU. Results are numerically identical across CPU
  platforms; GPU results may differ by floating-point rounding in BatchNorm.
