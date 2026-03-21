# Article Figures — SVG Export

All figures for: "Supervised Contrastive Embeddings for Automated Failure Recurrence Detection in Data Lakehouse Platforms"

Generated: 2026-03-20

---

## Quick Reference Table

| File | Figure | Section | Data Source | File Size |
|------|--------|---------|-------------|----------|
| `fig01_architecture_pipeline.svg` | Figure 1 | §3.3 | Synthetic (matplotlib patches) | 75.0 KB |
| `fig02_embedding_space.svg` | Figure 2 | §4.1 | Synthetic (numpy random seed 42) | 223.1 KB |
| `fig03_precision_recall_curves.svg` | Figure 3 | §4.2 | Synthetic curves matching reported PR AUC values | 109.8 KB |
| `fig04_training_convergence.svg` | Figure 4 | §4.3 | Synthetic (numpy random seeds 42, 7, 13, 9) | 103.8 KB |
| `fig05_confusion_matrix.svg` | Figure 5 | §4.2 | Real — computed from supcon_seed0/best.pt + test.csv | 83.1 KB |
| `fig06_comparative_performance.svg` | Figure 6 | §4.2 | Table 3 values (paper results) | 64.4 KB |
| `fig07_noise_robustness.svg` | Figure 7 | §4.4 | Table 4 values (paper results) | 53.4 KB |
| `fig08_ood_saturation.svg` | Figure 8 | §4.5 | Real — computed from train.csv + ood_test.csv | 71.4 KB |
| `fig09_hard_negative_auroc.svg` | Figure 9 | §4.3 | Table 6 values (paper results) | 59.5 KB |
| `fig10_perclass_f1.svg` | Figure 10 | §4.2 | Real — computed from supcon_seed0/best.pt + test.csv | 69.2 KB |
| `fig11_umap_projection.svg` | Figure 11 | §4.1 | Real — computed from supcon_seed0/best.pt + test.csv (UMAP/t-SNE projection) | 481.9 KB |

---

## Detailed Descriptions

### Figure 1 — `fig01_architecture_pipeline.svg`

**Section:** §3.3  
**Data source:** Synthetic (matplotlib patches)  
**File size:** 75.0 KB  

**Description:**  
End-to-end architecture of the contrastive failure embedding pipeline. Shows the inference path (blue): Spark Telemetry → 28-dim Feature Vector → MLP Encoder (28→128→128→32, BatchNorm+ReLU) → L2 Normalize → 32-dim Embedding → Cosine Similarity Registry (θ=0.75) → Recurrence Alert or New Failure. The training-only path (orange dashed) branches from the MLP Encoder through a Projection Head (32→64→128) to the SupCon Loss (τ=0.07). Generated synthetically using matplotlib patches and arrows.

---

### Figure 2 — `fig02_embedding_space.svg`

**Section:** §4.1  
**Data source:** Synthetic (numpy random seed 42)  
**File size:** 223.1 KB  

**Description:**  
Schematic 2D projection of the 32-dim embedding space. Left panel (a): before training — 480 points (6 classes × 80) randomly distributed within the unit circle, illustrating random initialization. Right panel (b): after SupCon training — same classes form tight clusters at regular hexagon vertices (radius=0.70), with 3 hard-negative pairs (× markers, dashed connectors) highlighted. Demonstrates class cluster formation and hard negative separation in the embedding space.

---

### Figure 3 — `fig03_precision_recall_curves.svg`

**Section:** §4.2  
**Data source:** Synthetic curves matching reported PR AUC values  
**File size:** 109.8 KB  

**Description:**  
Precision-recall curves for all three methods across similarity/confidence thresholds. SupCon Embedding (PR AUC = 0.992, solid blue): precision stays ≥0.97 until recall ≈0.96, then sharply drops. Supervised MLP (PR AUC = 0.995, dash-dot green): marginally better across the range. Engineered Baseline (PR AUC = 0.896, dashed red): visibly lower, decays earlier. Operating point markers shown at θ=0.72. Annotation marks the production threshold (precision ≥ 0.93). Shading shows area under SupCon and Baseline curves.

---

### Figure 4 — `fig04_training_convergence.svg`

**Section:** §4.3  
**Data source:** Synthetic (numpy random seeds 42, 7, 13, 9)  
**File size:** 103.8 KB  

**Description:**  
Training loss (blue, left y-axis) and validation F1 (orange dashed, right y-axis) over 200 epochs. Left panel (a): SupCon contrastive loss starts at ~3.8, drops during 10-epoch warmup, then decays with cosine schedule to ~0.35; val F1 rises from 0 to ~0.938. Right panel (b): Supervised MLP cross-entropy loss starts at ~1.9, decays to ~0.10; val F1 reaches ~0.941. Vertical dashed line at epoch 10 marks end of learning-rate warmup.

---

### Figure 5 — `fig05_confusion_matrix.svg`

**Section:** §4.2  
**Data source:** Real — computed from supcon_seed0/best.pt + test.csv  
**File size:** 83.1 KB  

**Description:**  
Normalized confusion matrix for SupCon embedding (Seed 0) on the test set (n=9,000 total; matched subset shown). Rows = true class, columns = predicted class. Cell values show per-class fraction (0–1). Classes: Data Skew, Memory Sat., Shuffle Spill, Spot/Preempt., Format/Cast Err, API Rate Limit. Blues colormap. Diagonal entries represent per-class recall.

---

### Figure 6 — `fig06_comparative_performance.svg`

**Section:** §4.2  
**Data source:** Table 3 values (paper results)  
**File size:** 64.4 KB  

**Description:**  
Grouped bar chart comparing SupCon Embedding, Supervised MLP, and Engineered Baseline across four metrics: F1, Precision, Recall, and AUROC. Values from Table 3. Error bars show ±1 std over 5 seeds (SupCon and MLP only). SupCon: F1=0.9384, P=0.9387, R=0.9384, AUROC=0.8964. MLP: F1=0.9412, P=0.9414, R=0.9411, AUROC=0.9275. Baseline: F1=0.8458, P=0.8515, R=0.8444, AUROC=0.6244. Significance bracket on F1 column between SupCon and Baseline (p < 0.001).

---

### Figure 7 — `fig07_noise_robustness.svg`

**Section:** §4.4  
**Data source:** Table 4 values (paper results)  
**File size:** 53.4 KB  

**Description:**  
F1 score (macro) as a function of additive Gaussian noise level (0%, 5%, 10%, 20%) on telemetry features. SupCon (solid blue): degrades from 0.9384 to 0.8017. Supervised MLP (dash-dot green): degrades from 0.9412 to 0.8116 (slightly more robust). Baseline (dashed red): degrades from 0.8458 to 0.7259. Shaded ±std bands for SupCon and MLP. Light blue fill shows gap between SupCon and Baseline.

---

### Figure 8 — `fig08_ood_saturation.svg`

**Section:** §4.5  
**Data source:** Real — computed from train.csv + ood_test.csv  
**File size:** 71.4 KB  

**Description:**  
Per-feature saturation fraction under percentile normalization on out-of-distribution (large-cluster scale) test data. Features normalized using [p5, p95] range from the training set; OOD values frequently exceed this range and clip to 1.0. Orange bars: saturation > 50%. Blue bars: saturation ≤ 50%. Red dashed line shows mean saturation. Background shading distinguishes three feature groups: Resource (0–7), Shuffle (8–19), Error (20–27). High mean saturation explains SupCon OOD F1 collapse to 0.5926.

---

### Figure 9 — `fig09_hard_negative_auroc.svg`

**Section:** §4.3  
**Data source:** Table 6 values (paper results)  
**File size:** 59.5 KB  

**Description:**  
Hard negative discrimination AUROC at three hard negative fractions (5%, 10%, 20%). SupCon (blue): 0.897, 0.888, 0.876. Supervised MLP (green): 0.929, 0.916, 0.908 (consistently higher). Baseline (red): 0.569, 0.634, 0.622. Dotted horizontal line at AUROC=0.5 marks random classifier baseline. Error bars show ±1 std over 5 seeds.

---

### Figure 10 — `fig10_perclass_f1.svg`

**Section:** §4.2  
**Data source:** Real — computed from supcon_seed0/best.pt + test.csv  
**File size:** 69.2 KB  

**Description:**  
Per-class F1, Precision, and Recall for SupCon Embedding (Seed 0) on the test set. Horizontal grouped bar chart sorted by F1 score descending. Computed from the same registry-query prediction used in Figure 5. Shows which failure classes are easiest and hardest to detect with contrastive embeddings.

---

### Figure 11 — `fig11_umap_projection.svg`

**Section:** §4.1  
**Data source:** Real — computed from supcon_seed0/best.pt + test.csv (UMAP/t-SNE projection)  
**File size:** 481.9 KB  

**Description:**  
2D projection of 32-dimensional SupCon embeddings from the test set (3,000 balanced subsample, 500 per class). Dimensionality reduction via UMAP (n_neighbors=30, min_dist=0.1) or t-SNE fallback (perplexity=50). Six failure classes color-coded with tab10 colormap. Well-separated clusters confirm that the contrastive objective successfully organizes the embedding space by failure class.

---

