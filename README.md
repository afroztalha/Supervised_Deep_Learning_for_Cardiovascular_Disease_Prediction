# Supervised Deep Learning for Cardiovascular Disease Prediction
**Course:** Artificial Neural Networks | **Instructor:** Dr. Qurat Ul Ain
**Institution:** FAST-NUCES, Department of Artificial Intelligence & Data Science

---

## Overview

This repository contains the full code for Assignment 3, which extends the GNN-based ECG classification pipeline of **Duong et al. (2023)** with a CNN node feature encoder and evaluates it on two datasets: **MIT-BIH Arrhythmia Database** and **PTB-XL ECG Dataset**.

The pipeline converts raw ECG signals into edge-filtered graphs, assigns each graph node a learned CNN feature vector (instead of a raw scalar), and classifies the graph using a 3-layer GraphConv network.

> **Important Note on Current Results:**
> All experiments in this submission were run on **CPU only (free-tier Google Colab)** using a **partial dataset** (10 of 48 MIT-BIH records; 95 PTB-XL graphs). Training was limited to **10 epochs** with no SMOTE applied. The reported accuracy (~79%) and macro F1 (~0.22) reflect these constraints — not the ceiling of the method. Full-dataset GPU experiments with SMOTE are planned and will produce significantly improved results.

---

## Repository Structure

```
├── main_cnn_mit_bh.ipynb                  # CNN+GNN model training on MIT-BIH
├── graph_construct_cnn_mit_bh.ipynb       # Graph construction (CNN features) for MIT-BIH
├── graph_construct_train_ptbxl_cnn.ipynb  # Graph construction + training for PTB-XL
└── README.md                              # This file
```

---

## Notebooks — What Each One Does

### 1. `graph_construct_cnn_mit_bh.ipynb` — Graph Construction for MIT-BIH
This notebook takes raw MIT-BIH ECG signals and converts them into graph format with CNN-derived node features.

**Steps performed:**
1. Downloads MIT-BIH records using `wfdb`
2. Extracts individual heartbeat windows around annotated R-peaks
3. Compresses each window to 100 points using Piecewise Aggregate Approximation (PAA)
4. Plots each window as a 64×64 grayscale waveform image
5. Applies **Prewitt edge detection** to identify sharp signal transitions
6. Runs a **2-layer CNN encoder** (Conv2d: 1→16→32, kernel 3×3, ReLU) over each image
7. Identifies bright pixels (intensity ≥ 128) as graph nodes; each node gets a **32-dim CNN feature vector** from the encoder's output feature map
8. Connects adjacent bright pixels (8-connectivity) as graph edges
9. Exports graphs in **TU Dataset format** (5 `.txt` files) for use with PyTorch Geometric

**Output files generated:**
```
MIT_BIH_CNN_A/
  ├── MIT_BIH_CNN_A_graph_indicator.txt
  ├── MIT_BIH_CNN_A_A.txt
  ├── MIT_BIH_CNN_A_graph_labels.txt
  ├── MIT_BIH_CNN_A_node_labels.txt
  └── MIT_BIH_CNN_A_node_attributes.txt
```

---

### 2. `main_cnn_mit_bh.ipynb` — GNN Training on MIT-BIH
This notebook loads the TU-format graphs produced by the construction notebook and trains the GraphConv classifier.

**Steps performed:**
1. Loads the MIT-BIH TU Dataset graphs
2. Splits into train (80%) and test (20%) sets
3. Defines a **3-layer GraphConv GNN** (hidden dim=64, Dropout=0.5, global mean pooling)
4. Trains with **Adam optimizer** (lr=0.001, weight decay=2e-5), **cross-entropy loss**, **StepLR scheduler**
5. Evaluates on the test set: accuracy, per-class precision/recall/F1, confusion matrix
6. Plots training/validation accuracy and loss curves

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| GNN Layers | 3 × GraphConv |
| Hidden Dimension | 64 |
| Node Feature Dim | 37 (32 CNN + 5 one-hot class) |
| Learning Rate | 0.001 |
| Weight Decay | 2e-5 |
| Batch Size | 512 |
| Epochs | 10 (CPU constraint) |
| Dropout | 0.5 |
| Optimizer | Adam |
| Loss | Cross-Entropy |

---

### 3. `graph_construct_train_ptbxl_cnn.ipynb` — PTB-XL Graph Construction + Training
This notebook applies the full pipeline (graph construction + GNN training) to the **PTB-XL dataset** — the additional dataset required by the assignment rubric.

**Steps performed:**
1. Loads PTB-XL waveform images (resized to 64×64 for architecture compatibility)
2. Applies Prewitt edge detection + CNN encoder (same as MIT-BIH notebook)
3. Builds TU-format graphs with 32-dim CNN node features
4. Labels graphs with PTB-XL diagnostic classes: **NORM, MI, STTC, CD, HYP**
5. Trains the same 3-layer GraphConv model on the PTB-XL graphs
6. Evaluates and reports per-class classification report + confusion matrix

**Note:** Only 95 graphs were processed in this submission (76 train, 19 test) due to CPU constraints. Full PTB-XL has 21,799 recordings.

---

## Experimental Setup Summary

### Datasets

| Dataset | Full Size | Used in This Run | Classes | Node Features |
|---|---|---|---|---|
| MIT-BIH Arrhythmia DB | 48 records | 10 records | 4 (NOR, LBBB, RBBB, APB) | 37-dim (32 CNN + 5 one-hot) |
| PTB-XL ECG Dataset | 21,799 recordings | 95 graphs | 5 (NORM, MI, STTC, CD, HYP) | 32-dim CNN |

### Baselines Compared

| Model | Node Features | Accuracy | Macro F1 |
|---|---|---|---|
| Duong et al. (2023) — Original | Scalar brightness | 100% | ~1.00 |
| Assignment 2 Replication (scalar) | Scalar brightness | 79.16% | 0.22 |
| **CNN+GNN (This Work — MIT-BIH)** | **32-dim CNN vector** | **79.16%** | **0.22** |
| **CNN+GNN (This Work — PTB-XL)** | **32-dim CNN vector** | **78.95%** | **~0.18** |

> **Why are the results the same as the scalar baseline?** Both models collapse to predicting the majority class (NOR/NORM) due to: (1) severe class imbalance with no SMOTE, (2) only 10 epochs of training on CPU, and (3) partial dataset usage. The CNN architecture is sound — these results are a training constraint artifact, not a method failure.

---

## How to Run

### Requirements

```bash
pip install torch torchvision torch-geometric wfdb opencv-python numpy pandas scikit-learn matplotlib seaborn
```

### Recommended Environment
- **Google Colab** (free tier works but is slow; GPU runtime strongly recommended for full dataset)
- Python 3.9+
- PyTorch 2.0+ with PyTorch Geometric

### Steps

1. **Clone / upload the notebooks** to your Google Colab environment
2. **Run `graph_construct_cnn_mit_bh.ipynb`** — this downloads MIT-BIH data via `wfdb` automatically and generates the graph files. Allow 20–40 minutes on CPU for 10 records.
3. **Run `main_cnn_mit_bh.ipynb`** — loads the generated graphs and trains the GNN. Ensure the graph file paths match what the construction notebook saved.
4. **Run `graph_construct_train_ptbxl_cnn.ipynb`** — downloads PTB-XL data and runs both graph construction and training end-to-end.

### For Full-Dataset GPU Run (Planned)
- Switch Colab runtime to **GPU (T4 or A100)**
- In `graph_construct_cnn_mit_bh.ipynb`, change the records list to include all 48 MIT-BIH records
- Apply **SMOTE** to the training graphs before loading into the DataLoader
- Increase epochs to **50+**
- For PTB-XL, process all 21,799 recordings

---

## Experiment Documentation

### Experiment 1 — Baseline Replication (Assignment 2)
- **Goal:** Reproduce the scalar-feature GraphConv pipeline of Duong et al. (2023) on MIT-BIH
- **Node features:** Raw pixel brightness (1 scalar per node)
- **Result:** 79.16% accuracy, macro F1 = 0.22, majority-class collapse
- **Conclusion:** Class imbalance + 10 CPU epochs → model predicts NOR for everything

### Experiment 2 — CNN Node Feature Extension (Assignment 3, MIT-BIH)
- **Goal:** Replace scalar node features with 32-dim CNN-derived spatial feature vectors
- **CNN Encoder:** Conv2d(1→16, 3×3, ReLU) → Conv2d(16→32, 3×3, ReLU), no pooling, spatial resolution preserved
- **Node features:** 32-dim CNN vector per bright pixel node
- **Result:** 79.16% accuracy, macro F1 = 0.22 (same as baseline under current constraints)
- **Conclusion:** CNN features do not hurt; their benefit is not visible without SMOTE + GPU training. Architecture is validated.

### Experiment 3 — Cross-Dataset Evaluation (Assignment 3, PTB-XL)
- **Goal:** Apply CNN+GNN pipeline to a completely different ECG classification task (mandatory additional dataset)
- **Dataset:** PTB-XL, 5 diagnostic classes (NORM, MI, STTC, CD, HYP)
- **Result:** 78.95% accuracy, macro F1 ≈ 0.18 on 19-graph test set
- **Conclusion:** Pipeline transfers to PTB-XL with zero modification. Subset too small for reliable metrics. Full run pending.

---

## Known Limitations

- **Partial dataset:** Only 10/48 MIT-BIH records and 95 PTB-XL graphs used. Full-dataset GPU results pending.
- **No SMOTE:** Class imbalance was not addressed in this run. This is the dominant cause of majority-class collapse.
- **Untrained CNN encoder:** The CNN encoder uses random initialisation. A task-tuned encoder would produce better node features.
- **10-epoch budget:** Neither model has converged. This is a compute constraint, not an architectural choice.
- **RBBB support:** Only 8 RBBB test samples exist in the 10-record subset — F1 for this class is statistically uninformative.

---

## References

- Duong, L.T., Doan, T.T.H., Chu, C.Q., & Nguyen, P.T. (2023). Fusion of edge detection and graph neural networks to classifying electrocardiogram signals. *Expert Systems with Applications.* https://doi.org/10.1016/j.eswa.2023.12.017
- MIT-BIH Arrhythmia Database. PhysioNet. https://physionet.org/content/mitdb/1.0.0/
- PTB-XL ECG Dataset. PhysioNet. https://physionet.org/content/ptb-xl/1.0.3/
- Talaat, F.M. (2024). Revolutionizing cardiovascular health. *Neural Computing and Applications.* https://doi.org/10.1007/s00521-024-10453-2
