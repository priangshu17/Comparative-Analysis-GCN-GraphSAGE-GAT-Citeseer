# GNN Benchmarks on Citeseer

This repository provides a clear, reproducible comparison of **GCN**, **GraphSAGE**, and **GAT** on the **Citeseer** citation network dataset using PyTorch Geometric.  
It includes complete training code, evaluation metrics, graph visualizations, and t-SNE embedding plots for each model.

---

## 📌 Overview

Graph Neural Networks (GNNs) allow message passing on graph-structured data.  
Citeseer is a classic benchmark dataset from the Planetoid suite, consisting of scientific publications connected via citation links.

This project evaluates three widely used GNN architectures:

- **GCN** — Graph Convolutional Network  
- **GraphSAGE** — Aggregates neighborhood features  
- **GAT** — Uses attention to weight neighbors dynamically  

Each model is trained under a consistent pipeline to enable fair comparison.

---

## ⚙️ Hyperparameter Configuration

All three models were trained under a unified pipeline to ensure fairness.  
Below are the exact hyperparameters used for each architecture.

---

### **Common Settings**

| Hyperparameter | Value |
|----------------|-------|
| Dataset        | Citeseer (Planetoid split) |
| Epochs         | 200 |
| Optimizer      | Adam |
| Learning Rate  | 0.01 |
| Weight Decay   | 5e-4 |
| Hidden Dimension | 64 |
| Dropout        | 0.5 (0.6 for GAT) |
| Loss Function  | CrossEntropyLoss |
| Device         | CUDA (if available) |
| Seed           | 42 |

---

### **GCN Configuration**

| Component | Setting |
|----------|----------|
| Layers | 2 GCNConv layers |
| Hidden Dim | 64 |
| Activation | ReLU |
| Dropout | 0.5 |
| Final Layer | Linear(64 → #classes) |
| Notes | Standard 2-layer GCN as in Kipf & Welling |

---

### **GraphSAGE Configuration**

| Component | Setting |
|----------|----------|
| Layers | 2 SAGEConv layers |
| Aggregator | Mean aggregator |
| Hidden Dim | 64 |
| Activation | ReLU |
| Dropout | 0.5 |
| Final Layer | Linear(64 → #classes) |
| Notes | Transductive setup (no sampling) |

---

### **GAT Configuration**

| Component | Setting |
|----------|----------|
| Layer 1 | GATConv(in_dim → 16, heads=4, concat=True) |
| Output of Layer 1 | 16 × 4 = 64 dims |
| Layer 2 | GATConv(64 → 64, heads=1, concat=False) |
| Final Embedding Dim | 64 |
| Activation | ELU |
| Dropout | 0.6 |
| Final Layer | Linear(64 → #classes) |
| Notes | Matches the general design of GAT except fewer heads (paper uses 8) |

---

## 📝 Additional Notes

- Citeseer models are sensitive to learning rate and dropout;  
  larger embeddings (64-dim) can slightly overfit.
- GAT performance strongly depends on number of heads;  
  increasing heads usually boosts Citeseer accuracy.
- We use the standard Planetoid train/val/test masks for reproducibility.

---

## Results

### GCN
| Metric    | Score        |
| --------- | ------------ |
| Accuracy  | `0.6230`  |
| F1 Macro  | `0.5873`   |
| Precision | `0.6261` |
| Recall    | `0.5971`  |

### GraphSAGE
| Metric    | Score         |
| --------- | ------------- |
| Accuracy  | `0.5940`  |
| F1 Macro  | `0.5664`   |
| Precision | `0.5838` |
| Recall    | `0.5679`  |

### GAT
| Metric    | Score        |
| --------- | ------------ |
| Accuracy  | `0.6450`  |
| F1 Macro  | `0.6113`   |
| Precision | `0.6168` |
| Recall    | `0.6174`  |









