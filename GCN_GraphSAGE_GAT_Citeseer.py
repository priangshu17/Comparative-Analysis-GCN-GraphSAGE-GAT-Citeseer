"""
GCN vs GraphSAGE vs GAT on Citeseer dataset
- Loads the Citeseer dataset using PyTorch Geometric
- Implements three models
- Trains each model and evaluate on test set
- Extracts node embeddings and visualize them (t-SNE)
- Visualizes the raw graph (node color = label)
- Outputs metrics (Accuracy, Precision, Recall, F1 macro)

Run:
    uv run GCN_GraphSAGE_GAT_citeseer.py
    
Requirements:
    torch, torch_geometric, scikit-learn, matplotlib, networkX
"""

import os 
import random 
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv 
from torch_geometric.utils import to_networkx 
from sklearn.manifold import TSNE 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt 
import networkx as nx 


# Utilities and Reproducibility
SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

SAVE_DIR = "results_citeseer"
os.makedirs(SAVE_DIR, exist_ok = True)


# Dataset
dataset = Planetoid(root = "data/Planetoid", name = "Citeseer")
data = dataset[0].to(DEVICE)
num_features = dataset.num_node_features
num_classes = dataset.num_classes
print(f"Dataset: Citeseer | #nodes = {data.num_nodes} #edges = {data.num_edges} #features = {num_features} #classes = {num_classes}")


# Models

class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout 
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        embeddings = x 
        out = self.lin(embeddings)
        return out, embeddings
    

class GraphSAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout 
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        embeddings = x
        out = self.lin(embeddings)
        return out, embeddings 
    

class GATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads = 4, dropout = 0.6):
        super().__init__()
        # First GAT Layer with multi-head, we'll concat 
        self.gat1 = GATConv(in_dim, hidden_dim // heads, heads = heads, concat = True, dropout = dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads = heads, concat = False, dropout = dropout)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout 
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.gat2(x, edge_index)
        embeddings = x 
        out = self.lin(embeddings)
        return out, embeddings
    
    
# Training / Evaluation

def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    logits, embeddings = model(data.x, data.edge_index)
    preds = logits.argmax(dim = 1).cpu().numpy()
    y = data.y.cpu().numpy()
    
    def masked_metrics(mask):
        mask_idx = mask.cpu().numpy()
        if mask_idx.sum() == 0:
            return {'acc': float('nan')}
        y_true = y[mask_idx]
        y_pred = preds[mask_idx]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average = 'macro', zero_division = 0)
        prec = precision_score(y_true, y_pred, average = 'macro', zero_division = 0)
        rec = recall_score(y_true, y_pred, average = 'macro', zero_division = 0)
        return {'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec}
    
    train_metrics = masked_metrics(data.train_mask)
    val_metrics = masked_metrics(data.val_mask)
    test_metrics = masked_metrics(data.test_mask)
    return train_metrics, val_metrics, test_metrics, preds, embeddings.detach().cpu().numpy()


# Full train / eval pipeline
def run_experiment(model_class, model_name, lr = 0.005, weight_decay = 5e-4, epochs = 200, hidden_dim = 64):
    print(f"\nRunning: {model_name}")
    model = model_class(num_features, hidden_dim, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val = 0.0
    best_test = 0.0
    best_state = None 
    
    
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, data, optimizer, criterion)
        train_m, val_m, test_m, preds, embeddings = test(model, data)
        val_acc = val_m.get('acc', 0)
        test_acc = test_m.get('acc', 0)
        
        if val_acc is not None and not np.isnan(val_acc) and val_acc >= best_val:
            best_val = val_acc
            best_test = test_acc 
            best_state = model.state_dict()
            
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val Acc {val_acc:.4f} | Test Acc {test_acc:.4f}")
            
    # Load Best
    if best_state is not None:
        model.load_state_dict(best_state)
        
    train_m, val_m, test_m, preds, embeddings = test(model, data)
    
    # Save metrics and artifacts
    metrics = {'train': train_m, 'val': val_m, 'test': test_m}
    torch.save({'model_state': model.state_dict(), 'metrics': metrics}, os.path.join(SAVE_DIR, f'{model_name}_checkpoint.pt'))
    
    # Save embeddings & preds
    np.save(os.path.join(SAVE_DIR, f'{model_name}_embeddings.npy'), embeddings)
    np.save(os.path.join(SAVE_DIR, f'{model_name}_preds.npy'), preds)
    
    print(f"Finished {model_name} | Test Acc: {test_m.get('acc'):.4f} | F1: {test_m.get('f1'):.4f}")
    return model, metrics, preds, embeddings 


# Visualization helpers
def plot_raw_graph(data, path=None, size=(8,6)):
    G = to_networkx(data, to_undirected=True)
    # layout
    pos = nx.spring_layout(G, seed=SEED)
    labels = data.y.cpu().numpy()
    plt.figure(figsize=size)
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=labels, cmap='tab10')
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title('Citeseer graph (node color = true label)')
    plt.axis('off')
    if path:
        plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()
    
def plot_embeddings(embeddings, labels, title, path=None, use_tsne=True):
    # embeddings: (N, D)
    if use_tsne:
        reducer = TSNE(n_components=2, random_state=SEED, init='pca')
        emb2d = reducer.fit_transform(embeddings)
    else:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=SEED)
            emb2d = reducer.fit_transform(embeddings)
        except Exception:
            reducer = TSNE(n_components=2, random_state=SEED, init='pca')
            emb2d = reducer.fit_transform(embeddings)


    plt.figure(figsize=(8,6))
    scatter = plt.scatter(emb2d[:,0], emb2d[:,1], c=labels, s=8, cmap='tab10')
    plt.legend(*scatter.legend_elements(), title='classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()
    
    
# Run the three models
experiments = [
    (GCNNet, "GCN"),
    (GraphSAGENet, "GraphSAGE"),
    (GATNet, "GAT")
]

results = {}
for cls, name in experiments:
    model, metrics, preds, embeddings = run_experiment(cls, name, lr = 0.01, epochs = 400, hidden_dim = 64)
    results[name] = {'model': model, 'metrics': metrics, 'preds': preds, 'embeddings': embeddings}
    

# Visualize Graph and Embeddings


# raw graph
plot_raw_graph(data, path=os.path.join(SAVE_DIR, 'citeseer_raw_graph.png'))

# ground-truth embedding plot using raw features transformed (for baseline visualization)
print('\nPlotting raw-features (t-SNE)')
raw_feats = data.x.cpu().numpy()
plot_embeddings(raw_feats, data.y.cpu().numpy(), title='t-SNE of raw node features (Citeseer)', path=os.path.join(SAVE_DIR, 'raw_features_tsne.png'))


# model embeddings
for name in results:
    emb = results[name]['embeddings']
    plot_embeddings(emb, data.y.cpu().numpy(), title=f't-SNE of embeddings: {name}', path=os.path.join(SAVE_DIR, f'{name}_embeddings_tsne.png'))

# Comparative Metrics Table
print('\n--- Comparative Metrics (Test set) ---')
print(f"{'Model':<12} {'Acc':>6} {'F1':>8} {'Prec':>8} {'Rec':>8}")
for name in results:
    m = results[name]['metrics']['test']
    print(f"{name:<12} {m.get('acc',0):6.4f} {m.get('f1',0):8.4f} {m.get('prec',0):8.4f} {m.get('rec',0):8.4f}")


# Save comparison to file
with open(os.path.join(SAVE_DIR, 'comparison_metrics.txt'), 'w') as f:
    f.write('Model\tAcc\tF1\tPrec\tRec\n')
    for name in results:
        m = results[name]['metrics']['test']
        f.write(f"{name}\t{m.get('acc',0):.4f}\t{m.get('f1',0):.4f}\t{m.get('prec',0):.4f}\t{m.get('rec',0):.4f}\n")


print('\nAll artifacts (plots, checkpoints) are saved in', SAVE_DIR)
print('Script finished.')