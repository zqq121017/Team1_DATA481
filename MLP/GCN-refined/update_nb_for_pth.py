import nbformat as nbf
import os
import glob

nb_path = 'Model_Applications_Insights.ipynb'
if not os.path.exists(nb_path):
    print("Notebook not found!")
    exit(1)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

code_graph = """import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# We redefine the model classes here so we don't accidentally execute the 
# training loop hidden inside the gcn_sensor_graph.py script when importing.
class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        return self.linear(out)

class SensorGraphModel(nn.Module):
    def __init__(self, embed_dim=64, gcn_dim=64):
        super().__init__()
        
        self.act_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, embed_dim)
        )
        self.pup_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, embed_dim)
        )
        self.eeg_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(32, embed_dim)
        )
        
        # Adaptive Adjacency Matrix (5 nodes: 3 Brain, 1 Behavior, 1 Autonomic)
        self.adj = nn.Parameter(torch.rand(5, 5))
        
        self.gcn1 = GraphConvLayer(embed_dim, gcn_dim)
        self.gcn2 = GraphConvLayer(gcn_dim, gcn_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(gcn_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )
        
    def forward(self, eeg, act, pup):
        B = eeg.size(0)
        
        # Encode Autonomic (Pupil) and Behavior (Action) Nodes
        node_act = self.act_cnn(act).unsqueeze(1) # (B, 1, embed_dim)
        node_pup = self.pup_cnn(pup).unsqueeze(1) # (B, 1, embed_dim)
        
        # Encode Brain Nodes (EEG electrodes Fz, Cz, POz)
        eeg_nodes = []
        for i in range(3):
            eeg_nodes.append(self.eeg_cnn(eeg[:, i]).unsqueeze(1)) 
            
        # Combine all 5 nodes: (B, 5, embed_dim)
        nodes = torch.cat([eeg_nodes[0], eeg_nodes[1], eeg_nodes[2], node_act, node_pup], dim=1)
        
        # Normalize adaptive adjacency matrix
        adj_norm = torch.softmax(self.adj, dim=-1)
        
        # Message Passing
        x = torch.relu(self.gcn1(nodes, adj_norm))
        x = torch.relu(self.gcn2(x, adj_norm))
        
        # Pooling (Mean across all nodes)
        graph_embed = x.mean(dim=1) 
        
        # Final Prediction
        return self.fc(graph_embed)

# Instantiate the model
model = SensorGraphModel()

# Look for the latest trained .pth file for the GCN
log_dir = "training_logs"
if not os.path.exists(log_dir):
    log_dir = "MLP/GCN-refined/training_logs"
pth_files = sorted(glob.glob(os.path.join(log_dir, "*gcn_sensor_graph*.pth")))

if pth_files:
    latest_pth = pth_files[-1]
    print(f"Loading trained weights from: {latest_pth}")
    model.load_state_dict(torch.load(latest_pth, map_location=torch.device('cpu')))
else:
    print("No trained .pth file found! Using randomly initialized weights.")

# The adjacency matrix (adj) is a learned parameter of shape (5, 5)
adj_matrix = model.adj.detach()

# The model uses softmax to normalize the graph connections during forward pass
adj_norm = torch.softmax(adj_matrix, dim=-1).numpy()

labels = ['EEG: Fz', 'EEG: Cz', 'EEG: POz', 'Behavior: Action', 'Autonomic: Pupil']

plt.figure(figsize=(8, 6))
sns.heatmap(adj_norm, annot=True, cmap='viridis', xticklabels=labels, yticklabels=labels)
plt.title("Learned Feature Graph (Adjacency Matrix)\\nHow Sensors Communicate During Prediction")
plt.ylabel("Source Node")
plt.xlabel("Receiving Node")
plt.show()

print("Insight: High values in this heatmap indicate strong functional connectivity learned by the GCN.")
"""

nb.cells[3].source = code_graph

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook updated to load .pth file.")
