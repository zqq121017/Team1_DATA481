import pandas as pd
import numpy as np
import pickle
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import stft as scipy_stft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION 
# ═══════════════════════════════════════════════════════════════════════
EEG_CH_INDICES = list(range(20))       # All 20 channels
FREQ_BANDS     = {"delta": (0.5, 4.0), "alpha": (8.0, 13.0)}
SFREQ          = 256
NPERSEG        = 64
NOVERLAP       = 56

HPARAMS = {
    "diff_map":           {"Easy": 1, "Medium": 2, "Hard": 3},
    "batch_size":         32,
    "epochs":             300,
    "learning_rate":      0.001,
    "weight_decay":       1e-3,
    "kfold_splits":       5,
    "node_embed_dim":     64,
    "gcn_hidden_dim":     64,
}

# ═══════════════════════════════════════════════════════════════════════
#  EEG FEATURE EXTRACTION (STFT)
# ═══════════════════════════════════════════════════════════════════════
def eeg_to_tfpower_uniform(raw_20ch_384):
    maps = []
    for ch_idx in EEG_CH_INDICES:
        signal = raw_20ch_384[ch_idx].astype(np.float64)
        freqs, _, Zxx = scipy_stft(
            signal, fs=SFREQ, nperseg=NPERSEG, noverlap=NOVERLAP,
            boundary="zeros", padded=True,
        )
        power = np.abs(Zxx) ** 2
        for lo, hi in FREQ_BANDS.values():
            mask = (freqs >= lo) & (freqs <= hi)
            band_power = power[mask, :].T.astype(np.float32)
            maps.append(band_power)
    
    # Pad to uniform F width
    f_max = max(m.shape[1] for m in maps)
    padded = []
    for m in maps:
        pad_w = f_max - m.shape[1]
        padded.append(np.pad(m, ((0, 0), (0, pad_w))) if pad_w > 0 else m)
    
    # Returns shape (40, T, F_max). The 40 bands are:
    # Ch0_delta, Ch0_alpha, Ch1_delta, Ch1_alpha, ..., Ch19_delta, Ch19_alpha
    return np.stack(padded, axis=0).astype(np.float32)

def pilot_to_tfpower(series):
    return np.stack([eeg_to_tfpower_uniform(arr) for arr in series.values])

def pilot_average_bio(row, prefix):
    yaw = np.array(row[f'yaw{prefix}'], dtype=float)
    pitch = np.array(row[f'pitch{prefix}'], dtype=float)
    thrust = np.array(row[f'thrust{prefix}'], dtype=float)
    return (yaw + pitch + thrust) / 3.0

# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING & MERGING
# ═══════════════════════════════════════════════════════════════════════
base_path  = "../../../data"
merge_keys = ['teamID', 'sessionID', 'trialID', 'ringID']

def load_pkl(filename):
    with open(os.path.join(base_path, filename), 'rb') as f:
        return pickle.load(f)

print("Loading and merging data...")
# NOTE: Speech signal is removed as per instruction
mod_files = ['epoched_eeg.pkl', 'epoched_pupil.pkl', 'epoched_action.pkl']
df_main = load_pkl('team_performance.pkl')[merge_keys + ['difficulty']]

for mod_file in mod_files:
    mod_df = load_pkl(mod_file)
    to_drop = ['difficulty', 'communication']
    cols_to_drop = [c for c in to_drop if c in mod_df.columns]
    if cols_to_drop: mod_df = mod_df.drop(columns=cols_to_drop)
    df_main = df_main.merge(mod_df, on=merge_keys, how='inner')
    del mod_df
    gc.collect()

df_main['ring_score'] = df_main['difficulty'].map(HPARAMS["diff_map"])
df_main = df_main.dropna(subset=['ring_score']).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════
#  PROCESSING & AGGREGATION
# ═══════════════════════════════════════════════════════════════════════
print("Processing EEG...")
eeg_yaw = pilot_to_tfpower(df_main['yawEEG'])
eeg_pitch = pilot_to_tfpower(df_main['pitchEEG'])
eeg_thrust = pilot_to_tfpower(df_main['thrustEEG'])
eeg_combined = (eeg_yaw + eeg_pitch + eeg_thrust) / 3.0
df_main['eeg_data'] = [eeg_combined[i] for i in range(len(eeg_combined))]
del eeg_yaw, eeg_pitch, eeg_thrust, eeg_combined
gc.collect()

print("Processing Bio signals...")
for bio in ['Action', 'Pupil']:
    df_main[f'avg{bio}'] = df_main.apply(lambda r: pilot_average_bio(r, bio), axis=1)

print("Aggregating to trial level...")
trial_keys = ['teamID', 'sessionID', 'trialID']

def aggregate_eeg(series):
    return np.mean(list(series), axis=0) # (40, T, F)

def aggregate_bio_mean(series):
    return np.mean(list(series), axis=0) # (T_fixed,)

trial_df = df_main.groupby(trial_keys).agg({
    'ring_score': 'sum',
    'eeg_data': aggregate_eeg,
}).reset_index()

for bio in ['Action', 'Pupil']:
    bio_seq = df_main.groupby(trial_keys)[f'avg{bio}'].apply(aggregate_bio_mean).reset_index()
    bio_seq.rename(columns={f'avg{bio}': f'{bio}_seq'}, inplace=True)
    trial_df = trial_df.merge(bio_seq, on=trial_keys)

trial_df.rename(columns={'ring_score': 'trial_total_score'}, inplace=True)

# Build Tensors
X_eeg = np.stack(trial_df['eeg_data'].values).astype(np.float32) # (N, 40, T, F)

# Reshape EEG from (N, 40, T, F) to (N, 20 electrodes, 2 bands, T, F)
X_eeg = X_eeg.reshape(X_eeg.shape[0], 20, 2, X_eeg.shape[2], X_eeg.shape[3])

X_act = np.stack(trial_df['Action_seq'].values).astype(np.float32)[:, np.newaxis, :] # (N, 1, T_act)
X_pup = np.stack(trial_df['Pupil_seq'].values).astype(np.float32)[:, np.newaxis, :] # (N, 1, T_pup)
y = trial_df['trial_total_score'].values.astype(np.float32).reshape(-1, 1)

# Normalization
def normalize_3d(data):
    N, C, T = data.shape
    reshaped = data.transpose(0, 2, 1).reshape(-1, C)
    scaler = StandardScaler()
    normed = scaler.fit_transform(reshaped)
    return normed.reshape(N, T, C).transpose(0, 2, 1)

def normalize_5d(data):
    N, E, C, T, F = data.shape
    reshaped = data.transpose(0, 3, 4, 1, 2).reshape(-1, E*C)
    scaler = StandardScaler()
    normed = scaler.fit_transform(reshaped)
    return normed.reshape(N, T, F, E, C).transpose(0, 3, 4, 1, 2)

X_eeg = normalize_5d(X_eeg)
X_act = normalize_3d(X_act)
X_pup = normalize_3d(X_pup)

print(f"Data Prepared: EEG {X_eeg.shape}, Act {X_act.shape}, Pup {X_pup.shape}")

# ═══════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION (GCN SENSOR FUSION)
# ═══════════════════════════════════════════════════════════════════════
class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        # x: (B, N_nodes, in_dim)
        # adj: (N_nodes, N_nodes)
        out = torch.matmul(adj, x)
        return self.linear(out)

class SensorGraphModel(nn.Module):
    def __init__(self, embed_dim=64, gcn_dim=64):
        super().__init__()
        
        # Step A: 1D-CNNs for Action and Pupil
        self.act_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim)
        )
        
        self.pup_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim)
        )
        
        # Step B: 2D-CNN for EEG electrodes
        self.eeg_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embed_dim)
        )
        
        # Step C: GCN Fusion
        # Adaptive Adjacency Matrix (22 nodes: 20 Brain, 1 Behavior, 1 Autonomic)
        self.adj = nn.Parameter(torch.rand(22, 22))
        
        # Graph Convolution Layers
        self.gcn1 = GraphConvLayer(embed_dim, gcn_dim)
        self.gcn2 = GraphConvLayer(gcn_dim, gcn_dim)
        
        # Prediction Head
        self.fc = nn.Sequential(
            nn.Linear(gcn_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, eeg, act, pup):
        B = eeg.size(0)
        
        # Encode Autonomic (Pupil) and Behavior (Action) Nodes
        node_act = self.act_cnn(act).unsqueeze(1) # (B, 1, embed_dim)
        node_pup = self.pup_cnn(pup).unsqueeze(1) # (B, 1, embed_dim)
        
        # Encode Brain Nodes (20 EEG electrodes)
        # eeg: (B, 20, 2, T, F)
        eeg_nodes = []
        for i in range(20):
            eeg_nodes.append(self.eeg_cnn(eeg[:, i]).unsqueeze(1)) # (B, 1, embed_dim)
            
        # Combine all 22 nodes: (B, 22, embed_dim)
        nodes = torch.cat([*eeg_nodes, node_act, node_pup], dim=1)
        
        # Normalize adaptive adjacency matrix
        adj_norm = torch.softmax(self.adj, dim=-1)
        
        # Message Passing
        x = torch.relu(self.gcn1(nodes, adj_norm))
        x = torch.relu(self.gcn2(x, adj_norm))
        
        # Pooling (Mean across all nodes to get a graph-level representation)
        graph_embed = x.mean(dim=1) # (B, gcn_dim)
        
        # Final Prediction
        return self.fc(graph_embed)

# ═══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════
import csv
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}...")

kf = KFold(n_splits=HPARAMS["kfold_splits"], shuffle=True, random_state=42)
all_metrics = []

# Setup CSV Logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
csv_filename = os.path.join(log_dir, f"{HPARAMS['kfold_splits']}_kfold_gcn_full_eeg_{timestamp}.csv")

with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_MSE", "Val_MSE", "Val_RMSE", "Val_MAE", "Val_R2", "Fold", "Total_Params", "Trainable_Params"])

best_overall_val_mse = float('inf')
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_eeg)):
    print(f"\n--- Fold {fold+1} ---")
    
    model = SensorGraphModel(embed_dim=HPARAMS["node_embed_dim"], gcn_dim=HPARAMS["gcn_hidden_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"], weight_decay=HPARAMS["weight_decay"])
    criterion = nn.MSELoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # DataLoaders for memory efficiency
    train_data = TensorDataset(
        torch.tensor(X_eeg[train_idx], device=device),
        torch.tensor(X_act[train_idx], device=device),
        torch.tensor(X_pup[train_idx], device=device),
        torch.tensor(y[train_idx], device=device)
    )
    val_data = TensorDataset(
        torch.tensor(X_eeg[val_idx], device=device),
        torch.tensor(X_act[val_idx], device=device),
        torch.tensor(X_pup[val_idx], device=device),
        torch.tensor(y[val_idx], device=device)
    )
    train_loader = DataLoader(train_data, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=HPARAMS["batch_size"], shuffle=False)
    
    best_val_mse = float('inf')
    patience = 30
    patience_cnt = 0
    
    for epoch in range(HPARAMS["epochs"]):
        model.train()
        train_loss = 0.0
        for b_eeg, b_act, b_pup, b_y in train_loader:
            optimizer.zero_grad()
            preds = model(b_eeg, b_act, b_pup)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_y.size(0)
        train_loss /= len(train_idx)
        
        model.eval()
        val_loss = 0.0
        v_preds_list, v_y_list = [], []
        with torch.no_grad():
            for b_eeg, b_act, b_pup, b_y in val_loader:
                v_preds = model(b_eeg, b_act, b_pup)
                loss = criterion(v_preds, b_y)
                val_loss += loss.item() * b_y.size(0)
                v_preds_list.append(v_preds.cpu().numpy())
                v_y_list.append(b_y.cpu().numpy())
        val_loss /= len(val_idx)
        
        v_preds_all = np.concatenate(v_preds_list)
        v_y_all = np.concatenate(v_y_list)
        
        v_rmse = float(np.sqrt(val_loss))
        v_mae = float(mean_absolute_error(v_y_all, v_preds_all))
        v_r2 = float(r2_score(v_y_all, v_preds_all))
        
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, v_rmse, v_mae, v_r2, fold + 1, total_params, trainable_params])
        
        if val_loss < best_val_mse - 0.001:
            best_val_mse = val_loss
            patience_cnt = 0
            best_mae = v_mae
            best_r2 = v_r2
            if val_loss < best_overall_val_mse:
                best_overall_val_mse = val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
            
        if patience_cnt >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
            
    print(f"  Best Val MSE: {best_val_mse:.4f} | MAE: {best_mae:.4f} | R2: {best_r2:.4f}")
    all_metrics.append({'mse': best_val_mse, 'mae': best_mae, 'r2': best_r2})

print(f"\nTraining logs saved to: {csv_filename}")

if best_model_state is not None:
    model_filename = csv_filename.replace('.csv', '.pth')
    torch.save(best_model_state, model_filename)
    print(f"Best model weights saved to: {model_filename}")

print("\n" + "="*40)
print("FINAL CROSS-VALIDATION RESULTS")
print("="*40)
mses = [m['mse'] for m in all_metrics]
maes = [m['mae'] for m in all_metrics]
r2s  = [m['r2'] for m in all_metrics]
print(f"MSE:  {np.mean(mses):.4f} ± {np.std(mses):.4f}")
print(f"MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
print(f"R2:   {np.mean(r2s):.4f}  ± {np.std(r2s):.4f}")
print("="*40)