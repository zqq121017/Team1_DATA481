import pandas as pd
import numpy as np
import pickle
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import csv
import datetime

# ═══════════════════════════════════════════════════════════════════════
#  EEG & GRAPH CONFIGURATION 
# ═══════════════════════════════════════════════════════════════════════
EEG_CH_INDICES = [12, 18, 16]          # Fz, Cz, POz

HPARAMS = {
    "diff_map":            {"Easy": 1, "Medium": 2, "Hard": 3},
    "batch_size":         1,      # GCN uses full-population forward pass
    "epochs":             1200,
    "learning_rate":       0.001,
    "weight_decay":        1e-3,
    "dropout":            0.25,
    "hidden_size":        256,
    "k_neighbors":        12,
    "kfold_splits":       5,
}

def _fft_power_bins_1d(x: np.ndarray, fs: float, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    x = np.nan_to_num(x)
    out = np.zeros(n_bins, dtype=float)
    if x.size < 64: return out
    n_fft = int(min(1024, x.size))
    n_fft = max(64, n_fft)
    power = np.abs(np.fft.rfft(x, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    nyq = fs / 2.0
    bin_edges = np.linspace(0.0, nyq, n_bins + 1)
    for k in range(n_bins):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        mask = (freqs >= lo) & (freqs < hi) if k < n_bins - 1 else (freqs >= lo) & (freqs <= hi)
        out[k] = np.mean(power[mask]) if np.any(mask) else 0.0
    return np.nan_to_num(out)

def _pupil_fft_bins_under_1hz(x, fs, f_max, n_bins):
    x = np.asarray(x, dtype=float).ravel()
    x = np.nan_to_num(x)
    out = np.zeros(n_bins, dtype=float)
    if x.size < 32: return out
    n_fft = int(min(1024, x.size))
    n_fft = max(64, n_fft)
    power = np.abs(np.fft.rfft(x, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    bin_edges = np.linspace(0.0, f_max, n_bins + 1)
    for k in range(n_bins):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        mask = (freqs >= lo) & (freqs < hi) if k < n_bins - 1 else (freqs >= lo) & (freqs <= hi)
        out[k] = np.mean(power[mask]) if np.any(mask) else 0.0
    return np.nan_to_num(out)

def extract_concat_bio_blocks(yaw_series, pitch_series, thrust_series):
    blocks = []
    for y, p, t in zip(yaw_series, pitch_series, thrust_series):
        y_arr, p_arr, t_arr = np.array(y), np.array(p), np.array(t)
        if y_arr.ndim == 1: y_arr = y_arr.reshape(1, -1)
        if p_arr.ndim == 1: p_arr = p_arr.reshape(1, -1)
        if t_arr.ndim == 1: t_arr = t_arr.reshape(1, -1)
        combined = np.concatenate([y_arr, p_arr, t_arr], axis=0)
        blocks.append(combined)
    return np.concatenate(blocks, axis=1)

# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING & MERGING
# ═══════════════════════════════════════════════════════════════════════
base_path  = "../../../data"
merge_keys = ['teamID', 'sessionID', 'trialID', 'ringID']

def load_pkl(filename):
    with open(os.path.join(base_path, filename), 'rb') as f:
        return pickle.load(f)

print("Loading and merging data...")
mod_files = ['epoched_eeg.pkl', 'epoched_pupil.pkl', 'epoched_speech_event.pkl', 'epoched_action.pkl']
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
#  FEATURE EXTRACTION & SEQUENCE MERGING
# ═══════════════════════════════════════════════════════════════════════
print("\nProcessing trial sequences (Full spectrum EEG, Pupil, Action, Speech)...")

def get_trial_merged_data(df):
    trial_keys = ['teamID', 'sessionID', 'trialID']
    X1_list, X2_list, y_list = [], [], []

    grouped = df.groupby(trial_keys, sort=False)
    for _, g in grouped:
        diff_score = float(HPARAMS["diff_map"].get(g.iloc[0]['difficulty'], 2))
        weighted_score = float(len(g)) * diff_score
        
        # 1. EEG
        merged_eeg = extract_concat_bio_blocks(g['yawEEG'], g['pitchEEG'], g['thrustEEG'])
        eeg_fft = []
        C_eeg = merged_eeg.shape[0] // 3
        for pilot_offset in [0, C_eeg, 2 * C_eeg]:
            for ch_idx in EEG_CH_INDICES:
                actual_idx = pilot_offset + ch_idx
                ch_data = merged_eeg[actual_idx] if actual_idx < merged_eeg.shape[0] else np.zeros(merged_eeg.shape[1])
                eeg_fft.append(_fft_power_bins_1d(ch_data, fs=256.0, n_bins=64))
        eeg_fft = np.stack(eeg_fft, axis=0) # (9, 64)
        
        # 2. Pupil
        merged_pupil = extract_concat_bio_blocks(g['yawPupil'], g['pitchPupil'], g['thrustPupil'])
        pupil_fft = []
        for ch_idx in range(merged_pupil.shape[0]):
            pupil_fft.append(_pupil_fft_bins_under_1hz(merged_pupil[ch_idx], fs=100.0, f_max=1.0, n_bins=64))
        pupil_fft = np.stack(pupil_fft, axis=0) # (C_pupil * 3, 64)
        
        # 3. Action and Speech
        action_seq = extract_concat_bio_blocks(g['yawAction'], g['pitchAction'], g['thrustAction'])
        speech_seq = extract_concat_bio_blocks(g['yawSpeech'], g['pitchSpeech'], g['thrustSpeech'])
        
        max_t = max(eeg_fft.shape[1], pupil_fft.shape[1], action_seq.shape[1], speech_seq.shape[1])
        
        eeg_pad = np.pad(eeg_fft, ((0,0), (0, max_t - eeg_fft.shape[1])))
        pupil_pad = np.pad(pupil_fft, ((0,0), (0, max_t - pupil_fft.shape[1])))
        action_pad = np.pad(action_seq, ((0,0), (0, max_t - action_seq.shape[1])))
        speech_pad = np.pad(speech_seq, ((0,0), (0, max_t - speech_seq.shape[1])))
        
        trial_x2 = np.concatenate([eeg_pad, pupil_pad, action_pad, speech_pad], axis=0).astype(np.float32)
        
        X1_list.append([diff_score])
        X2_list.append(trial_x2)
        y_list.append(weighted_score)
        
    return X1_list, X2_list, np.array(y_list, dtype=np.float32)

X1_raw, X2_raw, y = get_trial_merged_data(df_main)
max_t = max(x.shape[1] for x in X2_raw)
n_feat_channels = X2_raw[0].shape[0]
X2 = np.stack([np.pad(x, ((0,0), (0, max_t - x.shape[1]))) for x in X2_raw]).astype(np.float32)
X1 = np.array(X1_raw, dtype=np.float32)

print(f"Dataset prepared: X1={X1.shape}, X2={X2.shape}, y={y.shape}")

# ═══════════════════════════════════════════════════════════════════════
#  MODEL & GRAPH UTILS
# ═══════════════════════════════════════════════════════════════════════
def build_knn_adj(X1, X2, k=10):
    n = X1.shape[0]
    x1_norm = np.sum(X1**2, axis=1, keepdims=True)
    d1 = x1_norm + x1_norm.T - 2.0 * (X1 @ X1.T)
    x2_flat = X2.reshape(n, -1)
    x2_norm = np.sum(x2_flat**2, axis=1, keepdims=True)
    d2 = x2_norm + x2_norm.T - 2.0 * (x2_flat @ x2_flat.T)
    dists = np.sqrt(np.maximum(d1 + d2, 0.0))
    np.fill_diagonal(dists, np.inf)
    knn_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n): A[i, knn_idx[i]] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)
    deg = np.sum(A, axis=1)
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, a_hat): return self.linear(a_hat @ x)

class RefinedGCN(nn.Module):
    def __init__(self, x1_dim, x2_channels, hidden_size=256, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(x2_channels, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        fused_dim = x1_dim + 64
        self.gc1 = GraphConv(fused_dim, hidden_size)
        self.gc2 = GraphConv(hidden_size, hidden_size // 2)
        self.gc3 = GraphConv(hidden_size // 2, hidden_size // 4)
        self.out = nn.Linear(hidden_size // 4, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, a_hat):
        x2_latent = self.encoder(x2).squeeze(-1)
        x = torch.cat([x1, x2_latent], dim=1)
        x = torch.relu(self.gc1(x, a_hat))
        x = self.dropout(x)
        x = torch.relu(self.gc2(x, a_hat))
        x = self.dropout(x)
        x = torch.relu(self.gc3(x, a_hat))
        x = self.dropout(x)
        return self.out(x).squeeze(1)

# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kf = KFold(n_splits=HPARAMS["kfold_splits"], shuffle=True, random_state=42)
all_metrics = []

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"5_kfold_gcn_tfpower_{timestamp}.csv"
csv_dir = "/home/kevin/Team1_DATA481/Kevin-CNN/training_logs"
os.makedirs(csv_dir, exist_ok=True)
csv_filepath = os.path.join(csv_dir, csv_filename)

with open(csv_filepath, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_MSE", "Val_MSE", "Val_RMSE", "Val_MAE", "Val_R2", "Fold"])

print(f"Training GCN on {device}...")
for fold, (train_idx, test_idx) in enumerate(kf.split(X1)):
    print(f"\n--- Fold {fold+1} ---")
    s1 = StandardScaler()
    X1_train_scaled = s1.fit_transform(X1[train_idx])
    X1_all_scaled = s1.transform(X1)
    
    # Element-wise scaling across the training trials [channel, time_step]
    mu2 = X2[train_idx].mean(axis=0, keepdims=True)
    sd2 = X2[train_idx].std(axis=0, keepdims=True)
    X2_all_scaled = (X2 - mu2) / (sd2 + 1e-8)
    
    A_hat = build_knn_adj(X1_all_scaled, X2_all_scaled, k=HPARAMS["k_neighbors"])
    x1_t, x2_t, y_t, a_t = torch.tensor(X1_all_scaled, device=device), torch.tensor(X2_all_scaled, device=device), torch.tensor(y, device=device), torch.tensor(A_hat, device=device)
    
    model = RefinedGCN(X1.shape[1], n_feat_channels, HPARAMS["hidden_size"], HPARAMS["dropout"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"], weight_decay=HPARAMS["weight_decay"])
    criterion = nn.MSELoss()
    
    train_mask, test_mask = torch.zeros(len(y), dtype=torch.bool, device=device), torch.zeros(len(y), dtype=torch.bool, device=device)
    train_mask[train_idx], test_mask[test_idx] = True, True
    
    for epoch in range(HPARAMS["epochs"]):
        model.train(); optimizer.zero_grad()
        preds = model(x1_t, x2_t, a_t)
        loss = criterion(preds[train_mask], y_t[train_mask])
        loss.backward(); optimizer.step()
        
        model.eval()
        with torch.no_grad():
            v_preds = model(x1_t, x2_t, a_t)[test_mask]
            v_y = y_t[test_mask]
            v_mse = criterion(v_preds, v_y).item()
            
            v_preds_np = v_preds.cpu().numpy()
            v_y_np = v_y.cpu().numpy()
            v_rmse = float(np.sqrt(v_mse))
            v_mae = float(mean_absolute_error(v_y_np, v_preds_np))
            v_r2 = float(r2_score(v_y_np, v_preds_np))
            
        with open(csv_filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, loss.item(), v_mse, v_rmse, v_mae, v_r2, fold + 1])
            
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:04d} | Train MSE: {loss.item():.4f} | Val MSE: {v_mse:.4f}")

    model.eval()
    with torch.no_grad():
        final_preds = model(x1_t, x2_t, a_t)[test_mask].cpu().numpy()
        y_true = y[test_idx]
        metrics = {
            'mse': float(np.mean((final_preds - y_true)**2)), 
            'mae': float(mean_absolute_error(y_true, final_preds)), 
            'r2': float(r2_score(y_true, final_preds))
        }
        all_metrics.append(metrics)
        print(f"  -> Fold Result | MSE: {metrics['mse']:.4f} | R2: {metrics['r2']:.4f}")

print("\n" + "="*40 + "\nFINAL GCN RESULTS\n" + "="*40)
for m in ['mse', 'mae', 'r2']:
    vals = [x[m] for x in all_metrics]
    print(f"{m.upper():<4}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")