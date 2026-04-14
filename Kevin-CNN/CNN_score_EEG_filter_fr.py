import pandas as pd
import numpy as np
import pickle
import os
import gc
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.optim as optim
from sklearn.model_selection import KFold
import csv
import datetime

# ═══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

_USER = {
    "depth":            3,
    "filters":          64,
    "kernel_size":      10,
    "padding":          1,
    "learning_rate":    0.0005,
    "diff_map":         {"Easy": 1, "Medium": 2, "Hard": 3},
    "top_k_channels":   20,   # how many channels to select in phase 2
}

_FIXED = {
    "model_name":        "ChannelFilterModel_TrialScore",
    "dropout":           0.4,
    "fc_hidden":         128,
    "optimizer":         "Adam",
    "loss_fn":           "MSELoss",
    "batch_size":        32,
    "epochs":            100,
    "kfold_splits":      5,
    "kfold_shuffle":     True,
    "kfold_random_state":42,
    # Input shapes
    "eeg_freq_bins":     384 // 2 + 1,   # 193
    "n_eeg_channels":    60,
    "bio_in_channels":   3,
    "bio_timesteps":     90,
}

def _conv_out(size, k, p, s=1):
    return math.floor((size + 2 * p - k) / s + 1)

def _pool_out(size, k=2):
    return size // k

def _compute_flat_sizes(user, fixed):
    k, p, d = user["kernel_size"], user["padding"], user["depth"]
    f = user["filters"]

    # Compute embed_dim for shared EEG encoder (1D over freq bins)
    t_eeg = fixed["eeg_freq_bins"]
    for _ in range(d):
        t_eeg = _pool_out(_conv_out(t_eeg, k, p))
    embed_dim = f * t_eeg

    # Bio: 1-D conv (temporal only)
    t_bio = fixed["bio_timesteps"]
    for _ in range(d):
        t_bio = _pool_out(_conv_out(t_bio, k, p))
    bio_flat = f * t_bio

    fusion = embed_dim + bio_flat * 3
    return embed_dim, bio_flat, fusion

_embed_dim, _bio_flat, _fusion = _compute_flat_sizes(_USER, _FIXED)

HPARAMS = {
    **_USER,
    **_FIXED,
    "embed_dim":    _embed_dim,
    "bio_flat":     _bio_flat,
    "fusion_input": _fusion,
}

os.makedirs("training_logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
phase1_log = f"training_logs/phase1_5fold_EEG_filter_fr_{timestamp}.csv"
phase2_log = f"training_logs/phase2_topk_EEG_filter_fr_{timestamp}.csv"

def _write_csv_header(path, hparams, phase=1, selected_indices=None):
    with open(path, mode='w', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow([f"# {hparams['model_name']} — Phase {phase} Configuration"])
        w.writerow(["# Model uses Channel-Wise Attention Pooling for EEG interpretability"])
        w.writerow(["# Two-phase procedure: Phase 1 (All 60 channels), Phase 2 (Top-K channels)"])
        w.writerow([])
        if phase == 2 and selected_indices is not None:
            w.writerow(["# SELECTED TOP-K CHANNELS"])
            w.writerow(["# Channel_indices", str(list(selected_indices))])
            w.writerow([])
        
        w.writerow(["# USER-EDITABLE HYPERPARAMETERS"])
        for key in ("depth", "filters", "kernel_size", "padding", "learning_rate", "top_k_channels"):
            w.writerow(["# " + key, hparams[key]])
        w.writerow(["# diff_map", str(hparams["diff_map"])])
        w.writerow([])
        
        w.writerow(["# FIXED CONSTANTS"])
        for key in ("dropout", "fc_hidden", "optimizer", "loss_fn", "batch_size", "epochs", 
                    "kfold_splits", "eeg_freq_bins", "n_eeg_channels", "bio_in_channels", "bio_timesteps"):
            w.writerow(["# " + key, hparams[key]])
        w.writerow([])
        
        w.writerow(["# AUTO-COMPUTED SIZES"])
        w.writerow(["# embed_dim",    hparams["embed_dim"]])
        w.writerow(["# bio_flat",     hparams["bio_flat"]])
        w.writerow(["# fusion_input", hparams["fusion_input"]])
        w.writerow([])
        
        w.writerow(["# EEG BRANCH — Channel-Wise Attention Pooling"])
        w.writerow([f"# Input: ({hparams['n_eeg_channels'] if phase==1 else hparams['top_k_channels']}, {hparams['eeg_freq_bins']})"])
        w.writerow(["# Shared encoder (applied to each channel independently):"])
        w.writerow([f"#   [Conv1d → ReLU → MaxPool1d] × {hparams['depth']} → Flatten → {hparams['embed_dim']}"])
        w.writerow(["# Channel attention:"])
        w.writerow([f"#   Linear({hparams['embed_dim']} → 1) + Softmax(dim=channels)"])
        w.writerow([f"#   Weighted sum over channels → ({hparams['embed_dim']},)"])
        w.writerow([])
        
        w.writerow(["Epoch", "Train_MSE", "Val_MSE", "Val_RMSE", "Val_MAE", "Val_R2", "Fold", "Total_Params", "Trainable_Params"])

# ═══════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

class ChannelEncoder(nn.Module):
    """Shared encoder applied to each EEG channel independently."""
    def __init__(self, hparams):
        super().__init__()
        d = hparams["depth"]
        f = hparams["filters"]
        k = hparams["kernel_size"]
        p = hparams["padding"]
        
        layers = []
        in_ch = 1
        for _ in range(d):
            layers += [nn.Conv1d(in_ch, f, k, padding=p), nn.ReLU(), nn.MaxPool1d(2)]
            in_ch = f
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ChannelAttention(nn.Module):
    """Learns attention weights for each channel and performs weighted sum."""
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_linear = nn.Linear(embed_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ch_embeds):
        # ch_embeds: (batch, n_channels, embed_dim)
        weights = self.attn_linear(ch_embeds)      # (batch, n_channels, 1)
        weights = self.softmax(weights)            # (batch, n_channels, 1)
        attended = (weights * ch_embeds).sum(dim=1) # (batch, embed_dim)
        return attended, weights.squeeze(-1)       # (batch, embed_dim), (batch, n_channels)

class ChannelFilterModel(nn.Module):
    def __init__(self, hparams, n_channels=60):
        super().__init__()
        self.n_channels = n_channels
        self.channel_encoder = ChannelEncoder(hparams)
        self.channel_attn = ChannelAttention(hparams["embed_dim"])
        
        def _bio_branch():
            layers, in_ch = [], hparams["bio_in_channels"]
            for _ in range(hparams["depth"]):
                layers += [nn.Conv1d(in_ch, hparams["filters"], hparams["kernel_size"], padding=hparams["padding"]), 
                           nn.ReLU(), nn.MaxPool1d(2)]
                in_ch = hparams["filters"]
            layers.append(nn.Flatten())
            return nn.Sequential(*layers)

        self.act_branch = _bio_branch()
        self.pup_branch = _bio_branch()
        self.spc_branch = _bio_branch()

        self.fc = nn.Sequential(
            nn.Linear(hparams["fusion_input"], hparams["fc_hidden"]),
            nn.ReLU(),
            nn.Dropout(hparams["dropout"]),
            nn.Linear(hparams["fc_hidden"], 1),
        )

    def forward(self, eeg, act, pup, spc):
        # eeg: (batch, n_channels, 193)
        batch = eeg.size(0)
        ch_embeds = []
        for i in range(self.n_channels):
            ch = eeg[:, i:i+1, :]                           # (batch, 1, 193)
            ch_embeds.append(self.channel_encoder(ch))      # (batch, embed_dim)
        ch_embeds = torch.stack(ch_embeds, dim=1)           # (batch, n_channels, embed_dim)
        
        eeg_out, attn_weights = self.channel_attn(ch_embeds) # (batch, embed_dim), (batch, n_channels)
        
        b2 = self.act_branch(act)
        b3 = self.pup_branch(pup)
        b4 = self.spc_branch(spc)
        
        fused = torch.cat([eeg_out, b2, b3, b4], dim=1)
        return self.fc(fused), attn_weights

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING (Reused from CNN_score_pre_fr.py)
# ═══════════════════════════════════════════════════════════════════════

base_path  = "../../data"
merge_keys = ['teamID', 'sessionID', 'trialID', 'ringID']

def load_pkl(filename):
    with open(os.path.join(base_path, filename), 'rb') as f:
        return pickle.load(f)

print("Loading and merging data...")
mod_files = ['epoched_eeg.pkl', 'epoched_pupil.pkl', 'epoched_speech_event.pkl', 'epoched_action.pkl']
df_main = load_pkl('team_performance.pkl')[merge_keys + ['difficulty']]

for mod_file in mod_files:
    mod_df = load_pkl(mod_file)
    cols_to_drop = [c for c in ['difficulty', 'communication'] if c in mod_df.columns]
    if cols_to_drop: mod_df = mod_df.drop(columns=cols_to_drop)
    df_main = df_main.merge(mod_df, on=merge_keys, how='inner')
    del mod_df
    gc.collect()

df_main['ring_score'] = df_main['difficulty'].map(HPARAMS["diff_map"])
df_main = df_main.dropna(subset=['ring_score']).reset_index(drop=True)

def to_freq(arr):
    return np.abs(np.fft.rfft(arr, axis=-1)).astype(np.float32)

print("Processing EEG to frequency domain...")
eeg_raw = np.stack([
    to_freq(np.stack(df_main['yawEEG'].values)),
    to_freq(np.stack(df_main['pitchEEG'].values)),
    to_freq(np.stack(df_main['thrustEEG'].values)),
], axis=1).reshape(len(df_main), HPARAMS["n_eeg_channels"], HPARAMS["eeg_freq_bins"]).astype(np.float32)

def process_branch(df, cols):
    data = np.stack([np.stack(df[c].values) for c in cols], axis=1)
    if data.ndim == 2: data = data.reshape(len(df), 1, -1)
    return data.astype(np.float32)

act_raw = process_branch(df_main, ['yawAction', 'pitchAction', 'thrustAction'])
pup_raw = process_branch(df_main, ['yawPupil',  'pitchPupil',  'thrustPupil'])
spc_raw = process_branch(df_main, ['yawSpeech', 'pitchSpeech', 'thrustSpeech'])

df_main['eeg_data'] = [eeg_raw[i] for i in range(len(df_main))]
df_main['act_data'] = [act_raw[i] for i in range(len(df_main))]
df_main['pup_data'] = [pup_raw[i] for i in range(len(df_main))]
df_main['spc_data'] = [spc_raw[i] for i in range(len(df_main))]

def aggregate_arrays(series): return np.mean(list(series), axis=0)
trial_keys = ['teamID', 'sessionID', 'trialID']
trial_aggregated = df_main.groupby(trial_keys).agg({
    'ring_score': 'sum', 'eeg_data': aggregate_arrays, 'act_data': aggregate_arrays,
    'pup_data': aggregate_arrays, 'spc_data': aggregate_arrays,
}).reset_index()
trial_aggregated.rename(columns={'ring_score': 'trial_total_score'}, inplace=True)

eeg_trial = np.stack(trial_aggregated['eeg_data'].values).astype(np.float32)
act_trial = np.stack(trial_aggregated['act_data'].values).astype(np.float32)
pup_trial = np.stack(trial_aggregated['pup_data'].values).astype(np.float32)
spc_trial = np.stack(trial_aggregated['spc_data'].values).astype(np.float32)
y_trial_scores = trial_aggregated['trial_total_score'].values.astype(np.float32)

def normalize_3d(data):
    s, c, t = data.shape
    scaler = StandardScaler()
    reshaped = data.transpose(0, 2, 1).reshape(-1, c)
    normed = scaler.fit_transform(reshaped)
    return normed.reshape(s, t, c).transpose(0, 2, 1).astype(np.float32)

eeg_norm = normalize_3d(eeg_trial)
pup_norm = normalize_3d(pup_trial)

# ═══════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def run_kfold(dataset, hparams, log_path, phase=1, n_channels=60):
    kf = KFold(n_splits=hparams["kfold_splits"], shuffle=hparams["kfold_shuffle"], random_state=hparams["kfold_random_state"])
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    fold_results = []
    all_fold_attn_weights = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\nPhase {phase} - Fold {fold+1}/{hparams['kfold_splits']}")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=hparams["batch_size"], shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=hparams["batch_size"], shuffle=False)
        
        model = ChannelFilterModel(hparams, n_channels=n_channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])
        criterion = nn.MSELoss()
        total_p, trainable_p = count_params(model)
        
        last_val_metrics = {}
        fold_attn_weights = []

        for epoch in range(hparams["epochs"]):
            model.train()
            tr_preds, tr_targets = [], []
            for b_eeg, b_act, b_pup, b_spc, b_y in train_loader:
                inputs = [x.to(device) for x in [b_eeg, b_act, b_pup, b_spc]]
                target = b_y.to(device).view(-1, 1)
                optimizer.zero_grad()
                out, _ = model(*inputs)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                tr_preds.append(out.detach().cpu().numpy())
                tr_targets.append(target.detach().cpu().numpy())
            
            tr_mse = float(np.mean((np.concatenate(tr_preds) - np.concatenate(tr_targets))**2))
            
            model.eval()
            val_preds, val_targets, ep_attn = [], [], []
            with torch.no_grad():
                for b_eeg, b_act, b_pup, b_spc, b_y in val_loader:
                    inputs = [x.to(device) for x in [b_eeg, b_act, b_pup, b_spc]]
                    out, attn = model(*inputs)
                    val_preds.append(out.cpu().numpy())
                    val_targets.append(b_y.numpy())
                    ep_attn.append(attn.cpu().numpy())
            
            val_preds = np.concatenate(val_preds).flatten()
            val_targets = np.concatenate(val_targets).flatten()
            v_mse = float(np.mean((val_preds - val_targets)**2))
            v_rmse, v_mae, v_r2 = np.sqrt(v_mse), mean_absolute_error(val_targets, val_preds), r2_score(val_targets, val_preds)
            
            with open(log_path, mode='a', newline='') as f_out:
                csv.writer(f_out).writerow([epoch+1, f"{tr_mse:.4f}", f"{v_mse:.4f}", f"{v_rmse:.4f}", f"{v_mae:.4f}", f"{v_r2:.4f}", fold+1, total_p, trainable_p])
            
            last_val_metrics = {"mse": v_mse, "rmse": v_rmse, "mae": v_mae, "r2": v_r2}
            if epoch == hparams["epochs"] - 1:
                fold_attn_weights = np.concatenate(ep_attn, axis=0) # (N_val, n_channels)

        fold_results.append(last_val_metrics)
        all_fold_attn_weights.append(np.mean(fold_attn_weights, axis=0)) # (n_channels,)
        print(f"  Final Val MSE: {last_val_metrics['mse']:.4f} | R2: {last_val_metrics['r2']:.4f}")

        # Log per-fold channel importance
        if phase == 1:
            with open(log_path, mode='a', newline='') as f_out:
                w = csv.writer(f_out)
                w.writerow([])
                w.writerow([f"# Channel importance scores — Fold {fold+1}"])
                w.writerow(["# Channel_idx", "Mean_attention_weight"])
                for idx, weight in enumerate(all_fold_attn_weights[-1]):
                    w.writerow(["# " + str(idx), f"{weight:.6f}"])
                w.writerow([])

    return fold_results, np.array(all_fold_attn_weights)

# ═══════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════

# Phase 1: Full Model
_write_csv_header(phase1_log, HPARAMS, phase=1)
full_ds = TensorDataset(torch.tensor(eeg_norm), torch.tensor(act_trial), torch.tensor(pup_norm), torch.tensor(spc_trial), torch.tensor(y_trial_scores))
print("\nStarting Phase 1 (All 60 channels)...")
p1_results, p1_attn_folds = run_kfold(full_ds, HPARAMS, phase1_log, phase=1, n_channels=60)

# Calculate Overall Importance
mean_importance = np.mean(p1_attn_folds, axis=0)
np.save("training_logs/channel_importance_{timestamp}.npy", mean_importance)
ranks = np.argsort(np.argsort(-mean_importance))
top_k_indices = np.argsort(-mean_importance)[:HPARAMS["top_k_channels"]]
np.save("training_logs/top_k_indices_{timestamp}.npy", top_k_indices)

# Log Overall Importance to Phase 1 CSV
with open(phase1_log, mode='a', newline='') as f_out:
    w = csv.writer(f_out)
    w.writerow(["# Overall channel importance (mean across folds)"])
    w.writerow(["# Channel_idx", "Mean_attention_weight", "Rank"])
    for i in range(60):
        w.writerow(["# " + str(i), f"{mean_importance[i]:.6f}", ranks[i]])

# Plotting
plt.figure(figsize=(12, 6))
colors = ['coral' if i in top_k_indices else 'lightgray' for i in range(60)]
plt.bar(range(60), mean_importance, color=colors)
plt.axhline(y=1/60, color='blue', linestyle='--', alpha=0.5, label='Uniform Baseline (1/60)')
plt.xlabel("Channel Index")
plt.ylabel("Mean Attention Weight")
plt.title(f"EEG Channel Importance (Top {HPARAMS['top_k_channels']} Highlighted)")
plt.legend()
plt.tight_layout()
plt.savefig("training_logs/channel_importance_plot.png")
print(f"Channel importance plot saved to training_logs/channel_importance_plot.png")

# Phase 2: Top-K Channels
print(f"\nStarting Phase 2 (Top {HPARAMS['top_k_channels']} channels)...")
_write_csv_header(phase2_log, HPARAMS, phase=2, selected_indices=top_k_indices)
eeg_topk = eeg_norm[:, top_k_indices, :]
topk_ds = TensorDataset(torch.tensor(eeg_topk), torch.tensor(act_trial), torch.tensor(pup_norm), torch.tensor(spc_trial), torch.tensor(y_trial_scores))
p2_results, _ = run_kfold(topk_ds, HPARAMS, phase2_log, phase=2, n_channels=HPARAMS["top_k_channels"])

print("\n" + "="*50)
print("  TRAINING COMPLETE")
print("="*50)
print(f"Phase 1 Log: {phase1_log}")
print(f"Phase 2 Log: {phase2_log}")
