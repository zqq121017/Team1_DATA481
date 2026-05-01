import pandas as pd
import numpy as np
import pickle
import os
import gc
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import stft as scipy_stft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

# ═══════════════════════════════════════════════════════════════════════
#  EEG CONFIGURATION 
# ═══════════════════════════════════════════════════════════════════════
EEG_CH_INDICES = [12, 18, 16]          # Fz, Cz, POz (based on 20-ch raw)
EEG_CH_NAMES   = ["Fz", "Cz", "POz"]
FREQ_BANDS     = {
    "delta": (0.5,  4.0),
    "alpha": (8.0, 13.0),
}
SFREQ          = 256          # EEG sampling frequency in Hz
NPERSEG        = 64           # STFT window length
NOVERLAP       = 56           # hop = 8 samples -> T ≈ ceil(384/8) = 48 time frames
# Delta 0.5–4 Hz  → bins at 2Hz(1), 4Hz(2)           → 2 bins
# Alpha 8–13 Hz   → bins at 8Hz(4),10Hz(5),12Hz(6)   → 3 bins
# Total F = 5 freq bins kept

# ═══════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS & CONFIG
# ═══════════════════════════════════════════════════════════════════════
HPARAMS = {
    "diff_map":            {"Easy": 1, "Medium": 2, "Hard": 3},
    "batch_size":         32,
    "epochs":             300,
    "learning_rate":       0.001,
    "weight_decay":        1e-3,
    "dropout":            0.2,
    "hidden_layers":      [128, 64, 32],
    "kfold_splits":       5,
    "early_stop_patience": 30,
    "early_stop_delta":    0.001,
}

# ═══════════════════════════════════════════════════════════════════════
#  EEG → TIME-FREQUENCY POWER TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════
def eeg_to_tfpower_uniform(raw_20ch_384):
    """
    Extracts 3 channels, performs STFT, computes power, 
    filters to Delta+Alpha, and pads bands to uniform F width.
    """
    maps = []
    for ch_idx in EEG_CH_INDICES:
        signal = raw_20ch_384[ch_idx].astype(np.float64)
        freqs, _, Zxx = scipy_stft(
            signal, fs=SFREQ, nperseg=NPERSEG, noverlap=NOVERLAP,
            boundary="zeros", padded=True,
        )
        power = np.abs(Zxx) ** 2                           # (freq_full, T)
        for lo, hi in FREQ_BANDS.values():
            mask = (freqs >= lo) & (freqs <= hi)
            band_power = power[mask, :].T.astype(np.float32)  # (T, F_band)
            maps.append(band_power)

    # Pad to uniform F (max width is 3 for Alpha)
    f_max = max(m.shape[1] for m in maps)
    padded = []
    for m in maps:
        pad_w = f_max - m.shape[1]
        padded.append(np.pad(m, ((0, 0), (0, pad_w))) if pad_w > 0 else m)

    # Return shape (6, T, F_max)
    return np.stack(padded, axis=0).astype(np.float32)

def pilot_to_tfpower(series):
    return np.stack([eeg_to_tfpower_uniform(arr) for arr in series.values])

# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING & MERGING
# ═══════════════════════════════════════════════════════════════════════
base_path  = "../../../data"
merge_keys = ['teamID', 'sessionID', 'trialID', 'ringID']

def load_pkl(filename):
    with open(os.path.join(base_path, filename), 'rb') as f:
        return pickle.load(f)

print("Loading and merging data...")
mod_files = [
    'epoched_eeg.pkl',
    'epoched_pupil.pkl',
    'epoched_speech_event.pkl',
    'epoched_action.pkl',
    'epoched_raw_location.pkl',  # Contains the 'time' column
]
df_main = load_pkl('team_performance.pkl')[merge_keys + ['difficulty']]

for mod_file in mod_files:
    print(f"  Merging {mod_file}...")
    mod_df = load_pkl(mod_file)
    # Drop known non-feature or redundant metadata columns
    to_drop = ['difficulty', 'communication', 'location', 'ringX', 'ringY', 'ringZ', 'startTime', 'ringTime']
    cols_to_drop = [c for c in to_drop if c in mod_df.columns]
    if cols_to_drop:
        mod_df = mod_df.drop(columns=cols_to_drop)
    df_main = df_main.merge(mod_df, on=merge_keys, how='inner')
    del mod_df
    gc.collect()

df_main['ring_score'] = df_main['difficulty'].map(HPARAMS["diff_map"])
df_main = df_main.dropna(subset=['ring_score']).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════
print("\nTransforming EEG to Time-Frequency power maps...")
eeg_yaw   = pilot_to_tfpower(df_main['yawEEG'])    # (N, 6, T, F)
eeg_pitch = pilot_to_tfpower(df_main['pitchEEG'])  # (N, 6, T, F)
eeg_thrust= pilot_to_tfpower(df_main['thrustEEG']) # (N, 6, T, F)

# Average the three pilots as Kevin does
eeg_combined = (eeg_yaw + eeg_pitch + eeg_thrust) / 3.0
df_main['eeg_data'] = [eeg_combined[i] for i in range(len(eeg_combined))]
del eeg_yaw, eeg_pitch, eeg_thrust, eeg_combined
gc.collect()

# For other bio-signals, average pilots then extract mean/std (Aryan-style)
def pilot_average_bio(row, prefix):
    yaw = np.array(row[f'yaw{prefix}'], dtype=float)
    pitch = np.array(row[f'pitch{prefix}'], dtype=float)
    thrust = np.array(row[f'thrust{prefix}'], dtype=float)
    combined = (yaw + pitch + thrust) / 3.0
    return combined

print("Processing Bio signals (Action, Pupil, Speech)...")
for bio in ['Action', 'Pupil', 'Speech']:
    df_main[f'avg{bio}'] = df_main.apply(lambda r: pilot_average_bio(r, bio), axis=1)

# ═══════════════════════════════════════════════════════════════════════
#  TRIAL-LEVEL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════
print("Aggregating to trial level...")
trial_keys = ['teamID', 'sessionID', 'trialID']

def aggregate_eeg(series):
    # Mean across rings in trial
    return np.mean(list(series), axis=0)

def aggregate_bio_stats(series):
    # Concatenate all rings in trial to get full trial stats
    full_seq = np.concatenate(list(series))
    return pd.Series({
        'mean': np.nanmean(full_seq),
        'std':  np.nanstd(full_seq)
    })

trial_df = df_main.groupby(trial_keys).agg({
    'ring_score': 'sum',
    'eeg_data':   aggregate_eeg,
}).reset_index()

# Add bio stats separately to avoid complex agg dict
for bio in ['Action', 'Pupil', 'Speech']:
    stats = df_main.groupby(trial_keys)[f'avg{bio}'].apply(aggregate_bio_stats).unstack()
    stats.columns = [f'{bio}_mean', f'{bio}_std']
    trial_df = trial_df.merge(stats, on=trial_keys)

# Add Time statistics (shared across pilots for each ring)
print("Processing Time statistics...")
time_stats = df_main.groupby(trial_keys)['time'].apply(aggregate_bio_stats).unstack()
time_stats.columns = ['Time_mean', 'Time_std']
trial_df = trial_df.merge(time_stats, on=trial_keys)

trial_df.rename(columns={'ring_score': 'trial_total_score'}, inplace=True)
print(f"  Total trials: {len(trial_df)}")

# ═══════════════════════════════════════════════════════════════════════
#  PREPARE MLP FEATURES (Flatten EEG)
# ═══════════════════════════════════════════════════════════════════════
def prepare_mlp_input(row):
    # Flatten (6, T, F) EEG into a long vector
    eeg_flat = row['eeg_data'].flatten()
    # Add bio and time stats
    # NOTE: difficulty_score is EXCLUDED to prevent metadata leakage
    other_feats = [
        row['Action_mean'], row['Action_std'],
        row['Pupil_mean'],  row['Pupil_std'],
        row['Speech_mean'], row['Speech_std'],
        row['Time_mean'],   row['Time_std']
    ]
    return np.concatenate([eeg_flat, other_feats])

print("Flattening features for MLP...")
X = np.stack(trial_df.apply(prepare_mlp_input, axis=1).values).astype(np.float32)
y = trial_df['trial_total_score'].values.astype(np.float32).reshape(-1, 1)

# ═══════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════
class BasicMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        layers = []
        curr_size = input_size
        for h_size in HPARAMS["hidden_layers"]:
            layers.append(nn.Linear(curr_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(HPARAMS["dropout"]))
            curr_size = h_size
        layers.append(nn.Linear(curr_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ═══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP (5-Fold CV)
# ═══════════════════════════════════════════════════════════════════════
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}...")

kf = KFold(n_splits=HPARAMS["kfold_splits"], shuffle=True, random_state=123)
all_metrics = []

# Setup CSV Logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
csv_filename = os.path.join(log_dir, f"{HPARAMS['kfold_splits']}_kfold_mlp_eeg_refined_{timestamp}.csv")

with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_MSE", "Val_MSE", "Val_RMSE", "Val_MAE", "Val_R2", "Fold", "Total_Params", "Trainable_Params"])

best_overall_val_mse = float('inf')
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1} ---")
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val   = scaler.transform(X[val_idx])
    
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Tensors
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_val_t   = torch.tensor(X_val,   device=device)
    y_val_t   = torch.tensor(y_val,   device=device)
    
    model = BasicMLP(X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"], weight_decay=HPARAMS["weight_decay"])
    criterion = nn.MSELoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_val_mse = float('inf')
    fold_train_history = []
    fold_val_history = []
    
    for epoch in range(HPARAMS["epochs"]):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        tr_loss = criterion(preds, y_train_t)
        tr_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_preds = model(X_val_t)
            v_mse = criterion(v_preds, y_val_t).item()
        
        fold_train_history.append(tr_loss.item())
        fold_val_history.append(v_mse)
        
        v_preds_np = v_preds.cpu().numpy()
        v_rmse = float(np.sqrt(v_mse))
        v_mae = float(mean_absolute_error(y_val, v_preds_np))
        v_r2 = float(r2_score(y_val, v_preds_np))
        
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, tr_loss.item(), v_mse, v_rmse, v_mae, v_r2, fold + 1, total_params, trainable_params])

        if v_mse < best_val_mse - 0.001:
            best_val_mse = v_mse
            # Snapshot metrics
            best_mae = v_mae
            best_r2 = v_r2
            best_tr_mse = tr_loss.item()
            if v_mse < best_overall_val_mse:
                best_overall_val_mse = v_mse
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:03d} | Train MSE: {tr_loss.item():.4f} | Val MSE: {v_mse:.4f}")
            
    print(f"  Best Val MSE: {best_val_mse:.4f} | MAE: {best_mae:.4f} | R2: {best_r2:.4f}")
    all_metrics.append({
        'mse': best_val_mse, 
        'mae': best_mae, 
        'r2': best_r2, 
        'train_mse': best_tr_mse,
        'train_hist': fold_train_history,
        'val_hist': fold_val_history
    })

print(f"\nTraining logs saved to: {csv_filename}")

if best_model_state is not None:
    model_filename = csv_filename.replace('.csv', '.pth')
    torch.save(best_model_state, model_filename)
    print(f"Best model weights saved to: {model_filename}")

# ═══════════════════════════════════════════════════════════════════════
#  OVERFITTING ANALYSIS & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

print("\n" + "="*40)
print("OVERFITTING ANALYSIS")
print("="*40)

train_mses = [m['train_mse'] for m in all_metrics]
val_mses   = [m['mse'] for m in all_metrics]

mean_train = np.mean(train_mses)
mean_val   = np.mean(val_mses)
gap        = (mean_val - mean_train) / mean_train * 100

print(f"Mean Train MSE: {mean_train:.4f}")
print(f"Mean Val MSE:   {mean_val:.4f}")
print(f"Performance Gap: {gap:.2f}% (Val is {gap:.2f}% higher than Train)")

if gap > 20:
    print("STATUS: Significant Overfitting detected. Consider increasing dropout or weight decay.")
elif gap > 10:
    print("STATUS: Moderate Overfitting. The model is generalizing reasonably but could be improved.")
else:
    print("STATUS: Low Overfitting. The model generalizes well.")

# Plotting Loss Curves
plt.figure(figsize=(10, 6))
for i, m in enumerate(all_metrics):
    plt.plot(m['train_hist'], label=f'Fold {i+1} Train', linestyle='--')
    plt.plot(m['val_hist'], label=f'Fold {i+1} Val')

plt.title("MLP Training vs Validation MSE (All Folds)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = "overfitting_analysis.png"
plt.savefig(plot_path)
print(f"\nLoss curve plot saved to: {plot_path}")

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
