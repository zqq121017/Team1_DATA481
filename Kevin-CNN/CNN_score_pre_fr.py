import pandas as pd
import numpy as np
import pickle
import os
import gc
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.optim as optim
from sklearn.model_selection import KFold
import csv
import datetime

try:
    from generate_model_graph import generate_model_graph
    HAS_GRAPH_GEN = True
except ImportError:
    HAS_GRAPH_GEN = False
    print("Warning: generate_model_graph.py not found. Skipping graph generation.")

_USER = {
    "depth":        3,
    "filters":      64,
    "kernel_size":  10,
    "padding":      1,
    "learning_rate":0.0005,
    "diff_map": {"Easy": 1, "Medium": 2, "Hard": 3},
}

_FIXED = {
    "model_name":        "FullADCTModel_TrialScore",
    "dropout":           0.4,
    "fc_hidden":         128,
    "optimizer":         "Adam",
    "loss_fn":           "MSELoss",
    "batch_size":        32,
    "epochs":            100,
    "kfold_splits":      5,
    "kfold_shuffle":     True,
    "kfold_random_state":42,
    # Input shapes (fixed by dataset)
    "eeg_in_channels":   1,
    "eeg_spatial":       60,
    "eeg_timesteps":     384,
    # ── NEW: frequency bins derived from rfft of eeg_timesteps ──────────
    # np.fft.rfft of N real samples → N//2 + 1 complex bins → N//2 + 1 magnitudes
    "eeg_freq_bins":     384 // 2 + 1,   # = 193
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

    # ── EEG: Conv2d over (spatial × freq_bins) instead of (spatial × timesteps) ──
    h, w = fixed["eeg_spatial"], fixed["eeg_freq_bins"]   # CHANGED: was eeg_timesteps
    for _ in range(d):
        h = _pool_out(_conv_out(h, k, p))
        w = _pool_out(_conv_out(w, k, p))
    eeg_flat = f * h * w

    # Bio: 1-D conv (temporal only) — unchanged
    t = fixed["bio_timesteps"]
    for _ in range(d):
        t = _pool_out(_conv_out(t, k, p))
    bio_flat = f * t

    fusion = eeg_flat + bio_flat * 3
    return eeg_flat, bio_flat, fusion

_eeg_flat, _bio_flat, _fusion = _compute_flat_sizes(_USER, _FIXED)

HPARAMS = {
    **_USER,
    **_FIXED,
    "eeg_flat":    _eeg_flat,
    "bio_flat":    _bio_flat,
    "fusion_input":_fusion,
}


split_folds = HPARAMS["kfold_splits"]
os.makedirs("training_logs", exist_ok=True)
log_filename = (
    f"training_logs/{split_folds}_kfold_trial_score_log_fr_"
    f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

def _write_csv_header(path, hparams):
    k, p, d, f = hparams["kernel_size"], hparams["padding"], hparams["depth"], hparams["filters"]
    fc_h = hparams["fc_hidden"]

    # ── EEG layers now describe spatial × freq_bins ──────────────────────
    eeg_layers = []
    h, w = hparams["eeg_spatial"], hparams["eeg_freq_bins"]   # CHANGED
    for i in range(d):
        h_c, w_c = _conv_out(h, k, p), _conv_out(w, k, p)
        h_p, w_p = _pool_out(h_c), _pool_out(w_c)
        eeg_layers.append(
            f"  L{i+1}: Conv2d({hparams['eeg_in_channels'] if i==0 else f}→{f}, "
            f"k={k}, p={p}) + ReLU + MaxPool2d(2)  →  {f}×{h_p}×{w_p}"
        )
        h, w = h_p, w_p

    bio_layers = []
    t = hparams["bio_timesteps"]
    for i in range(d):
        t_c = _conv_out(t, k, p)
        t_p = _pool_out(t_c)
        bio_layers.append(
            f"  L{i+1}: Conv1d({hparams['bio_in_channels'] if i==0 else f}→{f}, "
            f"k={k}, p={p}) + ReLU + MaxPool1d(2)  →  {f}×{t_p}"
        )
        t = t_p

    with open(path, mode='w', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow(["# FullADCTModel_TrialScore — Configuration"])
        w.writerow(["# EEG branch uses FREQUENCY DOMAIN (rfft magnitude spectra)"])  # CHANGED note
        w.writerow([])
        w.writerow(["# USER-EDITABLE HYPERPARAMETERS"])
        w.writerow(["depth",        hparams["depth"]])
        w.writerow(["filters",      hparams["filters"]])
        w.writerow(["kernel_size",  hparams["kernel_size"]])
        w.writerow(["padding",      hparams["padding"]])
        w.writerow(["learning_rate",hparams["learning_rate"]])
        w.writerow(["diff_map",     str(hparams["diff_map"])])
        w.writerow([])
        w.writerow(["# FIXED CONSTANTS"])
        for key in ("model_name","dropout","fc_hidden","optimizer","loss_fn",
                    "batch_size","epochs","kfold_splits","kfold_random_state",
                    "eeg_in_channels","eeg_spatial","eeg_timesteps","eeg_freq_bins",  # CHANGED: added eeg_freq_bins
                    "bio_in_channels","bio_timesteps"):
            w.writerow([key, hparams[key]])
        w.writerow([])
        w.writerow(["# AUTO-COMPUTED SIZES"])
        w.writerow(["eeg_flat",    hparams["eeg_flat"]])
        w.writerow(["bio_flat",    hparams["bio_flat"]])
        w.writerow(["fusion_input",hparams["fusion_input"]])
        w.writerow([])
        # ── Updated branch description ────────────────────────────────────
        w.writerow(["# BRANCH 1 — EEG Encoder (Conv2d, FREQUENCY DOMAIN)"])          # CHANGED
        w.writerow([f"  Input: {hparams['eeg_in_channels']}×{hparams['eeg_spatial']}×{hparams['eeg_freq_bins']}  (spatial × freq bins)"])  # CHANGED
        for row in eeg_layers:
            w.writerow([row])
        w.writerow([f"  Flatten → {hparams['eeg_flat']:,}"])
        w.writerow([])
        w.writerow(["# BRANCHES 2-4 — Action / Pupil / Speech (Conv1d)"])
        w.writerow([f"  Input: {hparams['bio_in_channels']}×{hparams['bio_timesteps']}  (each branch)"])
        for row in bio_layers:
            w.writerow([row])
        w.writerow([f"  Flatten → {hparams['bio_flat']:,}  (per branch)"])
        w.writerow([])
        w.writerow(["# FUSION HEAD"])
        w.writerow([f"  Concat: {hparams['eeg_flat']:,} + {hparams['bio_flat']:,}×3 = {hparams['fusion_input']:,}"])
        w.writerow([f"  Linear({hparams['fusion_input']:,} → {fc_h}) + ReLU"])
        w.writerow([f"  Dropout(p={hparams['dropout']})"])
        w.writerow([f"  Linear({fc_h} → 1)  [regression output for TRIAL TOTAL SCORE]"])
        w.writerow([])
        w.writerow(["Epoch",
                    "Train_MSE",
                    "Val_MSE", "Val_RMSE", "Val_MAE", "Val_R2",
                    "Fold",
                    "Total_Params", "Trainable_Params"])

_write_csv_header(log_filename, HPARAMS)
print(f"Logging results to: {log_filename}")


base_path  = "../../data"
merge_keys = ['teamID', 'sessionID', 'trialID', 'ringID']

def load_pkl(filename):
    with open(os.path.join(base_path, filename), 'rb') as f:
        return pickle.load(f)

mod_files = ['epoched_eeg.pkl', 'epoched_pupil.pkl',
             'epoched_speech_event.pkl', 'epoched_action.pkl']
df_main = load_pkl('team_performance.pkl')[merge_keys + ['difficulty']]

for mod_file in mod_files:
    print(f"Merging {mod_file}...")
    mod_df = load_pkl(mod_file)
    cols_to_drop = [c for c in ['difficulty', 'communication'] if c in mod_df.columns]
    if cols_to_drop:
        mod_df = mod_df.drop(columns=cols_to_drop)
    df_main = df_main.merge(mod_df, on=merge_keys, how='inner')
    del mod_df
    gc.collect()

df_main['ring_score'] = df_main['difficulty'].map(HPARAMS["diff_map"])
df_main = df_main.dropna(subset=['ring_score']).reset_index(drop=True)
print(f"Total ring-level records: {len(df_main)}")


print("\n" + "="*70)
print("  AGGREGATING DATA BY TRIAL")
print("="*70)

trial_keys = ['teamID', 'sessionID', 'trialID']

def process_branch(df, cols):
    data = np.stack([np.stack(df[c].values) for c in cols], axis=1)
    if data.ndim == 2:
        data = data.reshape(len(df), 1, -1)
    return data.astype(np.float32)

# ── EEG: compute rfft magnitude spectra BEFORE stacking ─────────────────────
# Each of yawEEG / pitchEEG / thrustEEG is shape (N_samples, 384).
# np.fft.rfft returns complex values; np.abs gives the one-sided magnitude spectrum
# of shape (N_samples, 193).  We then reshape to (N_samples, 60_spatial, 193_freq).
#
# The spatial dimension (60) comes from concatenating the three channel spectra
# laid out side-by-side — same convention as the original time-domain code.
# ────────────────────────────────────────────────────────────────────────────
def to_freq(arr):
    """Apply rfft along the last axis and return magnitude spectra as float32."""
    return np.abs(np.fft.rfft(arr, axis=-1)).astype(np.float32)

eeg_raw = np.stack([
    to_freq(np.stack(df_main['yawEEG'].values)),    # (N, 193)  CHANGED
    to_freq(np.stack(df_main['pitchEEG'].values)),  # (N, 193)  CHANGED
    to_freq(np.stack(df_main['thrustEEG'].values)), # (N, 193)  CHANGED
], axis=1).reshape(
    len(df_main),
    HPARAMS["eeg_spatial"],
    HPARAMS["eeg_freq_bins"],   # CHANGED: was eeg_timesteps (384), now freq bins (193)
).astype(np.float32)

# Bio signals are time-domain — unchanged
act_raw = process_branch(df_main, ['yawAction', 'pitchAction', 'thrustAction'])
pup_raw = process_branch(df_main, ['yawPupil',  'pitchPupil',  'thrustPupil'])
spc_raw = process_branch(df_main, ['yawSpeech', 'pitchSpeech', 'thrustSpeech'])

df_main['eeg_data'] = [eeg_raw[i] for i in range(len(df_main))]
df_main['act_data'] = [act_raw[i] for i in range(len(df_main))]
df_main['pup_data'] = [pup_raw[i] for i in range(len(df_main))]
df_main['spc_data'] = [spc_raw[i] for i in range(len(df_main))]

print("  Grouping by trial and computing:")
print("    - Total score (sum of ring scores)")
print("    - Average features (mean across rings in trial)")

def aggregate_arrays(series):
    return np.mean(list(series), axis=0)

trial_aggregated = df_main.groupby(trial_keys).agg({
    'ring_score': 'sum',
    'eeg_data': aggregate_arrays,
    'act_data': aggregate_arrays,
    'pup_data': aggregate_arrays,
    'spc_data': aggregate_arrays,
}).reset_index()

trial_aggregated.rename(columns={'ring_score': 'trial_total_score'}, inplace=True)

print(f"\n  Original ring records:  {len(df_main):,}")
print(f"  Aggregated trials:      {len(trial_aggregated):,}")
print(f"  Score range:            [{trial_aggregated['trial_total_score'].min():.0f}, "
      f"{trial_aggregated['trial_total_score'].max():.0f}]")
print(f"  Mean trial score:       {trial_aggregated['trial_total_score'].mean():.2f}")
print("="*70)

eeg_trial = np.stack(trial_aggregated['eeg_data'].values).astype(np.float32)
act_trial = np.stack(trial_aggregated['act_data'].values).astype(np.float32)
pup_trial = np.stack(trial_aggregated['pup_data'].values).astype(np.float32)
spc_trial = np.stack(trial_aggregated['spc_data'].values).astype(np.float32)
y_trial_scores = trial_aggregated['trial_total_score'].values.astype(np.float32)

del df_main, eeg_raw, act_raw, pup_raw, spc_raw
gc.collect()

def normalize_3d(data):
    s, c, t = data.shape
    scaler   = StandardScaler()
    reshaped = data.transpose(0, 2, 1).reshape(-1, c)
    normed   = scaler.fit_transform(reshaped)
    return normed.reshape(s, t, c).transpose(0, 2, 1).astype(np.float32)

print("\nNormalizing features...")
eeg_norm = normalize_3d(eeg_trial)   # shape: (N_trials, 60, 193)  CHANGED
pup_norm = normalize_3d(pup_trial)

del eeg_trial, pup_trial
gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  — no structural changes; dimensions flow in via HPARAMS
# ═══════════════════════════════════════════════════════════════════════

class FullADCTModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        d   = hparams["depth"]
        f   = hparams["filters"]
        k   = hparams["kernel_size"]
        p   = hparams["padding"]
        dr  = hparams["dropout"]
        fc_h= hparams["fc_hidden"]
        eeg_flat  = hparams["eeg_flat"]
        bio_flat  = hparams["bio_flat"]
        fusion    = hparams["fusion_input"]

        # Branch 1: EEG  (Conv2d — spatial × freq_bins)  CHANGED comment only
        eeg_layers = []
        in_ch = 1
        for _ in range(d):
            eeg_layers += [nn.Conv2d(in_ch, f, k, padding=p), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = f
        eeg_layers.append(nn.Flatten())
        self.eeg_branch = nn.Sequential(*eeg_layers)

        def _bio_branch():
            layers, in_ch = [], hparams["bio_in_channels"]
            for _ in range(d):
                layers += [nn.Conv1d(in_ch, f, k, padding=p), nn.ReLU(), nn.MaxPool1d(2)]
                in_ch = f
            layers.append(nn.Flatten())
            return nn.Sequential(*layers)

        self.act_branch = _bio_branch()
        self.pup_branch = _bio_branch()
        self.spc_branch = _bio_branch()

        self.fc = nn.Sequential(
            nn.Linear(fusion, fc_h),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(fc_h, 1),
        )

    def forward(self, eeg, act, pup, spc):
        b1 = self.eeg_branch(eeg.unsqueeze(1))
        b2 = self.act_branch(act)
        b3 = self.pup_branch(pup)
        b4 = self.spc_branch(spc)
        return self.fc(torch.cat((b1, b2, b3, b4), dim=1))


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_details(model, hparams):
    total, trainable = count_params(model)
    d, f = hparams["depth"], hparams["filters"]
    k, p = hparams["kernel_size"], hparams["padding"]
    fc_h = hparams["fc_hidden"]
    W    = 66
    SEP  = "=" * W
    SEP2 = "-" * W

    # ── EEG layers now use freq_bins on the width axis ───────────────────
    eeg_desc = []
    h, w = hparams["eeg_spatial"], hparams["eeg_freq_bins"]   # CHANGED
    in_ch = 1
    for i in range(d):
        h_c, w_c = _conv_out(h, k, p), _conv_out(w, k, p)
        h_p, w_p = _pool_out(h_c), _pool_out(w_c)
        eeg_desc.append(
            f"  Layer {i+1}  Conv2d({in_ch}→{f}, k={k}, p={p}) + ReLU + MaxPool2d(2)")
        eeg_desc.append(f"           Output: {f} × {h_p} × {w_p}")
        h, w, in_ch = h_p, w_p, f
    eeg_flat = f * h * w

    bio_desc = []
    t, in_ch = hparams["bio_timesteps"], hparams["bio_in_channels"]
    for i in range(d):
        t_c = _conv_out(t, k, p)
        t_p = _pool_out(t_c)
        bio_desc.append(
            f"  Layer {i+1}  Conv1d({in_ch}→{f}, k={k}, p={p}) + ReLU + MaxPool1d(2)")
        bio_desc.append(f"           Output: {f} × {t_p}")
        t, in_ch = t_p, f
    bio_flat = f * t
    fusion   = eeg_flat + bio_flat * 3

    lines = [
        SEP,
        "  FullADCTModel_TrialScore — Architecture Summary",
        "  EEG branch: FREQUENCY DOMAIN (rfft magnitude spectra)",   # CHANGED
        SEP,
        "  TASK: Predict total trial score (sum of all passed rings)",
        SEP2,
        "  USER-EDITABLE HYPERPARAMETERS",
        SEP2,
        f"  {'Depth':<30} {d}",
        f"  {'Filters':<30} {f}",
        f"  {'Kernel Size':<30} {k}",
        f"  {'Padding':<30} {p}",
        f"  {'Learning Rate':<30} {hparams['learning_rate']}",
        f"  {'Difficulty Map':<30} {hparams['diff_map']}",
        SEP2,
        "  FIXED CONSTANTS",
        SEP2,
        f"  {'Dropout':<30} {hparams['dropout']}",
        f"  {'FC Hidden Units':<30} {fc_h}",
        f"  {'Optimizer':<30} {hparams['optimizer']}",
        f"  {'Loss Function':<30} {hparams['loss_fn']}",
        f"  {'Batch Size':<30} {hparams['batch_size']}",
        f"  {'Epochs per Fold':<30} {hparams['epochs']}",
        f"  {'K-Fold Splits':<30} {hparams['kfold_splits']}",
        SEP2,
        f"  BRANCH 1 — EEG Encoder  (Conv2d, freq domain, depth={d})",   # CHANGED
        SEP2,
        # ── Report spatial × freq_bins ───────────────────────────────────
        f"  Input Shape  {hparams['eeg_in_channels']} × {hparams['eeg_spatial']} × {hparams['eeg_freq_bins']}  (spatial × freq bins)",  # CHANGED
        *eeg_desc,
        f"  Flatten  →  {eeg_flat:,}",
        SEP2,
        f"  BRANCHES 2-4 — Action / Pupil / Speech  (Conv1d, depth={d})",
        SEP2,
        f"  Input Shape  {hparams['bio_in_channels']} × {hparams['bio_timesteps']}  (each branch)",
        *bio_desc,
        f"  Flatten  →  {bio_flat:,}  (per branch)",
        SEP2,
        "  FUSION HEAD",
        SEP2,
        f"  Concat   {eeg_flat:,} + {bio_flat:,} × 3 = {fusion:,}",
        f"  FC-1     Linear({fusion:,} → {fc_h}) + ReLU",
        f"  Dropout  p = {hparams['dropout']}",
        f"  FC-2     Linear({fc_h} → 1)  [trial total score output]",
        SEP2,
        "  PARAMETER COUNT",
        SEP2,
        f"  {'Total Parameters':<30} {total:,}",
        f"  {'Trainable Parameters':<30} {trainable:,}",
        SEP,
    ]
    for line in lines:
        print(line)
    return total, trainable


# ── Training loop: identical to original ────────────────────────────────────

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

full_dataset = TensorDataset(
    torch.tensor(eeg_norm,       dtype=torch.float32),
    torch.tensor(act_trial,      dtype=torch.float32),
    torch.tensor(pup_norm,       dtype=torch.float32),
    torch.tensor(spc_trial,      dtype=torch.float32),
    torch.tensor(y_trial_scores, dtype=torch.float32),
)

kf = KFold(n_splits=split_folds,
           shuffle=HPARAMS["kfold_shuffle"],
           random_state=HPARAMS["kfold_random_state"])

print(f"\nStarting {split_folds}-Fold Cross-Validation on {device}...")

if HAS_GRAPH_GEN:
    generate_model_graph(out_dir="training_logs", hparams=HPARAMS)

train_mse_list  = []
train_rmse_list = []
train_mae_list  = []
train_r2_list   = []
val_mse_list    = []
val_rmse_list   = []
val_mae_list    = []
val_r2_list     = []

total_params = trainable_params = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)))):
    print(f"\n{'='*66}")
    print(f"  FOLD {fold + 1} / {split_folds}")
    print(f"{'='*66}")

    train_sub    = Subset(full_dataset, train_idx)
    val_sub      = Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_sub, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=HPARAMS["batch_size"], shuffle=False)

    model     = FullADCTModel(HPARAMS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"])
    criterion = nn.MSELoss()

    if fold == 0:
        total_params, trainable_params = print_model_details(model, HPARAMS)
    else:
        total_params, trainable_params = count_params(model)

    for epoch in range(HPARAMS["epochs"]):
        model.train()
        train_preds, train_targets = [], []
        for b_eeg, b_act, b_pup, b_spc, b_y in train_loader:
            inputs = [x.to(device) for x in [b_eeg, b_act, b_pup, b_spc]]
            target = b_y.to(device).view(-1, 1)
            optimizer.zero_grad()
            out  = model(*inputs)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_preds.append(out.detach().cpu().numpy())
            train_targets.append(target.detach().cpu().numpy())

        train_preds   = np.concatenate(train_preds).flatten()
        train_targets = np.concatenate(train_targets).flatten()
        tr_mse  = float(np.mean((train_preds - train_targets) ** 2))
        tr_rmse = float(np.sqrt(tr_mse))
        tr_mae  = float(mean_absolute_error(train_targets, train_preds))
        tr_r2   = float(r2_score(train_targets, train_preds))

        model.eval()
        val_preds, val_targets_ep = [], []
        with torch.no_grad():
            for b_eeg, b_act, b_pup, b_spc, b_y in val_loader:
                inputs = [x.to(device) for x in [b_eeg, b_act, b_pup, b_spc]]
                target = b_y.to(device).view(-1, 1)
                val_preds.append(model(*inputs).cpu().numpy())
                val_targets_ep.append(target.cpu().numpy())

        val_preds      = np.concatenate(val_preds).flatten()
        val_targets_ep = np.concatenate(val_targets_ep).flatten()
        v_mse  = float(np.mean((val_preds - val_targets_ep) ** 2))
        v_rmse = float(np.sqrt(v_mse))
        v_mae  = float(mean_absolute_error(val_targets_ep, val_preds))
        v_r2   = float(r2_score(val_targets_ep, val_preds))

        with open(log_filename, mode='a', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow([
                epoch + 1,
                f"{tr_mse:.4f}",
                f"{v_mse:.4f}", f"{v_rmse:.4f}", f"{v_mae:.4f}", f"{v_r2:.4f}",
                fold + 1,
                total_params, trainable_params,
            ])

        print(
            f"  Epoch {epoch+1:02d} | "
            f"Train MSE: {tr_mse:.4f}  RMSE: {tr_rmse:.4f}  "
            f"MAE: {tr_mae:.4f}  R²: {tr_r2:.4f}"
        )

    train_mse_list.append(tr_mse);   train_rmse_list.append(tr_rmse)
    train_mae_list.append(tr_mae);   train_r2_list.append(tr_r2)
    val_mse_list.append(v_mse);      val_rmse_list.append(v_rmse)
    val_mae_list.append(v_mae);      val_r2_list.append(v_r2)

    del model, optimizer, train_loader, val_loader
    gc.collect()


W   = 66
SEP = "=" * W
SEP2 = "-" * W

print(f"\n{SEP}")
print(f"  {split_folds}-FOLD CROSS-VALIDATION SUMMARY  (final epoch per fold)")
print(f"  TASK: Trial Total Score Prediction")
print(f"{SEP}")
print(f"  {'Metric':<10}  {'Train Mean':>12}  {'Train SD':>10}  {'Val Mean':>12}  {'Val SD':>10}")
print(f"  {SEP2}")
for label, t_list, v_list in [
    ("MSE",  train_mse_list,  val_mse_list),
    ("RMSE", train_rmse_list, val_rmse_list),
    ("MAE",  train_mae_list,  val_mae_list),
    ("R²",   train_r2_list,   val_r2_list),
]:
    print(
        f"  {label:<10}  "
        f"{np.mean(t_list):>12.4f}  {np.std(t_list):>10.4f}  "
        f"{np.mean(v_list):>12.4f}  {np.std(v_list):>10.4f}"
    )
print(f"  {SEP2}")
print(f"  Total Parameters    : {total_params:,}")
print(f"  Trainable Parameters: {trainable_params:,}")
print(f"  Device              : {device}")
print(f"  Log file            : {log_filename}")
print(f"{SEP}")

with open(log_filename, mode='a', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow([])
    writer.writerow(["# CROSS-VALIDATION SUMMARY (final epoch per fold)"])
    writer.writerow(["# TASK: Trial Total Score Prediction"])
    writer.writerow(["Metric",
                     "Train_Mean", "Train_SD_CV",
                     "Val_Mean",   "Val_SD_CV"])
    for label, t_list, v_list in [
        ("MSE",  train_mse_list,  val_mse_list),
        ("RMSE", train_rmse_list, val_rmse_list),
        ("MAE",  train_mae_list,  val_mae_list),
        ("R2",   train_r2_list,   val_r2_list),
    ]:
        writer.writerow([
            label,
            f"{np.mean(t_list):.4f}", f"{np.std(t_list):.4f}",
            f"{np.mean(v_list):.4f}", f"{np.std(v_list):.4f}",
        ])
    writer.writerow([])
    writer.writerow(["Total_Params",     total_params])
    writer.writerow(["Trainable_Params", trainable_params])
    writer.writerow(["Device",           device])