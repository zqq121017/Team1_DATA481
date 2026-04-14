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

# Model architecture graph generator (place generate_model_graph.py alongside this file)
from generate_model_graph import generate_model_graph

# ═══════════════════════════════════════════════════════════════════
#  ✏️  USER-EDITABLE HYPERPARAMETERS  — only change values here
# ═══════════════════════════════════════════════════════════════════
_USER = {
    "depth":        3,        # number of Conv layers per branch
    "filters":      64,       # convolutional filters per layer
    "kernel_size":  3,        # conv kernel size (int, same for all branches)
    "padding":      1,        # conv padding   (int, same for all branches)
    "learning_rate":0.0005,   # Adam LR
    # ── difficulty label → target score mapping ──
    "diff_map": {"Easy": 1, "Medium": 2, "Hard": 3},
}

# ═══════════════════════════════════════════════════════════════════
#  🔒  FIXED ARCHITECTURE CONSTANTS  (do not edit)
# ═══════════════════════════════════════════════════════════════════
_FIXED = {
    "model_name":        "FullADCTModel",
    "dropout":           0.4,
    "fc_hidden":         128,
    "optimizer":         "Adam",
    "loss_fn":           "MSELoss",
    "batch_size":        32,
    "epochs":            30,
    "kfold_splits":      5,
    "kfold_shuffle":     True,
    "kfold_random_state":42,
    # Input shapes (fixed by dataset)
    "eeg_in_channels":   1,
    "eeg_spatial":       60,
    "eeg_timesteps":     384,
    "bio_in_channels":   3,
    "bio_timesteps":     90,
}

# ═══════════════════════════════════════════════════════════════════
#  🤖  AUTO-COMPUTED DERIVED SIZES  (driven entirely by _USER)
# ═══════════════════════════════════════════════════════════════════
def _conv_out(size, k, p, s=1):
    return math.floor((size + 2 * p - k) / s + 1)

def _pool_out(size, k=2):
    return size // k

def _compute_flat_sizes(user, fixed):
    """Compute EEG flat and bio flat sizes from depth/kernel/padding."""
    k, p, d = user["kernel_size"], user["padding"], user["depth"]
    f = user["filters"]

    # EEG: 2-D conv  (spatial × temporal)
    h, w = fixed["eeg_spatial"], fixed["eeg_timesteps"]
    for _ in range(d):
        h = _pool_out(_conv_out(h, k, p))
        w = _pool_out(_conv_out(w, k, p))
    eeg_flat = f * h * w

    # Bio: 1-D conv  (temporal only)
    t = fixed["bio_timesteps"]
    for _ in range(d):
        t = _pool_out(_conv_out(t, k, p))
    bio_flat = f * t

    fusion = eeg_flat + bio_flat * 3
    return eeg_flat, bio_flat, fusion

_eeg_flat, _bio_flat, _fusion = _compute_flat_sizes(_USER, _FIXED)

# Merge everything into a single HPARAMS dict (read-only after this point)
HPARAMS = {
    **_USER,
    **_FIXED,
    "eeg_flat":    _eeg_flat,
    "bio_flat":    _bio_flat,
    "fusion_input":_fusion,
}

# ═══════════════════════════════════════════════════════════════════
#  Logging setup
# ═══════════════════════════════════════════════════════════════════
split_folds = HPARAMS["kfold_splits"]
os.makedirs("training_logs", exist_ok=True)
log_filename = (
    f"training_logs/{split_folds}_kfold_training_log_"
    f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

def _write_csv_header(path, hparams):
    """Write full model config block then the column headers."""
    k, p, d, f = hparams["kernel_size"], hparams["padding"], hparams["depth"], hparams["filters"]
    fc_h = hparams["fc_hidden"]

    # Build human-readable layer strings from actual computed sizes
    eeg_layers = []
    h, w = hparams["eeg_spatial"], hparams["eeg_timesteps"]
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
        w.writerow(["# FullADCTModel — Configuration"])
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
                    "eeg_in_channels","eeg_spatial","eeg_timesteps",
                    "bio_in_channels","bio_timesteps"):
            w.writerow([key, hparams[key]])
        w.writerow([])
        w.writerow(["# AUTO-COMPUTED SIZES"])
        w.writerow(["eeg_flat",    hparams["eeg_flat"]])
        w.writerow(["bio_flat",    hparams["bio_flat"]])
        w.writerow(["fusion_input",hparams["fusion_input"]])
        w.writerow([])
        w.writerow(["# BRANCH 1 — EEG Encoder (Conv2d)"])
        w.writerow([f"  Input: {hparams['eeg_in_channels']}×{hparams['eeg_spatial']}×{hparams['eeg_timesteps']}"])
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
        w.writerow([f"  Linear({fc_h} → 1)  [regression output]"])
        w.writerow([])
        # ── epoch log columns ──
        w.writerow(["Epoch",
                    "Train_MSE",
                    "Val_MSE", "Val_RMSE", "Val_MAE", "Val_R2",
                    "Fold",
                    "Total_Params", "Trainable_Params"])

_write_csv_header(log_filename, HPARAMS)
print(f"Logging results to: {log_filename}")

# ═══════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════
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

# Use diff_map from HPARAMS — automatically reflects any user change
df_main['target_score'] = df_main['difficulty'].map(HPARAMS["diff_map"])
df_main = df_main.dropna(subset=['target_score']).reset_index(drop=True)
print(f"Final aligned dataset size: {len(df_main)}")

def process_branch(df, cols):
    data = np.stack([np.stack(df[c].values) for c in cols], axis=1)
    if data.ndim == 2:
        data = data.reshape(len(df), 1, -1)
    return data.astype(np.float32)

eeg_raw = np.stack([
    np.stack(df_main['yawEEG'].values),
    np.stack(df_main['pitchEEG'].values),
    np.stack(df_main['thrustEEG'].values),
], axis=1).reshape(len(df_main),
                   HPARAMS["eeg_spatial"],
                   HPARAMS["eeg_timesteps"]).astype(np.float32)

act_raw = process_branch(df_main, ['yawAction', 'pitchAction', 'thrustAction'])
pup_raw = process_branch(df_main, ['yawPupil',  'pitchPupil',  'thrustPupil'])
spc_raw = process_branch(df_main, ['yawSpeech', 'pitchSpeech', 'thrustSpeech'])
y_labels = df_main['target_score'].values.astype(np.float32)

def normalize_3d(data):
    s, c, t = data.shape
    scaler   = StandardScaler()
    reshaped = data.transpose(0, 2, 1).reshape(-1, c)
    normed   = scaler.fit_transform(reshaped)
    return normed.reshape(s, t, c).transpose(0, 2, 1).astype(np.float32)

eeg_norm = normalize_3d(eeg_raw)
pup_norm = normalize_3d(pup_raw)

del df_main, eeg_raw, pup_raw
gc.collect()


# ═══════════════════════════════════════════════════════════════════
#  Model  — fully driven by HPARAMS
# ═══════════════════════════════════════════════════════════════════
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

        # ── Branch 1: EEG  (Conv2d) ──────────────────────────────
        eeg_layers = []
        in_ch = 1
        for _ in range(d):
            eeg_layers += [nn.Conv2d(in_ch, f, k, padding=p), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = f
        eeg_layers.append(nn.Flatten())
        self.eeg_branch = nn.Sequential(*eeg_layers)

        # ── Branches 2-4: Bio (Conv1d) ────────────────────────────
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

        # ── Fusion head ───────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_details(model, hparams):
    """Print architecture summary driven entirely by hparams (no hard-coded numbers)."""
    total, trainable = count_params(model)
    d, f = hparams["depth"], hparams["filters"]
    k, p = hparams["kernel_size"], hparams["padding"]
    fc_h = hparams["fc_hidden"]
    W    = 66
    SEP  = "=" * W
    SEP2 = "-" * W

    # Build EEG layer descriptions from actual sizes
    eeg_desc = []
    h, w = hparams["eeg_spatial"], hparams["eeg_timesteps"]
    in_ch = 1
    for i in range(d):
        h_c, w_c = _conv_out(h, k, p), _conv_out(w, k, p)
        h_p, w_p = _pool_out(h_c), _pool_out(w_c)
        eeg_desc.append(
            f"  Layer {i+1}  Conv2d({in_ch}→{f}, k={k}, p={p}) + ReLU + MaxPool2d(2)")
        eeg_desc.append(f"           Output: {f} × {h_p} × {w_p}")
        h, w, in_ch = h_p, w_p, f
    eeg_flat = f * h * w

    # Build bio layer descriptions
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
        "  FullADCTModel — Architecture Summary",
        SEP,
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
        f"  BRANCH 1 — EEG Encoder  (Conv2d, depth={d})",
        SEP2,
        f"  Input Shape  {hparams['eeg_in_channels']} × {hparams['eeg_spatial']} × {hparams['eeg_timesteps']}",
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
        f"  FC-2     Linear({fc_h} → 1)  [regression output]",
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


# ═══════════════════════════════════════════════════════════════════
#  Device
# ═══════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ═══════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════
full_dataset = TensorDataset(
    torch.tensor(eeg_norm,  dtype=torch.float32),
    torch.tensor(act_raw,   dtype=torch.float32),
    torch.tensor(pup_norm,  dtype=torch.float32),
    torch.tensor(spc_raw,   dtype=torch.float32),
    torch.tensor(y_labels,  dtype=torch.float32),
)

kf = KFold(n_splits=split_folds,
           shuffle=HPARAMS["kfold_shuffle"],
           random_state=HPARAMS["kfold_random_state"])

print(f"\nStarting {split_folds}-Fold Cross-Validation on {device}...")

# Generate architecture graph (uses HPARAMS → reflects current depth/filters/etc.)
generate_model_graph(out_dir="training_logs", hparams=HPARAMS)

# ═══════════════════════════════════════════════════════════════════
#  Per-fold metric collectors
# ═══════════════════════════════════════════════════════════════════
train_mse_list  = []
train_rmse_list = []
train_mae_list  = []
train_r2_list   = []

val_mse_list    = []
val_rmse_list   = []
val_mae_list    = []
val_r2_list     = []

total_params = trainable_params = 0

# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════
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

        # ── TRAIN ─────────────────────────────────────────────────
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

        # ── VALIDATE ──────────────────────────────────────────────
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

        # ── CSV LOG ───────────────────────────────────────────────
        with open(log_filename, mode='a', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow([
                epoch + 1,
                f"{tr_mse:.4f}",
                f"{v_mse:.4f}", f"{v_rmse:.4f}", f"{v_mae:.4f}", f"{v_r2:.4f}",
                fold + 1,
                total_params, trainable_params,
            ])

        # ── CONSOLE: train error only per epoch ───────────────────
        print(
            f"  Epoch {epoch+1:02d} | "
            f"Train MSE: {tr_mse:.4f}  RMSE: {tr_rmse:.4f}  "
            f"MAE: {tr_mae:.4f}  R²: {tr_r2:.4f}"
        )

    # Store final-epoch metrics for cross-val summary
    train_mse_list.append(tr_mse);   train_rmse_list.append(tr_rmse)
    train_mae_list.append(tr_mae);   train_r2_list.append(tr_r2)
    val_mse_list.append(v_mse);      val_rmse_list.append(v_rmse)
    val_mae_list.append(v_mae);      val_r2_list.append(v_r2)

    del model, optimizer, train_loader, val_loader
    gc.collect()


# ═══════════════════════════════════════════════════════════════════
#  Final summary — both Train and Validation
# ═══════════════════════════════════════════════════════════════════
W   = 66
SEP = "=" * W
SEP2 = "-" * W

print(f"\n{SEP}")
print(f"  {split_folds}-FOLD CROSS-VALIDATION SUMMARY  (final epoch per fold)")
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

# ── Append summary block to CSV ───────────────────────────────────
with open(log_filename, mode='a', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow([])
    writer.writerow(["# CROSS-VALIDATION SUMMARY (final epoch per fold)"])
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