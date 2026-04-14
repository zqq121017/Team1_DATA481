import pandas as pd
import numpy as np
import pickle
import os
import gc
import math
from scipy.signal import stft as scipy_stft
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

# ═══════════════════════════════════════════════════════════════════════
#  EEG CHANNEL & FREQUENCY BAND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
#  Original data: each EEG array is (20 channels × 384 timesteps)
#  We extract 3 channels by index:
#    Fz  → index 12 (row 13)
#    Cz  → index 18 (row 19)
#    POz → index 16 (row 17)
#  Then filter to 2 frequency bands:
#    Delta : 0.5 –  4 Hz
#    Alpha : 8   – 13 Hz
#  3 channels × 2 bands = 6 EEG feature maps (in_channels = 6)
#
#  STFT output axes:
#    x-axis : frequency  (F bins, filtered to delta+alpha)
#    y-axis : time       (T frames)
#  Conv2d input shape: (batch, 6, T, F)
# ═══════════════════════════════════════════════════════════════════════

EEG_CH_INDICES = [12, 18, 16]          # Fz, Cz, POz
EEG_CH_NAMES   = ["Fz", "Cz", "POz"]
FREQ_BANDS     = {
    "delta": (0.5,  4.0),
    "alpha": (8.0, 13.0),
}
SFREQ          = 256          # EEG sampling frequency in Hz
NPERSEG        = 64           # STFT window length → freq resolution = 128/64 = 2 Hz
                              # rfft bins: 0,2,4,6,8,10,12,14,...Hz  (33 bins total)
                              # Delta 0.5–4 Hz  → bins at 2Hz(1), 4Hz(2)           → 2 bins
                              # Alpha 8–13 Hz   → bins at 8Hz(4),10Hz(5),12Hz(6)   → 3 bins
                              # Total F = 5 freq bins kept
NOVERLAP       = 56           # hop = 8 samples → T ≈ ceil(384/8) = 48 time frames
# T ≈ 48, F = 5  (exact values computed at runtime by _stft_probe)

# ═══════════════════════════════════════════════════════════════════════
#  USER-EDITABLE HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════
_USER = {
    "depth":               3,
    "filters":             64,
    "kernel_size":         3,       # reduced from 10; TF maps are small (T≈49, F≈5)
    "padding":             1,
    "learning_rate":       0.0005,
    "weight_decay":        1e-4,    # AdamW decoupled L2 regularisation
    "early_stop_patience": 30,      # stop if val_MSE doesn't improve for 15 epochs
    "early_stop_delta":    0.001,   # minimum drop in val_MSE to count as improvement
    "diff_map":            {"Easy": 1, "Medium": 2, "Hard": 3},
}

_FIXED = {
    "model_name":         "FullADCTModel_TFPower_TrialScore",
    "dropout":            0.4,
    "fc_hidden":          128,
    "optimizer":          "AdamW",
    "loss_fn":            "MSELoss",
    "batch_size":         32,
    "epochs":             300,
    "kfold_splits":       5,
    "kfold_shuffle":      True,
    "kfold_random_state": 42,
    # EEG input (set after STFT probe; placeholders overwritten below)
    "eeg_in_channels":    6,        # 3 channels × 2 bands
    "eeg_time_frames":    None,     # T — filled after STFT probe
    "eeg_freq_bins":      None,     # F — filled after STFT probe
    # Bio branches (unchanged)
    "bio_in_channels":    3,
    "bio_timesteps":      90,
}


# ───────────────────────────────────────────────────────────────────────
# STFT PROBE  — compute exact T and F from a dummy signal
# ───────────────────────────────────────────────────────────────────────
def _stft_probe():
    dummy = np.zeros(384, dtype=np.float32)
    freqs, times, Zxx = scipy_stft(
        dummy, fs=SFREQ, nperseg=NPERSEG, noverlap=NOVERLAP,
        boundary="zeros", padded=True,
    )
    # collect freq indices for delta + alpha
    idx = []
    for lo, hi in FREQ_BANDS.values():
        mask = (freqs >= lo) & (freqs <= hi)
        idx.extend(np.where(mask)[0].tolist())
    idx = sorted(set(idx))
    return len(times), len(idx), freqs, idx

_T, _F, _PROBE_FREQS, _BAND_IDX = _stft_probe()
_FIXED["eeg_time_frames"] = _T
_FIXED["eeg_freq_bins"]   = _F

print(f"STFT probe  →  time_frames={_T},  freq_bins={_F}")
print(f"  Freq axis  : {_PROBE_FREQS[_BAND_IDX]} Hz  (delta + alpha)")


# ═══════════════════════════════════════════════════════════════════════
#  DIMENSION HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _conv_out(size, k, p, s=1):
    return math.floor((size + 2 * p - k) / s + 1)

def _pool_out(size, k=2):
    return size // k

def _compute_flat_sizes(user, fixed):
    k, p, d = user["kernel_size"], user["padding"], user["depth"]
    f = user["filters"]

    # EEG: Conv2d over (time_frames × freq_bins)
    # Asymmetric pooling: MaxPool2d(kernel_size=(2,1)) — pool TIME only,
    # keep FREQ axis intact (F is too small to pool: 5 bins → crashes at depth 3)
    h, w = fixed["eeg_time_frames"], fixed["eeg_freq_bins"]
    for _ in range(d):
        h = _pool_out(_conv_out(h, k, p), k=2)   # time: halved each layer
        w = _conv_out(w, k, p)                    # freq: conv only, NO pooling
    eeg_flat = f * h * w

    # Bio: Conv1d (temporal)
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
    "eeg_flat":     _eeg_flat,
    "bio_flat":     _bio_flat,
    "fusion_input": _fusion,
}

print(f"  eeg_flat={_eeg_flat},  bio_flat={_bio_flat},  fusion={_fusion}")


# ═══════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════
split_folds = HPARAMS["kfold_splits"]
os.makedirs("training_logs", exist_ok=True)
log_filename = (
    f"training_logs/{split_folds}_kfold_tfpower_trial_score_"
    f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

def _write_csv_header(path, hparams):
    k, p, d, f = (hparams["kernel_size"], hparams["padding"],
                  hparams["depth"], hparams["filters"])
    fc_h = hparams["fc_hidden"]

    eeg_layers = []
    h, w = hparams["eeg_time_frames"], hparams["eeg_freq_bins"]
    in_ch = hparams["eeg_in_channels"]
    for i in range(d):
        h_c, w_c = _conv_out(h, k, p), _conv_out(w, k, p)
        h_p = _pool_out(h_c)   # time: pooled
        w_p = w_c              # freq: conv only, no pool
        eeg_layers.append(
            f"  L{i+1}: Conv2d({in_ch}→{f}, k={k}, p={p}) + ReLU "
            f"+ MaxPool2d(2,1) → {f}×{h_p}×{w_p}"
        )
        h, w, in_ch = h_p, w_p, f

    bio_layers = []
    t = hparams["bio_timesteps"]
    in_ch = hparams["bio_in_channels"]
    for i in range(d):
        t_c = _conv_out(t, k, p)
        t_p = _pool_out(t_c)
        bio_layers.append(
            f"  L{i+1}: Conv1d({in_ch}→{f}, k={k}, p={p}) + ReLU + MaxPool1d(2) → {f}×{t_p}"
        )
        t, in_ch = t_p, f

    with open(path, mode='w', newline='') as f_out:
        w_csv = csv.writer(f_out)
        w_csv.writerow(["# FullADCTModel_TFPower_TrialScore — Configuration"])
        w_csv.writerow(["# EEG: STFT power (delta+alpha), channels Fz/Cz/POz"])
        w_csv.writerow([])
        w_csv.writerow(["# EEG CHANNEL CONFIG"])
        w_csv.writerow(["channels",      str(EEG_CH_NAMES)])
        w_csv.writerow(["ch_indices",    str(EEG_CH_INDICES)])
        w_csv.writerow(["freq_bands",    str(FREQ_BANDS)])
        w_csv.writerow(["sfreq_hz",      SFREQ])
        w_csv.writerow(["stft_nperseg",  NPERSEG])
        w_csv.writerow(["stft_noverlap", NOVERLAP])
        w_csv.writerow(["stft_freqs_hz", str(list(_PROBE_FREQS[_BAND_IDX].round(2)))])
        w_csv.writerow([])
        w_csv.writerow(["# USER-EDITABLE HYPERPARAMETERS"])
        w_csv.writerow(["depth",                hparams["depth"]])
        w_csv.writerow(["filters",              hparams["filters"]])
        w_csv.writerow(["kernel_size",          hparams["kernel_size"]])
        w_csv.writerow(["padding",              hparams["padding"]])
        w_csv.writerow(["learning_rate",        hparams["learning_rate"]])
        w_csv.writerow(["weight_decay",         hparams["weight_decay"]])
        w_csv.writerow(["early_stop_patience",  hparams["early_stop_patience"]])
        w_csv.writerow(["early_stop_delta",     hparams["early_stop_delta"]])
        w_csv.writerow(["diff_map",             str(hparams["diff_map"])])
        w_csv.writerow([])
        w_csv.writerow(["# FIXED CONSTANTS"])
        for key in ("model_name","dropout","fc_hidden","optimizer","loss_fn",
                    "batch_size","epochs","kfold_splits","kfold_random_state",
                    "eeg_in_channels","eeg_time_frames","eeg_freq_bins",
                    "bio_in_channels","bio_timesteps"):
            w_csv.writerow([key, hparams[key]])
        w_csv.writerow([])
        w_csv.writerow(["# AUTO-COMPUTED SIZES"])
        w_csv.writerow(["eeg_flat",     hparams["eeg_flat"]])
        w_csv.writerow(["bio_flat",     hparams["bio_flat"]])
        w_csv.writerow(["fusion_input", hparams["fusion_input"]])
        w_csv.writerow([])
        w_csv.writerow([
            f"# BRANCH 1 — EEG Encoder (Conv2d, TF-power, "
            f"in_channels={hparams['eeg_in_channels']})"
        ])
        w_csv.writerow([
            f"  Input: {hparams['eeg_in_channels']} × "
            f"{hparams['eeg_time_frames']} × {hparams['eeg_freq_bins']}  "
            f"(6 ch × time_frames × freq_bins)"
        ])
        for row in eeg_layers:
            w_csv.writerow([row])
        w_csv.writerow([f"  Flatten → {hparams['eeg_flat']:,}"])
        w_csv.writerow([])
        w_csv.writerow(["# BRANCHES 2-4 — Action / Pupil / Speech (Conv1d)"])
        w_csv.writerow([f"  Input: {hparams['bio_in_channels']}×{hparams['bio_timesteps']}"])
        for row in bio_layers:
            w_csv.writerow([row])
        w_csv.writerow([f"  Flatten → {hparams['bio_flat']:,}  (per branch)"])
        w_csv.writerow([])
        w_csv.writerow(["# FUSION HEAD"])
        w_csv.writerow([
            f"  Concat: {hparams['eeg_flat']:,} + "
            f"{hparams['bio_flat']:,}×3 = {hparams['fusion_input']:,}"
        ])
        w_csv.writerow([f"  Linear({hparams['fusion_input']:,} → {fc_h}) + ReLU"])
        w_csv.writerow([f"  Dropout(p={hparams['dropout']})"])
        w_csv.writerow([f"  Linear({fc_h} → 1)  [regression: trial total score]"])
        w_csv.writerow([])
        w_csv.writerow(["# NOTE: Only epochs where val_MSE improved are logged"])
        w_csv.writerow(["# [BEST] marks the epoch restored at early stopping"])
        w_csv.writerow([])
        w_csv.writerow(["Epoch",
                        "Train_MSE",
                        "Val_MSE", "Val_RMSE", "Val_MAE", "Val_R2",
                        "Fold",
                        "Total_Params", "Trainable_Params",
                        "Note"])

_write_csv_header(log_filename, HPARAMS)
print(f"Logging to: {log_filename}")


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING & MERGING  (identical to original)
# ═══════════════════════════════════════════════════════════════════════
base_path  = "../../data"
merge_keys = ['teamID', 'sessionID', 'trialID', 'ringID']

def load_pkl(filename):
    with open(os.path.join(base_path, filename), 'rb') as f:
        return pickle.load(f)

mod_files = [
    'epoched_eeg.pkl',
    'epoched_pupil.pkl',
    'epoched_speech_event.pkl',
    'epoched_action.pkl',
]
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


# ═══════════════════════════════════════════════════════════════════════
#  EEG → TIME-FREQUENCY POWER TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════
#
#  For each ring row and each pilot (yaw/pitch/thrust):
#    1. Extract the raw EEG array  →  shape (20, 384)
#    2. Select 3 channel rows      →  shape (3, 384)   [Fz=12, Cz=18, POz=16]
#    3. For each of the 3 channels:
#         a. STFT  →  (freq_bins_full, time_frames)  complex
#         b. Power = |STFT|²        →  (freq_bins_full, time_frames)  real
#         c. Filter to delta+alpha  →  (F, time_frames)
#         d. Transpose              →  (time_frames, F)   [y=time, x=freq]
#    4. Stack 3 channels            →  (3, T, F)
#
#  Then stack all 3 pilots → (3 pilots × 3 channels, T, F) = (9, T, F)
#  But we treat each pilot separately and concatenate band-wise:
#    Each pilot: 3 channels × 2 bands = 6 maps → (6, T, F)
#  Then average across pilots at the ring level → final (6, T, F)
#
#  Implementation below merges the 3 channels × 2 bands into 6 explicitly
#  ordered as: [Fz_delta, Fz_alpha, Cz_delta, Cz_alpha, POz_delta, POz_alpha]
# ═══════════════════════════════════════════════════════════════════════

def eeg_to_tfpower(raw_20ch_384):
    """
    Parameters
    ----------
    raw_20ch_384 : np.ndarray  shape (20, 384)
        Raw EEG for one ring / one pilot.

    Returns
    -------
    tf_power : np.ndarray  shape (6, T, F)
        Time-frequency power maps for 3 channels × 2 bands.
        Axis 0 channels ordered: Fz_delta, Fz_alpha, Cz_delta, Cz_alpha,
                                  POz_delta, POz_alpha
        Axis 1 = time frames  (y-axis)
        Axis 2 = freq bins    (x-axis)
    """
    maps = []
    for ch_idx in EEG_CH_INDICES:                        # Fz, Cz, POz
        signal = raw_20ch_384[ch_idx].astype(np.float64) # (384,)
        freqs, times, Zxx = scipy_stft(
            signal, fs=SFREQ, nperseg=NPERSEG, noverlap=NOVERLAP,
            boundary="zeros", padded=True,
        )
        power = np.abs(Zxx) ** 2                          # (freq_full, T)

        for band_name, (lo, hi) in FREQ_BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            band_power = power[mask, :]                   # (F_band, T)
            # sum across the band's freq bins → (1, T), then keep as (1, T)
            # OR keep each bin separately → (F_band, T)
            # We keep each bin → concatenated F_band values across bands = F total
            maps.append(band_power.T.astype(np.float32))  # (T, F_band)

    # maps = [Fz_delta(T,Fd), Fz_alpha(T,Fa), Cz_delta, Cz_alpha, POz_delta, POz_alpha]
    # Each map may have different F_band size; concatenate along freq axis
    # then split back into per-channel-per-band maps of shape (T, F_per_band)
    # For simplicity stack as (6, T, F_per_band) assuming equal band sizes
    # If bands have unequal bin counts, concatenate on freq axis:
    combined = np.concatenate(maps, axis=1)               # (T, F_total)
    # Reshape to (6, T, F_per_band) if all bands are equal size,
    # otherwise keep as (1, T, F_total) and set in_channels=1
    # — Here we use 6 separate maps so stack individually:
    tf_stack = np.stack(maps, axis=0)                     # (6, T, F_per_band)
    # NOTE: If delta and alpha have different bin counts, F_per_band differs
    # between channels. In that case we pad to max F and stack.
    return tf_stack   # (6, T, F_per_band)  — F may differ per band


def eeg_to_tfpower_uniform(raw_20ch_384):
    """
    Same as eeg_to_tfpower but pads each band map to a uniform F width,
    then returns shape (6, T, F_max).  Ensures a rectangular tensor.
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

    # Pad to uniform F
    f_max = max(m.shape[1] for m in maps)
    padded = []
    for m in maps:
        pad_w = f_max - m.shape[1]
        padded.append(np.pad(m, ((0, 0), (0, pad_w))) if pad_w > 0 else m)

    return np.stack(padded, axis=0).astype(np.float32)     # (6, T, F_max)


# ── Apply transformation to all pilots ──────────────────────────────────────
print("\nComputing STFT time-frequency power maps...")

def pilot_to_tfpower(series):
    """Apply eeg_to_tfpower_uniform to a pandas Series of (20,384) arrays."""
    return np.stack([eeg_to_tfpower_uniform(arr) for arr in series.values])
    # returns (N_rows, 6, T, F)

eeg_yaw   = pilot_to_tfpower(df_main['yawEEG'])    # (N, 6, T, F)
eeg_pitch = pilot_to_tfpower(df_main['pitchEEG'])  # (N, 6, T, F)
eeg_thrust= pilot_to_tfpower(df_main['thrustEEG']) # (N, 6, T, F)

# Average the three pilots → (N, 6, T, F)
eeg_combined = (eeg_yaw + eeg_pitch + eeg_thrust) / 3.0
eeg_combined = eeg_combined.astype(np.float32)

# Update HPARAMS with exact runtime T and F
T_actual = eeg_combined.shape[2]
F_actual = eeg_combined.shape[3]
print(f"TF-power maps shape: {eeg_combined.shape}  (N, 6, T={T_actual}, F={F_actual})")

# Update HPARAMS if T/F changed from probe (should match)
if T_actual != HPARAMS["eeg_time_frames"] or F_actual != HPARAMS["eeg_freq_bins"]:
    print(f"  Updating HPARAMS: T {HPARAMS['eeg_time_frames']}→{T_actual}, "
          f"F {HPARAMS['eeg_freq_bins']}→{F_actual}")
    HPARAMS["eeg_time_frames"] = T_actual
    HPARAMS["eeg_freq_bins"]   = F_actual
    _eeg_flat2, _bio_flat2, _fusion2 = _compute_flat_sizes(_USER, HPARAMS)
    HPARAMS["eeg_flat"]     = _eeg_flat2
    HPARAMS["bio_flat"]     = _bio_flat2
    HPARAMS["fusion_input"] = _fusion2
    print(f"  Updated eeg_flat={_eeg_flat2}, bio_flat={_bio_flat2}, fusion={_fusion2}")

del eeg_yaw, eeg_pitch, eeg_thrust
gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  BIO SIGNAL BRANCHES  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════
def process_branch(df, cols):
    data = np.stack([np.stack(df[c].values) for c in cols], axis=1)
    if data.ndim == 2:
        data = data.reshape(len(df), 1, -1)
    return data.astype(np.float32)

act_raw = process_branch(df_main, ['yawAction',  'pitchAction',  'thrustAction'])
pup_raw = process_branch(df_main, ['yawPupil',   'pitchPupil',   'thrustPupil'])
spc_raw = process_branch(df_main, ['yawSpeech',  'pitchSpeech',  'thrustSpeech'])


# ═══════════════════════════════════════════════════════════════════════
#  STORE INTO df_main FOR TRIAL-LEVEL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════
df_main['eeg_data'] = [eeg_combined[i] for i in range(len(df_main))]
df_main['act_data'] = [act_raw[i]      for i in range(len(df_main))]
df_main['pup_data'] = [pup_raw[i]      for i in range(len(df_main))]
df_main['spc_data'] = [spc_raw[i]      for i in range(len(df_main))]

print("\n" + "="*70)
print("  AGGREGATING DATA BY TRIAL")
print("="*70)

trial_keys = ['teamID', 'sessionID', 'trialID']

def aggregate_arrays(series):
    return np.mean(list(series), axis=0)

trial_aggregated = df_main.groupby(trial_keys).agg({
    'ring_score': 'sum',
    'eeg_data':   aggregate_arrays,
    'act_data':   aggregate_arrays,
    'pup_data':   aggregate_arrays,
    'spc_data':   aggregate_arrays,
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
y_trial   = trial_aggregated['trial_total_score'].values.astype(np.float32)

del df_main, eeg_combined, act_raw, pup_raw, spc_raw
gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════
def normalize_4d(data):
    """
    Normalize (N, C, T, F) by standardising across N for each (C, T, F) position.
    Flatten N,T,F → fit scaler per C, reshape back.
    """
    N, C, T, F = data.shape
    normed = np.zeros_like(data, dtype=np.float32)
    for c in range(C):
        scaler   = StandardScaler()
        reshaped = data[:, c, :, :].reshape(N, -1)   # (N, T*F)
        normed[:, c, :, :] = scaler.fit_transform(reshaped).reshape(N, T, F)
    return normed

def normalize_3d(data):
    s, c, t = data.shape
    scaler   = StandardScaler()
    reshaped = data.transpose(0, 2, 1).reshape(-1, c)
    normed   = scaler.fit_transform(reshaped)
    return normed.reshape(s, t, c).transpose(0, 2, 1).astype(np.float32)

print("\nNormalizing features...")
eeg_norm = normalize_4d(eeg_trial)   # (N_trials, 6, T, F)
pup_norm = normalize_3d(pup_trial)

del eeg_trial, pup_trial
gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════
class FullADCTModel_TFPower(nn.Module):
    """
    Multi-branch CNN for trial score prediction.

    Branch 1  — EEG  : Conv2d on TF-power maps  (6, T, F)
    Branch 2  — Action: Conv1d
    Branch 3  — Pupil : Conv1d
    Branch 4  — Speech: Conv1d
    Fusion    — FC regression head
    """
    def __init__(self, hparams):
        super().__init__()
        d    = hparams["depth"]
        f    = hparams["filters"]
        k    = hparams["kernel_size"]
        p    = hparams["padding"]
        dr   = hparams["dropout"]
        fc_h = hparams["fc_hidden"]
        fusion = hparams["fusion_input"]

        # ── Branch 1: EEG TF-power (Conv2d) ─────────────────────────────
        # Input: (batch, 6, T, F)
        # Asymmetric pooling: MaxPool2d(kernel_size=(2,1)) pools TIME only.
        # F is too small (≈5 bins) to survive symmetric MaxPool2d(2,2) × 3 layers.
        eeg_layers = []
        in_ch = hparams["eeg_in_channels"]   # 6
        for _ in range(d):
            eeg_layers += [
                nn.Conv2d(in_ch, f, kernel_size=k, padding=p),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),   # (T//2, F unchanged)
            ]
            in_ch = f
        eeg_layers.append(nn.Flatten())
        self.eeg_branch = nn.Sequential(*eeg_layers)

        # ── Branches 2-4: Bio signals (Conv1d) ──────────────────────────
        def _bio_branch():
            layers = []
            in_ch_b = hparams["bio_in_channels"]
            for _ in range(d):
                layers += [
                    nn.Conv1d(in_ch_b, f, kernel_size=k, padding=p),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ]
                in_ch_b = f
            layers.append(nn.Flatten())
            return nn.Sequential(*layers)

        self.act_branch = _bio_branch()
        self.pup_branch = _bio_branch()
        self.spc_branch = _bio_branch()

        # ── Fusion head ──────────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(fusion, fc_h),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(fc_h, 1),
        )

    def forward(self, eeg, act, pup, spc):
        # eeg: (B, 6, T, F)  — no unsqueeze needed, already has channel dim
        b1 = self.eeg_branch(eeg)
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
    W    = 70
    SEP  = "=" * W
    SEP2 = "-" * W

    eeg_desc = []
    h, w = hparams["eeg_time_frames"], hparams["eeg_freq_bins"]
    in_ch = hparams["eeg_in_channels"]
    for i in range(d):
        h_c, w_c = _conv_out(h, k, p), _conv_out(w, k, p)
        h_p = _pool_out(h_c)   # time: halved
        w_p = w_c              # freq: conv only, no pool
        eeg_desc.append(
            f"  Layer {i+1}  Conv2d({in_ch}→{f}, k={k}, p={p}) + ReLU + MaxPool2d(2,1)")
        eeg_desc.append(f"           Output: {f} × {h_p} × {w_p}")
        h, w, in_ch = h_p, w_p, f
    eeg_flat = f * h * w

    bio_desc = []
    t = hparams["bio_timesteps"]
    in_ch = hparams["bio_in_channels"]
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
        "  FullADCTModel_TFPower_TrialScore — Architecture Summary",
        "  EEG branch: STFT Power (delta + alpha), channels Fz / Cz / POz",
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
        "  EEG PROCESSING CONFIG",
        SEP2,
        f"  {'Channels':<30} {EEG_CH_NAMES}  (idx {EEG_CH_INDICES})",
        f"  {'Freq Bands':<30} {list(FREQ_BANDS.keys())}",
        f"  {'STFT nperseg':<30} {NPERSEG}",
        f"  {'STFT noverlap':<30} {NOVERLAP}",
        f"  {'Sampling Freq':<30} {SFREQ} Hz",
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
        f"  BRANCH 1 — EEG Encoder  (Conv2d, TF-power, "
        f"in_ch={hparams['eeg_in_channels']}, depth={d})",
        SEP2,
        f"  Input Shape  {hparams['eeg_in_channels']} × "
        f"{hparams['eeg_time_frames']} × {hparams['eeg_freq_bins']}  "
        f"(channels × time_frames × freq_bins)",
        *eeg_desc,
        f"  Flatten  →  {eeg_flat:,}",
        SEP2,
        f"  BRANCHES 2-4 — Action / Pupil / Speech  (Conv1d, depth={d})",
        SEP2,
        f"  Input Shape  {hparams['bio_in_channels']} × {hparams['bio_timesteps']}",
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


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

full_dataset = TensorDataset(
    torch.tensor(eeg_norm,   dtype=torch.float32),
    torch.tensor(act_trial,  dtype=torch.float32),
    torch.tensor(pup_norm,   dtype=torch.float32),
    torch.tensor(spc_trial,  dtype=torch.float32),
    torch.tensor(y_trial,    dtype=torch.float32),
)

kf = KFold(
    n_splits=split_folds,
    shuffle=HPARAMS["kfold_shuffle"],
    random_state=HPARAMS["kfold_random_state"],
)

print(f"\nStarting {split_folds}-Fold Cross-Validation on {device}...")

if HAS_GRAPH_GEN:
    generate_model_graph(out_dir="training_logs", hparams=HPARAMS)

train_mse_list  = []; train_rmse_list = []
train_mae_list  = []; train_r2_list   = []
val_mse_list    = []; val_rmse_list   = []
val_mae_list    = []; val_r2_list     = []
total_params = trainable_params = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)))):
    print(f"\n{'='*70}")
    print(f"  FOLD {fold + 1} / {split_folds}")
    print(f"{'='*70}")

    train_sub    = Subset(full_dataset, train_idx)
    val_sub      = Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_sub, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=HPARAMS["batch_size"], shuffle=False)

    model     = FullADCTModel_TFPower(HPARAMS).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=HPARAMS["learning_rate"],
        weight_decay=HPARAMS["weight_decay"],
    )
    criterion = nn.MSELoss()

    if fold == 0:
        total_params, trainable_params = print_model_details(model, HPARAMS)
    else:
        total_params, trainable_params = count_params(model)

    # ── Early stopping state ─────────────────────────────────────────────
    best_val_mse    = float("inf")
    best_epoch      = 0
    patience_counter= 0
    best_weights    = None   # deepcopy of state_dict at best val_MSE
    best_metrics    = {}     # full metrics snapshot at best epoch

    for epoch in range(HPARAMS["epochs"]):
        # ── Train ────────────────────────────────────────────────────────
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

        # ── Validate ─────────────────────────────────────────────────────
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

        # ── Check improvement ────────────────────────────────────────────
        improved = v_mse < (best_val_mse - HPARAMS["early_stop_delta"])

        if improved:
            best_val_mse     = v_mse
            best_epoch       = epoch + 1
            patience_counter = 0
            # Save best weights (move to CPU to save GPU memory)
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = dict(
                tr_mse=tr_mse, tr_rmse=tr_rmse, tr_mae=tr_mae, tr_r2=tr_r2,
                v_mse=v_mse,   v_rmse=v_rmse,   v_mae=v_mae,   v_r2=v_r2,
            )
            # ── Log only improving epochs ─────────────────────────────────
            with open(log_filename, mode='a', newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([
                    epoch + 1,
                    f"{tr_mse:.4f}",
                    f"{v_mse:.4f}", f"{v_rmse:.4f}", f"{v_mae:.4f}", f"{v_r2:.4f}",
                    fold + 1,
                    total_params, trainable_params,
                    "improved",
                ])
            print(
                f"  Epoch {epoch+1:03d} ▲ | "
                f"Train MSE:{tr_mse:.4f} RMSE:{tr_rmse:.4f} "
                f"MAE:{tr_mae:.4f} R²:{tr_r2:.4f} | "
                f"Val  MSE:{v_mse:.4f} RMSE:{v_rmse:.4f} "
                f"MAE:{v_mae:.4f} R²:{v_r2:.4f}  ← best"
            )
        else:
            patience_counter += 1
            print(
                f"  Epoch {epoch+1:03d}   | "
                f"Train MSE:{tr_mse:.4f} | "
                f"Val  MSE:{v_mse:.4f}  "
                f"(no improvement {patience_counter}/{HPARAMS['early_stop_patience']})"
            )

        # ── Early stopping check ─────────────────────────────────────────
        if patience_counter >= HPARAMS["early_stop_patience"]:
            print(f"\n  ⏹  Early stopping triggered at epoch {epoch+1}. "
                  f"Best epoch: {best_epoch}  Val MSE: {best_val_mse:.4f}")
            break

    # ── Restore best weights & log the best row with [BEST] marker ───────
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        with open(log_filename, mode='a', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow([
                best_epoch,
                f"{best_metrics['tr_mse']:.4f}",
                f"{best_metrics['v_mse']:.4f}",
                f"{best_metrics['v_rmse']:.4f}",
                f"{best_metrics['v_mae']:.4f}",
                f"{best_metrics['v_r2']:.4f}",
                fold + 1,
                total_params, trainable_params,
                "[BEST] restored",
            ])
        print(f"  ✔  Best weights restored — Epoch {best_epoch} | "
              f"Val MSE:{best_metrics['v_mse']:.4f}  "
              f"RMSE:{best_metrics['v_rmse']:.4f}  "
              f"MAE:{best_metrics['v_mae']:.4f}  "
              f"R²:{best_metrics['v_r2']:.4f}")

    # ── Collect fold summary from best epoch (not final epoch) ───────────
    tr_mse  = best_metrics.get("tr_mse",  tr_mse)
    tr_rmse = best_metrics.get("tr_rmse", tr_rmse)
    tr_mae  = best_metrics.get("tr_mae",  tr_mae)
    tr_r2   = best_metrics.get("tr_r2",   tr_r2)
    v_mse   = best_metrics.get("v_mse",   v_mse)
    v_rmse  = best_metrics.get("v_rmse",  v_rmse)
    v_mae   = best_metrics.get("v_mae",   v_mae)
    v_r2    = best_metrics.get("v_r2",    v_r2)

    train_mse_list.append(tr_mse);   train_rmse_list.append(tr_rmse)
    train_mae_list.append(tr_mae);   train_r2_list.append(tr_r2)
    val_mse_list.append(v_mse);      val_rmse_list.append(v_rmse)
    val_mae_list.append(v_mae);      val_r2_list.append(v_r2)

    del model, optimizer, train_loader, val_loader
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════
W    = 70
SEP  = "=" * W
SEP2 = "-" * W

print(f"\n{SEP}")
print(f"  {split_folds}-FOLD CROSS-VALIDATION SUMMARY  (best epoch per fold)")
print(f"  TASK: Trial Total Score Prediction  |  EEG: TF-Power (delta+alpha)")
print(f"  Optimizer: AdamW  |  Early Stopping: patience={HPARAMS['early_stop_patience']}, "
      f"delta={HPARAMS['early_stop_delta']}")
print(f"{SEP}")
print(f"  {'Metric':<10}  {'Train Mean':>12}  {'Train SD':>10}  "
      f"{'Val Mean':>12}  {'Val SD':>10}")
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
    writer.writerow(["# CROSS-VALIDATION SUMMARY (best epoch per fold — early stopping)"])
    writer.writerow(["# TASK: Trial Total Score  |  EEG: STFT TF-Power delta+alpha"])
    writer.writerow([f"# Optimizer: AdamW  weight_decay={HPARAMS['weight_decay']}  "
                     f"patience={HPARAMS['early_stop_patience']}  "
                     f"delta={HPARAMS['early_stop_delta']}"])
    writer.writerow(["Metric",
                     "Train_Mean", "Train_SD",
                     "Val_Mean",   "Val_SD"])
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