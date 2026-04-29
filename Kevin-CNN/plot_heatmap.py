import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

base_path = "/home/kevin/data"
merge_keys = ['teamID', 'sessionID', 'trialID', 'ringID']

def load_pkl(filename):
    with open(os.path.join(base_path, filename), 'rb') as f:
        return pickle.load(f)

# Load data
print("Loading data...")
df_main = load_pkl('team_performance.pkl')[merge_keys + ['difficulty']]
eeg_df = load_pkl('epoched_eeg.pkl')

cols_to_drop = [c for c in ['difficulty', 'communication'] if c in eeg_df.columns]
if cols_to_drop:
    eeg_df = eeg_df.drop(columns=cols_to_drop)

df_main = df_main.merge(eeg_df, on=merge_keys, how='inner')

# Get the first group's first trial
# Sort to ensure we get the first team and their first trial
df_main = df_main.sort_values(by=['teamID', 'trialID', 'ringID']).reset_index(drop=True)

# Select the first row
first_row = df_main.iloc[0:1]
print(f"Selected Sample: TeamID: {first_row['teamID'].values[0]}, TrialID: {first_row['trialID'].values[0]}")

# Transform
eeg_spatial = 60
eeg_timesteps = 384

eeg_raw = np.stack([
    np.stack(first_row['yawEEG'].values),
    np.stack(first_row['pitchEEG'].values),
    np.stack(first_row['thrustEEG'].values),
], axis=1).reshape(1, eeg_spatial, eeg_timesteps).astype(np.float32)

def normalize_3d(data):
    s, c, t = data.shape
    scaler = StandardScaler()
    reshaped = data.transpose(0, 2, 1).reshape(-1, c)
    normed = scaler.fit_transform(reshaped)
    return normed.reshape(s, t, c).transpose(0, 2, 1).astype(np.float32)

eeg_norm = normalize_3d(eeg_raw)

# Extract the 2D array for the first sample
heatmap_data = eeg_norm[0]  # Shape: (60, 384)

# Plot
plt.figure(figsize=(12, 6))
plt.imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Normalized Power')
plt.title(f"Time-Frequency Power Heat Map (Team: {first_row['teamID'].values[0]}, Trial: {first_row['trialID'].values[0]})")
plt.xlabel("Timesteps")
plt.ylabel("Frequency / Spatial Dimension (60)")
plt.tight_layout()

# Save
output_path = "sample_heatmap.png"
plt.savefig(output_path)
print(f"Heatmap saved to {output_path}")

