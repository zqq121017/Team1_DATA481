import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.model_selection import KFold


base_path = "../../data"
merge_keys = ['teamID', 'sessionID', 'trialID', 'ringID']

def load_pkl(filename):
    with open(os.path.join(base_path, filename), 'rb') as f:
        return pickle.load(f)

# 1. Load the "Anchor" performance and action data first
mod_files = ['epoched_eeg.pkl', 'epoched_pupil.pkl', 'epoched_speech_event.pkl', 'epoched_action.pkl']
df_main = load_pkl('team_performance.pkl')[merge_keys + ['difficulty']]

for mod_file in mod_files:
    print(f"Merging {mod_file}...")
    mod_df = load_pkl(mod_file)
    
    # DROP redundant columns that aren't keys to avoid suffix collisions
    cols_to_drop = [c for c in ['difficulty', 'communication'] if c in mod_df.columns]
    if cols_to_drop:
        mod_df = mod_df.drop(columns=cols_to_drop)
    
    # Perform the merge
    df_main = df_main.merge(mod_df, on=merge_keys, how='inner')
    
    # Clean up memory immediately
    del mod_df
    gc.collect()

# Create target scores: Easy=1, Medium=2, Hard=3
diff_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
df_main['target_score'] = df_main['difficulty'].map(diff_map)

# Filter for successful rings and reset index
df_main = df_main.dropna(subset=['target_score']).reset_index(drop=True)
print(f"Final aligned dataset size: {len(df_main)}")

# Helper to convert columns to float32 arrays
def process_branch(df, cols, expected_steps):
    data = np.stack([np.stack(df[c].values) for c in cols], axis=1)
    # Ensure shape is (Samples, Channels, TimeSteps)
    if data.ndim == 2: data = data.reshape(len(df), 1, -1)
    return data.astype(np.float32)

# EEG Branch (60 channels total, 384 time steps)
eeg_raw = np.stack([
    np.stack(df_main['yawEEG'].values),
    np.stack(df_main['pitchEEG'].values),
    np.stack(df_main['thrustEEG'].values)
], axis=1).reshape(len(df_main), 60, 384).astype(np.float32)

# Behavioral/Physiological (90 time steps)
act_raw = process_branch(df_main, ['yawAction', 'pitchAction', 'thrustAction'], 90)
pup_raw = process_branch(df_main, ['yawPupil', 'pitchPupil', 'thrustPupil'], 90)
spc_raw = process_branch(df_main, ['yawSpeech', 'pitchSpeech', 'thrustSpeech'], 90)
y_labels = df_main['target_score'].values.astype(np.float32)

# Normalize Physiological signals as done in the paper [cite: 694, 696]
def normalize_3d(data):
    s, c, t = data.shape
    scaler = StandardScaler()
    # Reshape to (Samples*Steps, Channels) to normalize per sensor
    reshaped = data.transpose(0, 2, 1).reshape(-1, c)
    normed = scaler.fit_transform(reshaped)
    return normed.reshape(s, t, c).transpose(0, 2, 1).astype(np.float32)

eeg_norm = normalize_3d(eeg_raw)
pup_norm = normalize_3d(pup_raw)

# Final memory cleanup
del df_main, eeg_raw, pup_raw
gc.collect()



class FullADCTModel(nn.Module):
    def __init__(self):
        super(FullADCTModel, self).__init__()
        
        # Branch 1: EEG Encoder (60x384)
        self.eeg_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces spatial and temporal dims by half
            nn.Flatten()
        )
        
        # Helper for 1D branches (3 channels x 90 steps)
        def conv1d_block():
            return nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten()
            )
            
        self.act_branch = conv1d_block()
        self.pup_branch = conv1d_block()
        self.spc_branch = conv1d_block()
        
        # Fusion Layer: Calculate input size dynamically
        # EEG: 16 * 30 * 192 = 92160
        # Others: 8 * 45 = 360 each
        self.fc = nn.Sequential(
            nn.Linear(92160 + (360 * 3), 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1) # Output: Continuous performance score
        )

    def forward(self, eeg, act, pup, spc):
        b1 = self.eeg_branch(eeg.unsqueeze(1))
        b2 = self.act_branch(act)
        b3 = self.pup_branch(pup)
        b4 = self.spc_branch(spc)
        
        combined = torch.cat((b1, b2, b3, b4), dim=1)
        return self.fc(combined)
    

# -------------------------------------------------------------
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.optim as optim

if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available
model = FullADCTModel().to(device)
# 1. Setup Data and Device
full_dataset = TensorDataset(
    torch.tensor(eeg_norm, dtype=torch.float32), 
    torch.tensor(act_raw, dtype=torch.float32),
    torch.tensor(pup_norm, dtype=torch.float32), 
    torch.tensor(spc_raw, dtype=torch.float32),
    torch.tensor(y_labels, dtype=torch.float32)
)

# 2. Initialize Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

print(f"Starting 5-Fold Cross-Validation on {device}...")

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)))):
    print(f"\n--- FOLD {fold + 1} ---")
    
    # Create DataLoaders for this fold
    train_sub = Subset(full_dataset, train_idx)
    val_sub = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)
    
    # Initialize fresh model and optimizer for each fold
    model = FullADCTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    
    # --- Training Phase (Example for 10 epochs per fold) ---
    for epoch in range(10):
        model.train()
        for b_eeg, b_act, b_pup, b_spc, b_y in train_loader:
            inputs = [x.to(device) for x in [b_eeg, b_act, b_pup, b_spc]]
            target = b_y.to(device).view(-1, 1) # Fix target size warning
            
            optimizer.zero_grad()
            output = model(*inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    # --- Validation Phase ---
    model.eval()
    val_errors = []
    with torch.no_grad():
        for b_eeg, b_act, b_pup, b_spc, b_y in val_loader:
            inputs = [x.to(device) for x in [b_eeg, b_act, b_pup, b_spc]]
            target = b_y.to(device).view(-1, 1)
            
            output = model(*inputs)
            val_errors.append(criterion(output, target).item())
            
    avg_val_loss = np.mean(val_errors)
    fold_results.append(avg_val_loss)
    print(f"Fold {fold + 1} Validation MSE: {avg_val_loss:.4f}")
    
    # CRITICAL: Clean up memory before starting next fold
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.mps.empty_cache() # Clear Mac GPU memory

# 3. Final Evaluation
print(f"\n--- Cross-Validation Complete ---")
print(f"Mean MSE Across 5 Folds: {np.mean(fold_results):.4f}")
print(f"Standard Deviation: {np.std(fold_results):.4f}")