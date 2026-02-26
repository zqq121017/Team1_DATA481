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
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available
model = FullADCTModel().to(device)

# 1. Dataset Split
full_ds = TensorDataset(torch.from_numpy(eeg_norm), torch.from_numpy(act_raw),
                        torch.from_numpy(pup_norm), torch.from_numpy(spc_raw),
                        torch.from_numpy(y_labels))
train_size = int(0.8 * len(full_ds))
test_size = len(full_ds) - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
# Add a test_loader for efficient evaluation each epoch
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=0.0005) # Slower LR for stability
criterion = nn.MSELoss()

for epoch in range(50):
    # TRAINING PHASE
    model.train()
    total_train_loss = 0
    for b_eeg, b_act, b_pup, b_spc, b_y in train_loader:
        inputs = [x.to(device) for x in [b_eeg, b_act, b_pup, b_spc]]
        target = b_y.to(device)
        
        optimizer.zero_grad()
        output = model(*inputs).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    # TEST PHASE (Evaluation)
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for b_eeg, b_act, b_pup, b_spc, b_y in test_loader:
            inputs = [x.to(device) for x in [b_eeg, b_act, b_pup, b_spc]]
            target = b_y.to(device)
            output = model(*inputs).squeeze()
            test_loss = criterion(output, target)
            total_test_loss += test_loss.item()
            
    avg_train = total_train_loss / len(train_loader)
    avg_test = total_test_loss / len(test_loader)
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f}")

# 3. Variable Importance (Error Increase Test)
model.eval()
with torch.no_grad():
    # Evaluate on full test set
    t_eeg, t_act, t_pup, t_spc, t_y = [torch.stack([x[i] for x in test_ds]).to(device) for i in range(5)]
    base_err = criterion(model(t_eeg, t_act, t_pup, t_spc).squeeze(), t_y).item()
    
    # Test Modality Importance
    results = {}
    for name, zero_idx in [("EEG", 0), ("Actions", 1), ("Pupil", 2), ("Speech", 3)]:
        inputs = [t_eeg, t_act, t_pup, t_spc]
        inputs[zero_idx] = torch.zeros_like(inputs[zero_idx])
        err = criterion(model(*inputs).squeeze(), t_y).item()
        results[name] = err - base_err

print("\n--- Variable Importance Ranking ---")
for var, imp in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{var}: {imp:.4f}")

model.eval()
