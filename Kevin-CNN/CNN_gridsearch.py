import pandas as pd
import numpy as np
import pickle
import os
import gc
import csv
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from itertools import product

# --- 1. SETTINGS & DEVICE ---
# Optimization for Longleaf CPU
os.environ["OMP_NUM_THREADS"] = "64"
torch.set_num_threads(64)
device = torch.device("cpu")

base_path = "../../data"
merge_keys = ['teamID', 'sessionID', 'trialID', 'ringID']

# --- 2. DATA LOADING & PREPROCESSING ---
def load_pkl(filename):
    with open(os.path.join(base_path, filename), 'rb') as f:
        return pickle.load(f)

print("Merging multimodal datasets...")
df_main = load_pkl('team_performance.pkl')[merge_keys + ['difficulty']]
for mod_file in ['epoched_eeg.pkl', 'epoched_pupil.pkl', 'epoched_speech_event.pkl', 'epoched_action.pkl']:
    mod_df = load_pkl(mod_file)
    cols_to_drop = [c for c in ['difficulty', 'communication'] if c in mod_df.columns]
    df_main = df_main.merge(mod_df.drop(columns=cols_to_drop), on=merge_keys, how='inner')
    del mod_df
    gc.collect()

df_main['target_score'] = df_main['difficulty'].map({'Easy': 1, 'Medium': 2, 'Hard': 3})
df_main = df_main.dropna(subset=['target_score']).reset_index(drop=True)

def process_branch(df, cols):
    data = np.stack([np.stack(df[c].values) for c in cols], axis=1)
    return data.astype(np.float32)

def normalize_3d(data):
    s, c, t = data.shape
    scaler = StandardScaler()
    reshaped = data.transpose(0, 2, 1).reshape(-1, c)
    return scaler.fit_transform(reshaped).reshape(s, t, c).transpose(0, 2, 1).astype(np.float32)

# EEG (60x384), Others (3x90)
eeg_norm = normalize_3d(np.stack([np.stack(df_main[c].values) for c in ['yawEEG','pitchEEG','thrustEEG']], axis=1).reshape(len(df_main), 60, 384))
act_raw = process_branch(df_main, ['yawAction', 'pitchAction', 'thrustAction'])
pup_norm = normalize_3d(process_branch(df_main, ['yawPupil', 'pitchPupil', 'thrustPupil']))
spc_raw = process_branch(df_main, ['yawSpeech', 'pitchSpeech', 'thrustSpeech'])
y_labels = df_main['target_score'].values.astype(np.float32)

del df_main
gc.collect()

# --- 3. DYNAMIC MULTIMODAL MODEL ---
class DynamicADCTModel(nn.Module):
    def __init__(self, eeg_filters, other_filters, depth):
        super(DynamicADCTModel, self).__init__()
        
        # Branch 1: 2D EEG Encoder
        layers = []
        in_ch = 1
        for i in range(depth):
            layers.append(nn.Conv2d(in_ch, eeg_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = eeg_filters
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Flatten())
        self.eeg_net = nn.Sequential(*layers)
        
        # Branches 2-4: 1D Behavioral/Autonomic Encoders
        def make_1d(filters, d):
            l = []
            ich = 3
            for _ in range(d):
                l.append(nn.Conv1d(ich, filters, kernel_size=3, padding=1))
                l.append(nn.ReLU())
                ich = filters
            l.append(nn.MaxPool1d(2))
            l.append(nn.Flatten())
            return nn.Sequential(*l)
            
        self.act_net = make_1d(other_filters, depth)
        self.pup_net = make_1d(other_filters, depth)
        self.spc_net = make_1d(other_filters, depth)
        
        # Dynamic Fusion Size Calculation
        self.fc = None 

    def forward(self, eeg, act, pup, spc):
        f1 = self.eeg_net(eeg.unsqueeze(1))
        f2 = self.act_net(act)
        f3 = self.pup_net(pup)
        f4 = self.spc_net(spc)
        combined = torch.cat((f1, f2, f3, f4), dim=1)
        
        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(combined.shape[1], 128),
                nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(128, 1)
            ).to(eeg.device)
            
        return self.fc(combined)

# --- 4. GRID SEARCH EXECUTION ---
# Search Space
param_grid = {
    'depth': [1, 2, 3, 4, 5, 6],
    'filters': [16, 32, 64],
    'batch_size': [32, 64],
    'lr': [0.0005, 0.001]
}
epoch_num = 50

log_file = f"training_logs/grid_search_{datetime.datetime.now().strftime('%H%M%S')}.csv"
with open(log_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Depth', 'Filters', 'BS', 'LR', 'Epoch', 'Train_Err', 'Test_Err', 'EEG_Imp', 'Act_Imp', 'Pup_Imp', 'Spc_Imp'])

keys, values = zip(*param_grid.items())
for config_vals in product(*values):
    config = dict(zip(keys, config_vals))
    print(f"\n>> TESTING: {config}")
    
    # Dataset Split
    full_ds = TensorDataset(torch.from_numpy(eeg_norm), torch.from_numpy(act_raw), 
                            torch.from_numpy(pup_norm), torch.from_numpy(spc_raw), 
                            torch.from_numpy(y_labels))
    train_ds, test_ds = random_split(full_ds, [int(0.8*len(full_ds)), len(full_ds)-int(0.8*len(full_ds))])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    model = DynamicADCTModel(config['filters'], config['filters']//2, config['depth']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # Dynamic LR Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    for epoch in range(epoch_num):
        model.train()
        train_loss = 0
        for b_eeg, b_act, b_pup, b_spc, b_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_eeg, b_act, b_pup, b_spc).squeeze(), b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for b_eeg, b_act, b_pup, b_spc, b_y in test_loader:
                test_loss += criterion(model(b_eeg, b_act, b_pup, b_spc).squeeze(), b_y).item()
        
        # Modality Importance (Ablation)    
        importances = []
        with torch.no_grad():
            t_eeg, t_act, t_pup, t_spc, t_y = [torch.stack([x[i] for x in test_ds]) for i in range(5)]
            base = criterion(model(t_eeg, t_act, t_pup, t_spc).squeeze(), t_y).item()
            for idx in range(4):
                inputs = [t_eeg, t_act, t_pup, t_spc]
                inputs[idx] = torch.zeros_like(inputs[idx])
                importances.append(criterion(model(*inputs).squeeze(), t_y).item() - base)

        with open(log_file, 'a') as f:
            csv.writer(f).writerow([config['depth'], config['filters'], config['batch_size'], config['lr'], epoch+1, 
                                    train_loss/len(train_loader), test_loss/len(test_loader)] + importances)
    
    del model, optimizer, train_loader, test_loader
    gc.collect()