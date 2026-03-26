import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor  # Note: Use GradientBoostingRegressor if XGBoost is missing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.signal import welch

# =================================================================
# PHASE 1: Data Loading & Pre-processing
# =================================================================

def safe_load(file_name):
    if os.path.exists(file_name):
        return pd.read_pickle(file_name)
    return None

# Loading all available datasets
pupil_df = safe_load('epoched_pupil.pkl')
action_df = safe_load('epoched_action.pkl')
performance_df = safe_load('team_performance.pkl')
eeg_df = safe_load('epoched_eeg.pkl')
speech_df = safe_load('epoched_speech_event.pkl')  # NEW SPEECH DATA

# =================================================================
# PHASE 2: EEG Frequency Domain Feature Extraction
# =================================================================

def extract_eeg_freq_features(df, fs=256):
    if df is None: return pd.DataFrame()
    trial_data = []
    bands = {'theta': (4, 8), 'alpha': (8, 13)}

    for i in range(len(df)):
        row = df.iloc[i]
        trial_id = row.get('trialID', row.name)
        signal = next((v for v in row.values if isinstance(v, np.ndarray)), None)
        if signal is None: continue
        
        # Global Average (All 60 sensors combined)
        sig = signal[:, :600] if signal.shape[1] >= 600 else np.pad(signal, ((0,0),(0,600-signal.shape[1])))
        global_sig = np.nanmean(sig, axis=0) 
        
        # Fast Fourier Transform/FFT (Welch's Method), to get Power Spectral Density (PSD) for each EEG band
        # Welch's method calculates Thetha to Alpha ratio for the global signal
        freqs, psd = welch(global_sig, fs=fs, nperseg=256)
        
        row_feats = {'trialID': int(trial_id)}
        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            row_feats[f'eeg_{band}_pwr'] = np.mean(psd[idx])
            
        trial_data.append(row_feats)
    return pd.DataFrame(trial_data).groupby('trialID').mean().reset_index()

# =================================================================
# PHASE 3: Speech Feature Extraction
# =================================================================

def extract_speech_features(df):
    if df is None: return pd.DataFrame()
    trial_data = []
    for i in range(len(df)):
        row = df.iloc[i]
        trial_id = row.get('trialID', row.name)
        
        # Feature 1: Communication Frequency
        comm_status = 1 if str(row.get('communication', 'No')).lower() == 'yes' else 0
        
        # Feature 2: Speech Intensity (Aggregate activations)
        y_int = np.sum(row['yawSpeech']) if 'yawSpeech' in row else 0
        p_int = np.sum(row['pitchSpeech']) if 'pitchSpeech' in row else 0
        t_int = np.sum(row['thrustSpeech']) if 'thrustSpeech' in row else 0
        
        trial_data.append({
            'trialID': int(trial_id),
            'speech_comm': comm_status,
            'speech_intensity': y_int + p_int + t_int
        })
    return pd.DataFrame(trial_data).groupby('trialID').mean().reset_index()

# Pupil/Action Extraction
def extract_basic_features(df, prefix):
    if df is None: return pd.DataFrame()
    trial_data = []
    for i in range(len(df)):
        row = df.iloc[i]
        trial_id = row.get('trialID', row.name)
        signal = next((v for v in row.values if isinstance(v, np.ndarray)), None)
        if signal is None: continue
        flat = signal.flatten()[:600]
        trial_data.append({'trialID': int(trial_id), f'{prefix}_mean': np.nanmean(flat), f'{prefix}_std': np.nanstd(flat)})
    return pd.DataFrame(trial_data).groupby('trialID').mean().reset_index()

# =================================================================
# PHASE 4: FEATURE EXTRACTION & CONSOLIDATION
# =================================================================

print("🚀 Extracting Multi-Modal Global features...")

# extraction functions (This defines e_feat, s_feat, etc.)
e_feat = extract_eeg_freq_features(eeg_df)
s_feat = extract_speech_features(speech_df)
p_feat = extract_basic_features(pupil_df, 'pupil')
a_feat = extract_basic_features(action_df, 'action')

print("🔗 Merging Multi-Modal features...")

# begin merging with e_feat as the base, then add speech, pupil, and action features
X_final = e_feat.copy()

if not s_feat.empty:
    X_final = X_final.merge(s_feat, on='trialID', how='inner')

if not p_feat.empty:
    X_final = X_final.merge(p_feat, on='trialID', how='inner')

if not a_feat.empty:
    X_final = X_final.merge(a_feat, on='trialID', how='inner')

# merge with performance_df to create final_df for modeling
if performance_df is not None:
    def get_ring_weight(rid):
        return 1 if rid <= 5 else (2 if rid <= 10 else 3)
    
    perf_clean = performance_df.drop_duplicates(subset=['trialID', 'ringID']).copy()
    perf_clean['pts'] = perf_clean['ringID'].apply(get_ring_weight)
    y_scores = perf_clean.groupby('trialID')['pts'].sum().reset_index()
    
    # merge with X_final to create final_df for modeling
    final_df = X_final.merge(y_scores, on='trialID', how='inner')
    
    # define X_data and y_data for Phase 5
    X_data = final_df.drop(['trialID', 'pts'], axis=1)
    y_data = final_df['pts']
    
    print(f"✅ Data Prep Complete! Trials: {len(X_data)}")
    print(f"Features being used: {X_data.columns.tolist()}")
else:
    print("❌ ERROR: performance_df missing. Cannot create X_data.")

# =================================================================
# PHASE 5: FULL SCIENTIFIC EVALUATION (4 METRICS + DUAL PLOTS)
# =================================================================
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 1. DEFINE THE MODELS (This was missing or misplaced!)
print("🤖 Initializing Models...")
xgb_model = XGBRegressor(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.05, 
    subsample=0.8, 
    colsample_bytree=0.8,
    objective='reg:squarederror'
)

ada_model = AdaBoostRegressor(n_estimators=100, random_state=42)

# 2. Setup Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Run Predictions
print("📊 Calculating 4-Metric Performance for XGBoost and AdaBoost...")
y_pred_xgb = cross_val_predict(xgb_model, X_data, y_data, cv=kf)
y_pred_ada = cross_val_predict(ada_model, X_data, y_data, cv=kf)

# 4. Define the Metric Function
def calculate_all_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return [mae, mse, rmse, r2]

xgb_res = calculate_all_metrics(y_data, y_pred_xgb)
ada_res = calculate_all_metrics(y_data, y_pred_ada)

# 5. Final Standardized Output Table
print("\n" + "="*55)
print("📊 GLOBAL MULTI-MODAL MODEL: FINAL SCIENTIFIC RESULTS")
print("="*55)
print(f"{'Metric':<15} | {'XGBoost (Freq Domain)':<20} | {'AdaBoost (Freq Domain)':<20}")
print("-" * 55)

metric_names = ['MAE', 'MSE', 'RMSE', 'R2 Score']
for i, name in enumerate(metric_names):
    print(f"{name:<15} | {xgb_res[i]:<20.4f} | {ada_res[i]:<20.4f}")

print("="*55)

# 6. Visualizing Feature Importance for Both Models
print("🎨 Generating Side-by-Side Importance Graphs...")
xgb_model.fit(X_data, y_data)
ada_model.fit(X_data, y_data)

plt.figure(figsize=(16, 8))

# --- XGBoost Plot ---
plt.subplot(1, 2, 1)
imp_xgb = xgb_model.feature_importances_
ind_xgb = np.argsort(imp_xgb)
plt.title("XGBoost: Frequency Domain + Speech Importance", fontsize=12)
plt.barh(range(len(ind_xgb)), imp_xgb[ind_xgb], color='teal')
plt.yticks(range(len(ind_xgb)), [X_data.columns[i] for i in ind_xgb])
plt.xlabel("Importance Score")
plt.grid(axis='x', linestyle='--', alpha=0.7)

# --- AdaBoost Plot ---
plt.subplot(1, 2, 2)
imp_ada = ada_model.feature_importances_
ind_ada = np.argsort(imp_ada)
plt.title("AdaBoost: Frequency Domain + Speech Importance", fontsize=12)
plt.barh(range(len(ind_ada)), imp_ada[ind_ada], color='orchid')
plt.yticks(range(len(ind_ada)), [X_data.columns[i] for i in ind_ada])
plt.xlabel("Importance Score")
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.suptitle("Multi-Modal Feature Importance (Global Baseline Strategy)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()