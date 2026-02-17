import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# =================================================================
# PHASE 1: DATA VALIDATION & MULTIMODAL SYNCHRONIZATION
# Goal: Verify that raw physiological and behavioral files are 
# correctly formatted and accessible for the early prediction task.
# =================================================================

# LOADING
# extract the raw epoched data from the LIINC Laboratory OSF repository

def safe_load(file_name):
    if os.path.exists(file_name):
        print(f"✅ Successfully found: {file_name}")
        return pd.read_pickle(file_name)
    return None

pupil_df = safe_load('epoched_pupil.pkl')
action_df = safe_load('epoched_action.pkl')
performance_df = safe_load('team_performance.pkl')

# PREPROCESSING
# isolate the 'Early Window' (First 10 Seconds / 600 Samples)
# We use .select_dtypes and .astype(float) to ensure numeric integrity

if pupil_df is not None:
    try:
        # --- NEW FIX: ONLY SELECT NUMERIC COLUMNS ---
        # This ignores columns named 'T10', 'Trial_ID', etc.
        pupil_numeric = pupil_df.select_dtypes(include=[np.number])
        action_numeric = action_df.select_dtypes(include=[np.number])

        if pupil_numeric.empty:
            print("❌ The numeric data is hidden inside objects. Attempting deep extraction...")
            # If the numbers are stored inside 'objects', we force them out
            pupil_signal = np.hstack([x for x in pupil_df.iloc[0].values if not isinstance(x, str)])
            action_signal = np.hstack([x for x in action_df.iloc[0].values if not isinstance(x, str)])
        else:
            # Grab first 10 seconds (600 samples) from the numeric-only data to test the feasibility of an early warning system
            pupil_signal = pupil_numeric.iloc[0, :600].values.astype(float)
            action_signal = action_numeric.iloc[0, :600].values.astype(float)

        # Ensure we are only looking at the first 10s
        pupil_signal = pupil_signal[:600]
        action_signal = action_signal[:600]

        # 
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(pupil_signal, color='teal')
        plt.title('Verification: Pupil Size (First 10 Seconds)')
        plt.ylabel('Z-Score')

        plt.subplot(2, 1, 2)
        plt.step(range(len(action_signal)), action_signal, color='crimson')
        plt.title('Verification: Controller Actions')
        plt.ylabel('State (0/1)')
        plt.xlabel('Samples (60Hz)')

        plt.tight_layout()
        print("🚀 Success! Plotting now...")
        #comment out so that script can run in background
        #plt.show()

    except Exception as e:
        print(f"❌ Data snag: {e}")

# =================================================================
# PHASE 2 & 3: FEATURE EXTRACTION & DATA UNLOCK
# =================================================================

def extract_features_all_trials(df, prefix):
    """Loops through all trials to get 10s summary stats."""
    # Handle the 'unhashable' nesting issue we saw in the plots
    processed_rows = []
    for i in range(len(df)):
        # Extract numeric values, ignoring strings like 'T10'
        row_vals = np.hstack([x for x in df.iloc[i].values if not isinstance(x, str)])
        processed_rows.append(row_vals[:600]) # Keep first 10s (600 samples)
    
    data_matrix = np.array(processed_rows).astype(float)
    
    features = pd.DataFrame(index=df.index)
    features[f'{prefix}_mean'] = np.nanmean(data_matrix, axis=1)
    features[f'{prefix}_std'] = np.nanstd(data_matrix, axis=1)
    return features

# Generate the features
print("Summarizing 10s windows for all trials...")
pupil_features = extract_features_all_trials(pupil_df, 'pupil')
action_features = extract_features_all_trials(action_df, 'action')

# Define X (Feature Matrix)
X = pupil_features.join(action_features)
print(f"✅ Master Feature Matrix 'X' created with shape: {X.shape}")

# Calculate Performance Labels
# Count # of rings per trial from granular performance file
trial_scores = performance_df.groupby('trialID')['ringID'].count().reset_index()
trial_scores.columns = ['trialID', 'total_score']
med = trial_scores['total_score'].median()
trial_scores['performance_label'] = (trial_scores['total_score'] > med).astype(int)

# MAP LABELS TO FEATURES (Broadcasting)
# Map the trial-level performance label to EVERY 10s window in that trial
X_temp = X.copy()
X_temp['trialID'] = X.index.astype(int)

# Merge X with the trial scores so all 13,000+ rows get a label
merged_data = X_temp.merge(trial_scores[['trialID', 'performance_label']], on='trialID', how='inner')

# Define final X and y for the models
y_final = merged_data['performance_label']
X_final = merged_data.drop(['trialID', 'performance_label'], axis=1)

print(f"📊 DATA UNLOCKED!")
print(f"✅ New Feature Matrix shape: {X_final.shape}")

# DATA SPLIT & MODELS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.20, random_state=42)

print("\n🚀 Training Models on full dataset...")
# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
print(f"✅ AdaBoost Accuracy: {accuracy_score(y_test, ada.predict(X_test)):.2f}")

# XGBoost
try:
    xgb = XGBClassifier(n_estimators=100, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    print(f"✅ XGBoost Accuracy: {accuracy_score(y_test, xgb.predict(X_test)):.2f}")
except Exception as e:
    print(f"❌ XGBoost Error (check libomp): {e}")

# =================================================================
# PHASE 4: PRESENTATION VISUALS (XGBoost & AdaBoost)
# =================================================================
print("🚀 Opening Feature Importance windows...")

# AdaBoost Feature Importance
plt.figure(figsize=(10, 6))
plt.title("Physiological Dominance: Feature Importance (AdaBoost)", fontsize=12)

importances_ada = ada.feature_importances_
indices_ada = np.argsort(importances_ada)

plt.barh(range(len(indices_ada)), importances_ada[indices_ada], color='skyblue', align='center')
plt.yticks(range(len(indices_ada)), [X_final.columns[i] for i in indices_ada])
plt.xlabel('Relative Importance Score')
plt.tight_layout()
# keep window open and moves to the next
plt.show(block=False) 

# XGBoost Feature Importance
plt.figure(figsize=(10, 6))
plt.title("Physiological Dominance: Feature Importance (XGBoost)", fontsize=12)

importances_xgb = xgb.feature_importances_
indices_xgb = np.argsort(importances_xgb)

plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], color='salmon', align='center')
plt.yticks(range(len(indices_xgb)), [X_final.columns[i] for i in indices_xgb])
plt.xlabel('Relative Importance Score')
plt.tight_layout()
plt.show(block=False) # Keeps both windows open and moves to Phase 5

# =================================================================
# PHASE 5: HEAD-TO-HEAD CROSS-VALIDATION
# =================================================================
from sklearn.model_selection import cross_val_score

print("\n🛡️ Running 5-Fold Cross-Validation for BOTH models...")

# run the math for both models
ada_cv = cross_val_score(ada, X_final, y_final, cv=5)
xgb_cv = cross_val_score(xgb, X_final, y_final, cv=5)

# print both models results
print("\n" + "="*40)
print("📊 FINAL SCIENTIFIC RESULTS")
print("="*40)
print(f"○ AdaBoost Average CV:   {ada_cv.mean():.2f}")
print(f"○ XGBoost Average CV:    {xgb_cv.mean():.2f}")
print("-" * 40)
print(f"○ AdaBoost Fold Scores:  {ada_cv}")
print(f"○ XGBoost Fold Scores:   {xgb_cv}")
print("="*40)

# halt to prevent script from closing the windows automatically
print("\n✅ Analysis Complete. Results are in the terminal.")
print("✅ Close the graph windows to exit the script.")
plt.show()

# =================================================================
# PRELIMINARY STEPS EXPLAINED: DATA ENGINEERING & VALIDATION
# =================================================================
'''
o DATA VERIFICATION: 
  The project began by validating 'epoched_pupil' and 'epoched_action' 
  datasets to ensure high-frequency physiological signals and controller 
  inputs were successfully loaded and intact.

o ALIGNMENT & THE "0 vs 1" FIX: 
  A critical technical hurdle was addressed during synchronization. 
  Disparate datasets used different indexing (0-based vs. 1-based starts). 
  We manually manipulated the start points to align these sets, ensuring 
  the first 10 seconds of sensor data accurately reflects the first 10 
  seconds of mission performance.

o FEATURE EXTRACTION & MERGING: 
  Cleaned data was summarized into a master matrix. By focusing on the 
  early-stage window, we isolated cognitive load (pupil) and physical 
  behavior (actions) for predictive modeling.

o INITIAL MODELING & FEATURE IMPORTANCE: 
  AdaBoost initially outperformed XGBoost with a "lucky" 89% accuracy. 
  Feature analysis revealed 'pupil_std' as the most significant 
  predictor, proving that the stability of mental effort is more 
  telling than behavioral action counts.

o THE SANITY CHECK (CROSS-VALIDATION): 
  To ensure scientific rigor, we conducted 5-fold cross-validation. 
  This stress test revealed that XGBoost (0.69 CV) is actually more 
  consistent and reliable than AdaBoost (0.49 CV) for this specific 
  45-trial dataset.
'''

# =================================================================
# PRELIMINARY XGBoost & ADABoost CONCLUSIONS + NEXT STEPS
# =================================================================
'''
o REVISED MODEL SELECTION: 
  - XGBoost outperformed AdaBoost in 5-fold cross-validation (0.69 vs 0.49), 
    making it the more robust choice for early-stage prediction.

o INTERPRETATION:
  - AdaBoost's initial 89% accuracy was an artifact of overfitting on a 
    small test split. 
  - XGBoost demonstrated a superior ability to handle the 45-trial 
    bottleneck, maintaining a 70% success rate across diverse slices of 
    data.

o SCIENTIFIC TAKEAWAY:
  - Despite the model shift, Pupil Variance ('pupil_std') remains the 
    dominant physiological marker, justifying the "Physiological 
    Dominance" theory for team performance.

o CLASSIFIER PROBLEM APPLICATION:
  - Still a little confused about how we are going to apply the classifier portion
    to the actual problem. Such as, do we want to label teams as low, average, and high
    performance based on whether they fit into the categories of (0-5 rings, 6-10 rings, 11-15 rings)?
'''