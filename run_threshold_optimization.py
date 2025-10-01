"""
Threshold Optimization for Majority Vote Ensemble
Phase 1, Task 1.3: Optimize classification threshold to maximize accuracy

Target: 82-83% Kaggle (current best: 81.88%)
Expected improvement: +0.12-1.12pp
"""

import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.io_utils import load_data, get_target_name
from src.features import build_preprocessor, split_columns

print("=" * 80)
print("THRESHOLD OPTIMIZATION - PHASE 1, TASK 1.3")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("Objective: Find optimal threshold for majority_vote ensemble")
print("Current best: 81.88% Kaggle (majority_vote with default 0.5 threshold)")
print("Target: 82-83% Kaggle (+0.12-1.12pp improvement)")
print()

# Load data
print("Loading data...")
train_df, test_df, sample_submission_df = load_data('data')
target_name = get_target_name(sample_submission_df)

X_train = train_df.drop(columns=[target_name, 'id'], errors='ignore')
y_train = train_df[target_name]
X_test = test_df.drop(columns=['id'], errors='ignore')
test_ids = test_df['id'].values

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
print()

# Create preprocessor
print("Creating preprocessor...")
numeric_cols, categorical_cols = split_columns(X_train)
preprocessor = build_preprocessor(numeric_cols, categorical_cols)
print(f"✓ Preprocessor created: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
print()

# ============================================================================
# RECREATE MAJORITY VOTE BASE MODELS
# ============================================================================

print("=" * 80)
print("RECREATING MAJORITY VOTE BASE MODELS")
print("=" * 80)
print()

# Model 1: RF_Original (79.88% CV)
print("1. Random Forest (Original Features)")
rf_original_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'log2',
    'class_weight': None,
    'random_state': 42,
    'n_jobs': -1
}
rf_original = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(**rf_original_params))
])
print(f"   Parameters: n_estimators={rf_original_params['n_estimators']}, max_depth={rf_original_params['max_depth']}")
print()

# Model 2: RF_Poly (79.42% CV)
print("2. Random Forest (Polynomial Features)")
rf_poly_params = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'max_features': 0.3,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}
rf_poly = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(**rf_poly_params))
])
print(f"   Parameters: n_estimators={rf_poly_params['n_estimators']}, max_depth={rf_poly_params['max_depth']}")
print()

# Model 3: XGBoost_Original (79.26% CV)
print("3. XGBoost (Original Features)")
xgb_original_params = {
    'n_estimators': 136,
    'max_depth': 8,
    'learning_rate': 0.094,
    'subsample': 0.987,
    'colsample_bytree': 0.998,
    'gamma': 0.235,
    'min_child_weight': 3,
    'reg_alpha': 0.928,
    'reg_lambda': 0.428,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}
xgb_original = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', XGBClassifier(**xgb_original_params))
])
print(f"   Parameters: n_estimators={xgb_original_params['n_estimators']}, max_depth={xgb_original_params['max_depth']}")
print()

base_models = [
    ('rf_original', rf_original),
    ('rf_poly', rf_poly),
    ('xgb_original', xgb_original)
]

print(f"✓ Created {len(base_models)} base models")
print()

# ============================================================================
# THRESHOLD OPTIMIZATION WITH CROSS-VALIDATION
# ============================================================================

print("=" * 80)
print("THRESHOLD OPTIMIZATION")
print("=" * 80)
print()

# Thresholds to test
thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
print(f"Testing {len(thresholds)} thresholds: {thresholds}")
print()

# Cross-validation setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print("Cross-validation: 10-fold Stratified K-Fold")
print()

# Store results for each threshold
threshold_results = {}

print("Starting threshold optimization...")
print("-" * 80)

for threshold in thresholds:
    print(f"\nTesting threshold: {threshold:.2f}")
    print("-" * 60)
    
    fold_accuracies = []
    
    # Cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train all base models and get predictions
        model_predictions = []
        
        for model_name, model in base_models:
            # Train model
            model.fit(X_fold_train, y_fold_train)
            
            # Get probability predictions
            proba = model.predict_proba(X_fold_val)[:, 1]
            
            # Apply threshold
            pred = (proba >= threshold).astype(int)
            model_predictions.append(pred)
        
        # Majority vote
        model_predictions = np.array(model_predictions)
        majority_vote = (model_predictions.sum(axis=0) >= 2).astype(int)
        
        # Calculate accuracy
        accuracy = (majority_vote == y_fold_val.values).mean()
        fold_accuracies.append(accuracy)
        
        if fold_idx <= 3 or fold_idx == 10:  # Show first 3 and last fold
            print(f"  Fold {fold_idx:2d}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        elif fold_idx == 4:
            print(f"  ...")
    
    # Calculate mean and std
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    threshold_results[threshold] = {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'fold_accuracies': fold_accuracies
    }
    
    print(f"\n  Mean CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f} ({mean_accuracy*100:.2f}%)")

print()
print("=" * 80)
print("THRESHOLD OPTIMIZATION RESULTS")
print("=" * 80)
print()

# Display results table
print(f"{'Threshold':<12} {'CV Accuracy':<15} {'CV Std':<12} {'vs 0.50':<12}")
print("-" * 80)

baseline_accuracy = threshold_results[0.50]['mean_accuracy']

for threshold in thresholds:
    result = threshold_results[threshold]
    mean_acc = result['mean_accuracy']
    std_acc = result['std_accuracy']
    diff = mean_acc - baseline_accuracy
    
    marker = ""
    if threshold == 0.50:
        marker = " ← BASELINE"
    elif mean_acc == max(r['mean_accuracy'] for r in threshold_results.values()):
        marker = " ← BEST"
    
    print(f"{threshold:<12.2f} {mean_acc:.4f} ({mean_acc*100:.2f}%)  {std_acc:.4f}      "
          f"{diff:+.4f} ({diff*100:+.2f}pp){marker}")

print()

# Find optimal threshold
optimal_threshold = max(threshold_results.keys(), 
                       key=lambda t: threshold_results[t]['mean_accuracy'])
optimal_accuracy = threshold_results[optimal_threshold]['mean_accuracy']
optimal_std = threshold_results[optimal_threshold]['std_accuracy']

print(f"🏆 Optimal Threshold: {optimal_threshold:.2f}")
print(f"   CV Accuracy: {optimal_accuracy:.4f} ± {optimal_std:.4f} ({optimal_accuracy*100:.2f}%)")
print(f"   Improvement over 0.50: {(optimal_accuracy - baseline_accuracy)*100:+.2f}pp")
print()

# ============================================================================
# TRAIN FINAL MODELS AND GENERATE PREDICTIONS
# ============================================================================

print("=" * 80)
print("TRAINING FINAL MODELS WITH OPTIMAL THRESHOLD")
print("=" * 80)
print()

print(f"Using threshold: {optimal_threshold:.2f}")
print("Training on full training set...")
print()

# Train all base models on full training set
trained_models = []
for model_name, model in base_models:
    print(f"Training {model_name}...")
    start_time = datetime.now()
    model.fit(X_train, y_train)
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  ✓ Complete in {elapsed:.1f}s")
    trained_models.append((model_name, model))

print()
print("Generating predictions on test set...")

# Get predictions from each model
model_test_predictions = []
for model_name, model in trained_models:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= optimal_threshold).astype(int)
    model_test_predictions.append(pred)
    
    n_success = (pred == 1).sum()
    print(f"  {model_name}: {n_success} success ({n_success/len(pred)*100:.1f}%)")

# Majority vote
model_test_predictions = np.array(model_test_predictions)
final_predictions = (model_test_predictions.sum(axis=0) >= 2).astype(int)

n_success = (final_predictions == 1).sum()
n_failure = (final_predictions == 0).sum()

print()
print(f"✓ Final predictions (majority vote):")
print(f"  Success (1): {n_success} ({n_success/len(final_predictions)*100:.1f}%)")
print(f"  Failure (0): {n_failure} ({n_failure/len(final_predictions)*100:.1f}%)")
print()

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("Saving submission file...")
submission_file = 'submission_threshold_optimized.csv'

submission_df = pd.DataFrame({
    'id': test_ids,
    'labels': final_predictions
})
submission_df.to_csv(submission_file, index=False)

print(f"✓ Submission saved: {submission_file}")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("Saving results...")
results_data = {
    'optimal_threshold': float(optimal_threshold),
    'baseline_threshold': 0.50,
    'cv_accuracy': float(optimal_accuracy),
    'cv_std': float(optimal_std),
    'baseline_cv_accuracy': float(baseline_accuracy),
    'improvement_over_baseline': float(optimal_accuracy - baseline_accuracy),
    'all_thresholds': {
        str(t): {
            'mean_accuracy': float(r['mean_accuracy']),
            'std_accuracy': float(r['std_accuracy'])
        }
        for t, r in threshold_results.items()
    },
    'base_models': [name for name, _ in base_models],
    'prediction_distribution': {
        'success': int(n_success),
        'failure': int(n_failure),
        'pct_success': float(n_success / len(final_predictions) * 100),
        'pct_failure': float(n_failure / len(final_predictions) * 100)
    },
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('reports/threshold_optimization_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print("✓ Results saved: reports/threshold_optimization_results.json")
print()

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("=" * 80)
print("THRESHOLD OPTIMIZATION - SUMMARY")
print("=" * 80)
print()

print(f"Optimal Threshold: {optimal_threshold:.2f}")
print(f"CV Accuracy: {optimal_accuracy:.4f} ({optimal_accuracy*100:.2f}%)")
print(f"CV Std Dev: {optimal_std:.4f}")
print(f"Improvement over 0.50: {(optimal_accuracy - baseline_accuracy)*100:+.2f}pp")
print()

print(f"Submission File: {submission_file}")
print(f"Predictions: {len(final_predictions)}")
print()

# Expected Kaggle performance
print("Expected Kaggle Performance:")
print("-" * 60)

# Calculate expected Kaggle based on historical gaps
# Majority vote had +2.38pp gap, but threshold optimization may have different gap
# Conservative estimate: assume similar gap to baseline (0.50 threshold)

if optimal_threshold == 0.50:
    print("⚠️  Optimal threshold is 0.50 (same as baseline)")
    print("   No improvement expected on Kaggle")
    print(f"   Expected: 81.88% (same as current best)")
else:
    # Estimate Kaggle performance
    cv_improvement = (optimal_accuracy - baseline_accuracy) * 100
    
    # Conservative: assume 50% of CV improvement translates to Kaggle
    conservative_kaggle = 81.88 + (cv_improvement * 0.5)
    
    # Optimistic: assume 100% of CV improvement translates
    optimistic_kaggle = 81.88 + cv_improvement
    
    # Realistic: assume 75% of CV improvement translates
    realistic_kaggle = 81.88 + (cv_improvement * 0.75)
    
    print(f"  Optimistic: {optimistic_kaggle:.2f}% (100% of CV improvement)")
    print(f"  Realistic: {realistic_kaggle:.2f}% (75% of CV improvement)")
    print(f"  Conservative: {conservative_kaggle:.2f}% (50% of CV improvement)")
    print()
    
    if realistic_kaggle >= 82.0:
        print("✅ LIKELY TO MEET 82% TARGET!")
    elif realistic_kaggle >= 81.9:
        print("⚠️  May marginally improve (81.9-82%)")
    else:
        print("⚠️  Unlikely to reach 82% target")

print()
print("Comparison to Current Best:")
print(f"  Current best: 81.88% (majority_vote with threshold 0.50)")
print(f"  This submission: Expected ~{81.88 + (optimal_accuracy - baseline_accuracy)*100*0.75:.2f}%")
print()

print("=" * 80)
print("NEXT STEP: Upload submission_threshold_optimized.csv to Kaggle")
print("=" * 80)
print()

print("Decision Criteria:")
print("  - If Kaggle ≥82%: SUCCESS! Target achieved, stop here")
print("  - If Kaggle 81.9-82%: Marginal success, consider stopping")
print("  - If Kaggle <81.9%: STOP - plateau confirmed at 81.88%")
print()

print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

