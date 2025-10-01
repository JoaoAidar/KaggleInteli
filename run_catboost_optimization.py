"""
CatBoost Optimization with Bayesian Search (Optuna)
Phase 1, Task 1: Optimize CatBoost for maximum performance

Target: +0.5-1.5pp improvement over 81.88%
Expected: 82.5-83.5% Kaggle
"""

import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.io_utils import load_data, get_target_name
from src.features import build_preprocessor, split_columns

print("=" * 80)
print("CATBOOST OPTIMIZATION - PHASE 1, TASK 1")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("Objective: Optimize CatBoost with Bayesian search (Optuna)")
print("Current best: 81.88% Kaggle (majority_vote)")
print("Target: 82.5-83.5% Kaggle (+0.5-1.5pp improvement)")
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

# Preprocess data once
print("Preprocessing data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"✓ Data preprocessed: {X_train_processed.shape}")
print()

# ============================================================================
# BAYESIAN OPTIMIZATION WITH OPTUNA
# ============================================================================

print("=" * 80)
print("BAYESIAN OPTIMIZATION WITH OPTUNA")
print("=" * 80)
print()

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Cross-validation: 5-fold Stratified K-Fold")
print()

# Define objective function for Optuna
def objective(trial):
    """Objective function for Optuna optimization"""
    
    # Suggest hyperparameters
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_state': 42,
        'verbose': 0,
        'thread_count': -1
    }
    
    # Create model
    model = CatBoostClassifier(**params)
    
    # Cross-validation
    scores = cross_val_score(model, X_train_processed, y_train, 
                            cv=cv, scoring='accuracy', n_jobs=1)
    
    return scores.mean()

# Create Optuna study
print("Creating Optuna study...")
print("Optimization settings:")
print("  - Trials: 150")
print("  - Sampler: TPE (Tree-structured Parzen Estimator)")
print("  - Direction: Maximize accuracy")
print()

sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)

# Optimize
print("Starting optimization...")
print("This will take 30-60 minutes...")
print()

study.optimize(objective, n_trials=150, show_progress_bar=True)

print()
print("=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
print()

# Best parameters
best_params = study.best_params
best_score = study.best_value

print("Best Parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print()
print(f"Best CV Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
print()

# ============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================

print("=" * 80)
print("TRAINING FINAL MODEL")
print("=" * 80)
print()

print("Training CatBoost with best parameters on full training set...")
start_time = datetime.now()

final_model = CatBoostClassifier(**best_params, random_state=42, verbose=0)
final_model.fit(X_train_processed, y_train)

elapsed = (datetime.now() - start_time).total_seconds()
print(f"✓ Training complete in {elapsed:.1f}s")
print()

# Generate predictions
print("Generating predictions on test set...")
predictions = final_model.predict(X_test_processed)

n_success = (predictions == 1).sum()
n_failure = (predictions == 0).sum()

print(f"✓ Predictions generated:")
print(f"  Success (1): {n_success} ({n_success/len(predictions)*100:.1f}%)")
print(f"  Failure (0): {n_failure} ({n_failure/len(predictions)*100:.1f}%)")
print()

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("Saving submission file...")
submission_file = 'submission_catboost_optimized.csv'

submission_df = pd.DataFrame({
    'id': test_ids,
    'labels': predictions
})
submission_df.to_csv(submission_file, index=False)

print(f"✓ Submission saved: {submission_file}")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("Saving optimization results...")

results_data = {
    'best_params': best_params,
    'cv_accuracy': float(best_score),
    'n_trials': len(study.trials),
    'best_trial': study.best_trial.number,
    'prediction_distribution': {
        'success': int(n_success),
        'failure': int(n_failure),
        'pct_success': float(n_success / len(predictions) * 100),
        'pct_failure': float(n_failure / len(predictions) * 100)
    },
    'training_time_seconds': elapsed,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('reports/catboost_optimization_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print("✓ Results saved: reports/catboost_optimization_results.json")
print()

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("=" * 80)
print("CATBOOST OPTIMIZATION - SUMMARY")
print("=" * 80)
print()

print(f"Best CV Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
print(f"Optimization Trials: {len(study.trials)}")
print(f"Training Time: {elapsed:.1f}s")
print()

print(f"Submission File: {submission_file}")
print(f"Predictions: {len(predictions)}")
print()

# Expected Kaggle performance
print("Expected Kaggle Performance:")
print("-" * 60)

# Conservative estimate based on historical gaps
# CatBoost is typically robust, expect small negative gap
conservative_kaggle = best_score * 100 - 1.0  # -1pp gap
optimistic_kaggle = best_score * 100 + 0.5    # +0.5pp gap
realistic_kaggle = best_score * 100 - 0.5     # -0.5pp gap

print(f"  Optimistic: {optimistic_kaggle:.2f}% (+0.5pp gap)")
print(f"  Realistic: {realistic_kaggle:.2f}% (-0.5pp gap)")
print(f"  Conservative: {conservative_kaggle:.2f}% (-1.0pp gap)")
print()

improvement_realistic = realistic_kaggle - 81.88
print(f"Expected improvement over 81.88%: {improvement_realistic:+.2f}pp")
print()

if realistic_kaggle >= 83.0:
    print("✅ LIKELY TO REACH 83%+ TARGET!")
elif realistic_kaggle >= 82.0:
    print("⚠️  May reach 82-83% (marginal improvement)")
else:
    print("⚠️  Unlikely to significantly improve over 81.88%")

print()
print("Comparison to Current Best:")
print(f"  Current best: 81.88% (majority_vote)")
print(f"  This submission: Expected ~{realistic_kaggle:.2f}%")
print()

print("=" * 80)
print("NEXT STEP: Upload submission_catboost_optimized.csv to Kaggle")
print("=" * 80)
print()

print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

