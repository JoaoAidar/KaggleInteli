"""
Stacking Ensemble for Startup Success Prediction
Phase 1, Task 1.2: Create stacking ensemble with top 5 models

Expected improvement: +0.5-1.5pp → 82-83% Kaggle
"""

import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.io_utils import load_data, get_target_name
from src.features import build_preprocessor, split_columns

print("=" * 80)
print("STACKING ENSEMBLE - PHASE 1, TASK 1.2")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
# BASE LEARNERS (Top 5 Models from Model Zoo)
# ============================================================================

print("=" * 80)
print("DEFINING BASE LEARNERS")
print("=" * 80)
print()

# Base Learner 1: Random Forest (Original Features) - Best CV: 79.88%
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
print(f"   Parameters: {rf_original_params}")
print(f"   Expected CV: 79.88%")
print()

# Base Learner 2: Random Forest (Polynomial Features) - CV: 79.42%
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
print(f"   Parameters: {rf_poly_params}")
print(f"   Expected CV: 79.42%")
print()

# Base Learner 3: XGBoost (Original Features) - CV: 79.26%
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
print(f"   Parameters: {xgb_original_params}")
print(f"   Expected CV: 79.26%")
print()

# Base Learner 4: Extra Trees - High diversity
print("4. Extra Trees (High Diversity)")
et_params = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}
print(f"   Parameters: {et_params}")
print(f"   Expected CV: ~78-79%")
print()

# Base Learner 5: LightGBM - Fast and effective
print("5. LightGBM (Fast & Effective)")
lgbm_params = {
    'n_estimators': 300,
    'max_depth': 10,
    'learning_rate': 0.05,
    'num_leaves': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
print(f"   Parameters: {lgbm_params}")
print(f"   Expected CV: ~78-79%")
print()

# ============================================================================
# CREATE BASE LEARNERS
# ============================================================================

print("Creating base learner pipelines...")
from sklearn.pipeline import Pipeline

base_learners = [
    ('rf_original', Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(**rf_original_params))
    ])),
    ('rf_poly', Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(**rf_poly_params))
    ])),
    ('xgb_original', Pipeline([
        ('preprocessor', preprocessor),
        ('clf', XGBClassifier(**xgb_original_params))
    ])),
    ('extra_trees', Pipeline([
        ('preprocessor', preprocessor),
        ('clf', ExtraTreesClassifier(**et_params))
    ])),
    ('lightgbm', Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LGBMClassifier(**lgbm_params))
    ]))
]

print(f"✓ Created {len(base_learners)} base learners")
print()

# ============================================================================
# TEST MULTIPLE META-LEARNERS
# ============================================================================

print("=" * 80)
print("TESTING META-LEARNERS")
print("=" * 80)
print()

# Cross-validation setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1']

meta_learners = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, 
                             random_state=42, eval_metric='logloss', n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                               random_state=42, verbose=-1, n_jobs=-1)
}

results = {}
best_meta_learner = None
best_cv_accuracy = 0

for meta_name, meta_clf in meta_learners.items():
    print(f"\nTesting Meta-Learner: {meta_name}")
    print("-" * 60)
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_clf,
        cv=5,  # Internal CV for generating meta-features
        n_jobs=-1,
        verbose=0
    )
    
    # Evaluate with cross-validation
    print(f"Running 10-fold cross-validation...")
    start_time = datetime.now()
    
    cv_results = cross_validate(
        stacking_clf, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Calculate metrics
    cv_accuracy = cv_results['test_accuracy'].mean()
    cv_std = cv_results['test_accuracy'].std()
    cv_precision = cv_results['test_precision'].mean()
    cv_recall = cv_results['test_recall'].mean()
    cv_f1 = cv_results['test_f1'].mean()
    
    results[meta_name] = {
        'cv_accuracy': cv_accuracy,
        'cv_std': cv_std,
        'cv_precision': cv_precision,
        'cv_recall': cv_recall,
        'cv_f1': cv_f1,
        'training_time': elapsed
    }
    
    print(f"✓ CV Accuracy: {cv_accuracy:.4f} ± {cv_std:.4f}")
    print(f"  CV Precision: {cv_precision:.4f}")
    print(f"  CV Recall: {cv_recall:.4f}")
    print(f"  CV F1-Score: {cv_f1:.4f}")
    print(f"  Training time: {elapsed:.1f}s")
    
    # Track best meta-learner
    if cv_accuracy > best_cv_accuracy:
        best_cv_accuracy = cv_accuracy
        best_meta_learner = meta_name

print()
print("=" * 80)
print("META-LEARNER COMPARISON")
print("=" * 80)
print()

# Display results table
print(f"{'Meta-Learner':<20} {'CV Accuracy':<15} {'CV Std':<12} {'F1-Score':<12} {'Time (s)':<10}")
print("-" * 80)
for meta_name, metrics in results.items():
    marker = " ← BEST" if meta_name == best_meta_learner else ""
    print(f"{meta_name:<20} {metrics['cv_accuracy']:.4f} ({metrics['cv_accuracy']*100:.2f}%)  "
          f"{metrics['cv_std']:.4f}      {metrics['cv_f1']:.4f}      "
          f"{metrics['training_time']:.1f}{marker}")

print()
print(f"🏆 Best Meta-Learner: {best_meta_learner}")
print(f"   CV Accuracy: {best_cv_accuracy:.4f} ({best_cv_accuracy*100:.2f}%)")
print()

# ============================================================================
# TRAIN FINAL MODEL WITH BEST META-LEARNER
# ============================================================================

print("=" * 80)
print("TRAINING FINAL STACKING ENSEMBLE")
print("=" * 80)
print()

print(f"Using meta-learner: {best_meta_learner}")
print("Training on full training set...")

final_stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learners[best_meta_learner],
    cv=5,
    n_jobs=-1,
    verbose=0
)

start_time = datetime.now()
final_stacking_clf.fit(X_train, y_train)
training_time = (datetime.now() - start_time).total_seconds()

print(f"✓ Training complete in {training_time:.1f}s")
print()

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("Generating predictions on test set...")
predictions = final_stacking_clf.predict(X_test)

# Analyze predictions
n_success = (predictions == 1).sum()
n_failure = (predictions == 0).sum()
pct_success = n_success / len(predictions) * 100
pct_failure = n_failure / len(predictions) * 100

print(f"✓ Predictions generated")
print(f"  Success (1): {n_success} ({pct_success:.1f}%)")
print(f"  Failure (0): {n_failure} ({pct_failure:.1f}%)")
print()

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("Saving submission file...")
submission_file = 'submission_stacking.csv'

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

print("Saving results...")
results_data = {
    'best_meta_learner': best_meta_learner,
    'cv_accuracy': float(best_cv_accuracy),
    'cv_std': float(results[best_meta_learner]['cv_std']),
    'cv_precision': float(results[best_meta_learner]['cv_precision']),
    'cv_recall': float(results[best_meta_learner]['cv_recall']),
    'cv_f1': float(results[best_meta_learner]['cv_f1']),
    'training_time': float(training_time),
    'all_meta_learners': {
        name: {k: float(v) for k, v in metrics.items()}
        for name, metrics in results.items()
    },
    'base_learners': [name for name, _ in base_learners],
    'n_base_learners': len(base_learners),
    'prediction_distribution': {
        'success': int(n_success),
        'failure': int(n_failure),
        'pct_success': float(pct_success),
        'pct_failure': float(pct_failure)
    },
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('reports/stacking_ensemble_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print("✓ Results saved: reports/stacking_ensemble_results.json")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("STACKING ENSEMBLE - SUMMARY")
print("=" * 80)
print()
print(f"Base Learners: {len(base_learners)}")
for name, _ in base_learners:
    print(f"  - {name}")
print()
print(f"Best Meta-Learner: {best_meta_learner}")
print(f"CV Accuracy: {best_cv_accuracy:.4f} ({best_cv_accuracy*100:.2f}%)")
print(f"CV Std Dev: {results[best_meta_learner]['cv_std']:.4f}")
print(f"CV F1-Score: {results[best_meta_learner]['cv_f1']:.4f}")
print()
print(f"Submission File: {submission_file}")
print(f"Predictions: {len(predictions)}")
print()
print("Expected Kaggle Performance:")
print(f"  Optimistic: {best_cv_accuracy*100 + 0.5:.2f}% (small positive gap)")
print(f"  Realistic: {best_cv_accuracy*100 - 0.5:.2f}% (small negative gap)")
print(f"  Pessimistic: {best_cv_accuracy*100 - 1.5:.2f}% (larger negative gap)")
print()
print("Comparison to Current Best:")
print(f"  Current best: 81.88% (majority_vote)")
print(f"  Expected stacking: ~{best_cv_accuracy*100:.1f}%")
if best_cv_accuracy * 100 > 81.88:
    print(f"  Expected improvement: +{best_cv_accuracy*100 - 81.88:.2f}pp ✓")
else:
    print(f"  Expected difference: {best_cv_accuracy*100 - 81.88:.2f}pp")
print()
print("=" * 80)
print("NEXT STEP: Upload submission_stacking.csv to Kaggle")
print("=" * 80)
print()
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

