"""
Random Forest with GridSearchCV - Exhaustive hyperparameter search.

Based on colleagues' success with GridSearchCV, this script performs
an exhaustive search around the optimal parameter region identified
by RandomizedSearchCV.
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.io_utils import load_data, get_target_name
from src.features import split_columns, build_preprocessor

# Configuration
DATA_DIR = 'data'
SEED = 42
CV_FOLDS = 10
TARGET_CV_ACCURACY = 0.80  # Target for Kaggle ≥80%

# Parameter grid based on best RandomizedSearchCV results
# Best from model zoo: n_estimators=500, max_depth=10, min_samples_split=5,
#                      min_samples_leaf=1, max_features='log2', class_weight=None
PARAM_GRID = {
    'clf__n_estimators': [400, 500, 600, 700, 800],
    'clf__max_depth': [8, 10, 12, 15],
    'clf__min_samples_split': [2, 5, 8, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2', 0.3],
    'clf__class_weight': [None, 'balanced']
}


def calculate_grid_size(param_grid):
    """Calculate total number of parameter combinations."""
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("RANDOM FOREST WITH GRIDSEARCHCV")
    print("="*80)
    print("\nStrategy: Exhaustive search around optimal parameter region")
    print("Based on: Colleagues' success with GridSearchCV approach")
    print("="*80 + "\n")
    
    # Setup
    np.random.seed(SEED)
    os.makedirs('reports', exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    train_df, test_df, sample_submission_df = load_data(DATA_DIR)
    target_name = get_target_name(sample_submission_df)
    
    X_train = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y_train = train_df[target_name]
    X_test = test_df.drop(columns=['id'], errors='ignore')
    test_ids = test_df['id'].values
    
    print(f"✓ Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    print(f"  Features: {X_train.shape[1]} (original features - Config A)")
    
    # Build preprocessor
    print("\n🔧 Building preprocessing pipeline...")
    numeric_cols, categorical_cols = split_columns(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    print(f"✓ Preprocessor built:")
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Categorical features: {len(categorical_cols)}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=SEED, n_jobs=-1))
    ])
    
    # Display parameter grid
    print("\n📋 Parameter Grid:")
    total_combinations = calculate_grid_size(PARAM_GRID)
    print(f"  Total combinations: {total_combinations}")
    print(f"  Estimated time: {total_combinations * CV_FOLDS * 2 / 60:.0f}-{total_combinations * CV_FOLDS * 5 / 60:.0f} minutes")
    print()
    for param, values in PARAM_GRID.items():
        param_name = param.replace('clf__', '')
        print(f"  {param_name}: {values}")
    
    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    print(f"\n🔄 Cross-validation: {CV_FOLDS}-fold Stratified K-Fold")
    
    # GridSearchCV
    print("\n" + "="*80)
    print("STARTING GRIDSEARCHCV")
    print("="*80)
    print(f"\nThis will evaluate all {total_combinations} parameter combinations...")
    print("Progress will be shown for each fold.\n")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=2,  # Show progress
        return_train_score=False
    )
    
    print("Training started...\n")
    grid_search.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("GRIDSEARCHCV COMPLETE")
    print("="*80)
    print(f"\nTotal training time: {training_time/60:.1f} minutes")
    
    # Get best results
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    print("\n" + "="*80)
    print("BEST PARAMETERS FOUND")
    print("="*80)
    print()
    for param, value in best_params.items():
        param_name = param.replace('clf__', '')
        print(f"  {param_name}: {value}")
    
    print(f"\n📊 Best CV Accuracy: {best_cv_score:.4f}")
    
    # Compare to target
    if best_cv_score >= TARGET_CV_ACCURACY:
        print(f"✓ MEETS TARGET! ({TARGET_CV_ACCURACY:.1%})")
    else:
        gap = TARGET_CV_ACCURACY - best_cv_score
        print(f"⚠️  Below target by {gap:.4f} ({gap*100:.2f} percentage points)")
    
    # Compare to previous best
    print("\n" + "="*80)
    print("COMPARISON TO PREVIOUS BEST")
    print("="*80)
    
    previous_best_cv = 0.7988  # RF × A_original from model zoo
    previous_best_kaggle = 0.7899  # submission_advanced.csv
    
    print(f"\nPrevious Best:")
    print(f"  CV Accuracy: {previous_best_cv:.4f} (79.88%)")
    print(f"  Kaggle Accuracy: {previous_best_kaggle:.4f} (78.99%)")
    
    print(f"\nGridSearchCV Result:")
    print(f"  CV Accuracy: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")
    
    improvement_cv = best_cv_score - previous_best_cv
    if improvement_cv > 0:
        print(f"  ✓ Improvement: +{improvement_cv:.4f} (+{improvement_cv*100:.2f} percentage points)")
    elif improvement_cv < 0:
        print(f"  ⚠️  Decrease: {improvement_cv:.4f} ({improvement_cv*100:.2f} percentage points)")
    else:
        print(f"  = Same performance")
    
    # Expected Kaggle score (accounting for ~1.5% gap)
    expected_kaggle = best_cv_score - 0.015
    print(f"\nExpected Kaggle Accuracy: {expected_kaggle:.4f} ({expected_kaggle*100:.2f}%)")
    print(f"  (Accounting for ~1.5% CV-to-Kaggle gap)")
    
    if expected_kaggle >= 0.80:
        print(f"  ✓ Expected to meet 80% Kaggle target!")
    else:
        gap_to_target = 0.80 - expected_kaggle
        print(f"  ⚠️  Expected to be {gap_to_target:.4f} below 80% target")
    
    # Detailed cross-validation with best model
    print("\n" + "="*80)
    print("DETAILED CROSS-VALIDATION")
    print("="*80)
    print("\nEvaluating best model with multiple metrics...\n")
    
    cv_results = cross_validate(
        best_model, X_train, y_train,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        n_jobs=-1,
        return_train_score=False
    )
    
    cv_accuracy = cv_results['test_accuracy'].mean()
    cv_accuracy_std = cv_results['test_accuracy'].std()
    cv_precision = cv_results['test_precision'].mean()
    cv_recall = cv_results['test_recall'].mean()
    cv_f1 = cv_results['test_f1'].mean()
    
    print(f"CV Accuracy:  {cv_accuracy:.4f} ± {cv_accuracy_std:.4f}")
    print(f"CV Precision: {cv_precision:.4f}")
    print(f"CV Recall:    {cv_recall:.4f}")
    print(f"CV F1 Score:  {cv_f1:.4f}")
    
    # Save best parameters
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    params_file = 'reports/best_rf_gridsearch_params.json'
    with open(params_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_params = {}
        for k, v in best_params.items():
            if isinstance(v, (np.integer, np.floating)):
                serializable_params[k] = v.item()
            else:
                serializable_params[k] = v
        
        result_data = {
            'best_params': serializable_params,
            'cv_accuracy': float(cv_accuracy),
            'cv_accuracy_std': float(cv_accuracy_std),
            'cv_precision': float(cv_precision),
            'cv_recall': float(cv_recall),
            'cv_f1': float(cv_f1),
            'training_time_minutes': training_time / 60,
            'total_combinations_tested': total_combinations,
            'cv_folds': CV_FOLDS
        }
        json.dump(result_data, f, indent=2)
    
    print(f"\n✓ Best parameters saved: {params_file}")
    
    # Generate predictions
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    print("\nTraining best model on full training set...")
    best_model.fit(X_train, y_train)
    
    print("Generating predictions for test set...")
    predictions = best_model.predict(X_test)
    
    # Create submission
    submission_df = pd.DataFrame({
        'id': test_ids,
        'labels': predictions
    })
    
    submission_file = 'submission_rf_gridsearch.csv'
    submission_df.to_csv(submission_file, index=False)
    
    print(f"\n✓ Submission saved: {submission_file}")
    
    # Submission statistics
    n_success = (predictions == 1).sum()
    n_failure = (predictions == 0).sum()
    print(f"\nPrediction distribution:")
    print(f"  Success (1): {n_success} ({n_success/len(predictions)*100:.1f}%)")
    print(f"  Failure (0): {n_failure} ({n_failure/len(predictions)*100:.1f}%)")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\n✓ GridSearchCV completed successfully")
    print(f"✓ Evaluated {total_combinations} parameter combinations")
    print(f"✓ Best CV Accuracy: {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")
    print(f"✓ Expected Kaggle: {expected_kaggle:.4f} ({expected_kaggle*100:.2f}%)")
    print(f"✓ Submission file: {submission_file}")
    
    print("\n📊 Performance vs Previous Best:")
    print(f"  Previous: 79.88% CV → 78.99% Kaggle")
    print(f"  GridSearch: {cv_accuracy*100:.2f}% CV → {expected_kaggle*100:.2f}% Kaggle (expected)")
    
    if improvement_cv > 0:
        print(f"  ✓ Improvement: +{improvement_cv*100:.2f} percentage points")
    else:
        print(f"  ⚠️  No improvement over previous best")
    
    print("\n🎯 Recommendation:")
    if expected_kaggle >= 0.80:
        print("  Upload submission_rf_gridsearch.csv to Kaggle immediately!")
        print("  This is expected to meet the 80% target!")
    elif cv_accuracy > previous_best_cv:
        print("  Upload submission_rf_gridsearch.csv to Kaggle")
        print("  This shows improvement over previous best")
    else:
        print("  GridSearchCV did not improve over RandomizedSearchCV")
        print("  Consider using previous best submission (submission_advanced.csv)")
    
    print("\n" + "="*80)
    print("✓ GRIDSEARCHCV COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

