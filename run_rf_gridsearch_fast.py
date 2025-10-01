"""
Random Forest with GridSearchCV - Fast version with focused grid.

Smaller parameter grid for faster execution while still being exhaustive
around the optimal region.
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
TARGET_CV_ACCURACY = 0.80

# Focused parameter grid (smaller for faster execution)
# Based on best from model zoo: n_estimators=500, max_depth=10, 
# min_samples_split=5, min_samples_leaf=1, max_features='log2'
PARAM_GRID = {
    'clf__n_estimators': [400, 500, 600],  # Reduced from 5 to 3
    'clf__max_depth': [8, 10, 12],  # Reduced from 4 to 3
    'clf__min_samples_split': [2, 5, 8],  # Reduced from 4 to 3
    'clf__min_samples_leaf': [1, 2],  # Reduced from 3 to 2
    'clf__max_features': ['log2', 0.3],  # Reduced from 3 to 2
    'clf__class_weight': [None, 'balanced']  # Keep 2
}

# Total: 3 × 3 × 3 × 2 × 2 × 2 = 216 combinations (vs 960 in full grid)


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("RANDOM FOREST WITH GRIDSEARCHCV (FAST VERSION)")
    print("="*80)
    print("\nStrategy: Focused exhaustive search around optimal parameters")
    print("Grid size: Reduced for faster execution (~15-30 minutes)")
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
    print(f"  Features: {X_train.shape[1]} (original features)")
    
    # Build preprocessor
    print("\n🔧 Building preprocessing pipeline...")
    numeric_cols, categorical_cols = split_columns(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    print(f"✓ Preprocessor built: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=SEED, n_jobs=-1))
    ])
    
    # Display parameter grid
    total_combinations = 1
    for values in PARAM_GRID.values():
        total_combinations *= len(values)
    
    print("\n📋 Parameter Grid:")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Estimated time: {total_combinations * CV_FOLDS * 1.5 / 60:.0f}-{total_combinations * CV_FOLDS * 3 / 60:.0f} minutes")
    print()
    for param, values in PARAM_GRID.items():
        print(f"  {param.replace('clf__', '')}: {values}")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    print(f"\n🔄 Cross-validation: {CV_FOLDS}-fold Stratified K-Fold")
    
    # GridSearchCV
    print("\n" + "="*80)
    print("STARTING GRIDSEARCHCV")
    print("="*80)
    print(f"\nEvaluating {total_combinations} parameter combinations...\n")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=False
    )
    
    grid_search.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("GRIDSEARCHCV COMPLETE")
    print("="*80)
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    
    # Best results
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    print("\n" + "="*80)
    print("BEST PARAMETERS")
    print("="*80)
    for param, value in best_params.items():
        print(f"  {param.replace('clf__', '')}: {value}")
    
    print(f"\n📊 Best CV Accuracy: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")
    
    # Compare to target
    if best_cv_score >= TARGET_CV_ACCURACY:
        print(f"  ✓ MEETS 80% TARGET!")
    else:
        gap = TARGET_CV_ACCURACY - best_cv_score
        print(f"  ⚠️  Below target by {gap:.4f} ({gap*100:.2f}pp)")
    
    # Detailed CV
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    
    cv_results = cross_validate(
        best_model, X_train, y_train,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        n_jobs=-1
    )
    
    cv_acc = cv_results['test_accuracy'].mean()
    cv_std = cv_results['test_accuracy'].std()
    cv_prec = cv_results['test_precision'].mean()
    cv_rec = cv_results['test_recall'].mean()
    cv_f1 = cv_results['test_f1'].mean()
    
    print(f"\nCV Accuracy:  {cv_acc:.4f} ± {cv_std:.4f}")
    print(f"CV Precision: {cv_prec:.4f}")
    print(f"CV Recall:    {cv_rec:.4f}")
    print(f"CV F1:        {cv_f1:.4f}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    prev_cv = 0.7988
    prev_kaggle = 0.7899
    expected_kaggle = cv_acc - 0.015
    
    print(f"\nPrevious Best:")
    print(f"  CV: {prev_cv:.4f} (79.88%)")
    print(f"  Kaggle: {prev_kaggle:.4f} (78.99%)")
    
    print(f"\nGridSearchCV:")
    print(f"  CV: {cv_acc:.4f} ({cv_acc*100:.2f}%)")
    print(f"  Expected Kaggle: {expected_kaggle:.4f} ({expected_kaggle*100:.2f}%)")
    
    improvement = cv_acc - prev_cv
    if improvement > 0:
        print(f"\n  ✓ Improvement: +{improvement:.4f} (+{improvement*100:.2f}pp)")
    else:
        print(f"\n  ⚠️  Change: {improvement:.4f} ({improvement*100:.2f}pp)")
    
    # Save parameters
    params_file = 'reports/best_rf_gridsearch_params.json'
    with open(params_file, 'w') as f:
        result_data = {
            'best_params': {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) 
                           for k, v in best_params.items()},
            'cv_accuracy': float(cv_acc),
            'cv_std': float(cv_std),
            'cv_precision': float(cv_prec),
            'cv_recall': float(cv_rec),
            'cv_f1': float(cv_f1),
            'training_time_minutes': training_time / 60,
            'combinations_tested': total_combinations
        }
        json.dump(result_data, f, indent=2)
    
    print(f"\n✓ Parameters saved: {params_file}")
    
    # Generate submission
    print("\n" + "="*80)
    print("GENERATING SUBMISSION")
    print("="*80)
    
    print("\nTraining on full dataset...")
    best_model.fit(X_train, y_train)
    
    print("Predicting test set...")
    predictions = best_model.predict(X_test)
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'labels': predictions
    })
    
    submission_file = 'submission_rf_gridsearch.csv'
    submission_df.to_csv(submission_file, index=False)
    
    n_success = (predictions == 1).sum()
    print(f"\n✓ Submission saved: {submission_file}")
    print(f"  Success: {n_success} ({n_success/len(predictions)*100:.1f}%)")
    print(f"  Failure: {len(predictions)-n_success} ({(len(predictions)-n_success)/len(predictions)*100:.1f}%)")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n✓ GridSearchCV completed in {training_time/60:.1f} minutes")
    print(f"✓ Best CV: {cv_acc:.4f} ({cv_acc*100:.2f}%)")
    print(f"✓ Expected Kaggle: {expected_kaggle:.4f} ({expected_kaggle*100:.2f}%)")
    
    if expected_kaggle >= 0.80:
        print(f"\n🎯 EXPECTED TO MEET 80% TARGET!")
        print(f"   Upload {submission_file} to Kaggle immediately!")
    elif improvement > 0:
        print(f"\n✓ Improved over previous best by {improvement*100:.2f}pp")
        print(f"  Upload {submission_file} to Kaggle")
    else:
        print(f"\n⚠️  No improvement over previous best")
        print(f"  Consider using submission_advanced.csv (78.99% Kaggle)")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()

