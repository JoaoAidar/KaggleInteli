"""
Comprehensive advanced pipeline to improve Kaggle leaderboard score.

Current score: 78.26%
Target: 80%+

This script runs:
1. Feature engineering
2. Extensive hyperparameter tuning (RF, XGBoost, LightGBM)
3. Ensemble methods
4. Best model selection and submission generation
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.io_utils import load_data, get_target_name, save_submission
from src.features import split_columns, build_preprocessor
from src.modeling import build_pipelines
from src.feature_engineering import engineer_all_features
from src.advanced_models import (
    build_advanced_pipelines,
    tune_random_forest_extensive,
    tune_xgboost,
    tune_lightgbm,
    build_stacking_ensemble,
    build_voting_ensemble
)
from src.evaluation import evaluate_all


def main():
    """Run complete advanced pipeline."""
    print("\n" + "="*70)
    print("ADVANCED PIPELINE FOR KAGGLE LEADERBOARD IMPROVEMENT")
    print("="*70)
    print("\nCurrent Kaggle Score: 78.26%")
    print("Target Score: 80%+")
    print("Strategy: Feature Engineering + Advanced Models + Ensembles")
    print("="*70 + "\n")
    
    # Configuration
    DATA_DIR = 'data'
    SEED = 42
    N_ITER = 100  # Extensive tuning
    
    # Set random seeds
    np.random.seed(SEED)
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    train_df, test_df, sample_submission_df = load_data(DATA_DIR)
    target_name = get_target_name(sample_submission_df)
    
    X_train = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y_train = train_df[target_name]
    X_test = test_df.drop(columns=['id'], errors='ignore')
    
    print(f"\n✓ Data loaded:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {X_train.shape[1]}")
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    X_train_engineered = engineer_all_features(
        X_train, y=y_train, 
        create_poly=True, 
        select_features=False
    )
    X_test_engineered = engineer_all_features(
        X_test, y=None,
        create_poly=True,
        select_features=False
    )
    
    print(f"\n✓ Feature engineering complete:")
    print(f"  - Original features: {X_train.shape[1]}")
    print(f"  - Engineered features: {X_train_engineered.shape[1]}")
    print(f"  - New features created: {X_train_engineered.shape[1] - X_train.shape[1]}")
    
    # =========================================================================
    # STEP 3: Build Preprocessor
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: BUILDING PREPROCESSOR")
    print("="*70)
    
    numeric_cols, categorical_cols = split_columns(X_train_engineered)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    
    print(f"\n✓ Preprocessor built:")
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Categorical features: {len(categorical_cols)}")
    
    # =========================================================================
    # STEP 4: Extensive Hyperparameter Tuning
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: EXTENSIVE HYPERPARAMETER TUNING")
    print("="*70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tuning_results = {}
    best_models = {}
    
    # 4.1: Extensive Random Forest Tuning
    print("\n" + "-"*70)
    print("4.1: EXTENSIVE RANDOM FOREST TUNING")
    print("-"*70)
    
    pipelines = build_pipelines(preprocessor)
    rf_pipeline = pipelines['rf']
    
    best_rf, best_rf_score, best_rf_params = tune_random_forest_extensive(
        rf_pipeline, X_train_engineered, y_train, 
        cv=cv, n_iter=N_ITER, random_state=SEED
    )
    
    tuning_results['rf_extensive'] = best_rf_score
    best_models['rf_extensive'] = best_rf
    
    # Save parameters
    with open('reports/best_rf_extensive_params.json', 'w') as f:
        json.dump(best_rf_params, f, indent=2)
    
    # 4.2: XGBoost Tuning
    print("\n" + "-"*70)
    print("4.2: XGBOOST TUNING")
    print("-"*70)
    
    adv_pipelines = build_advanced_pipelines(preprocessor)
    xgb_pipeline = adv_pipelines['xgboost']
    
    best_xgb, best_xgb_score, best_xgb_params = tune_xgboost(
        xgb_pipeline, X_train_engineered, y_train,
        cv=cv, n_iter=50, random_state=SEED
    )
    
    tuning_results['xgboost'] = best_xgb_score
    best_models['xgboost'] = best_xgb
    
    # Save parameters
    with open('reports/best_xgb_params.json', 'w') as f:
        json.dump(best_xgb_params, f, indent=2)
    
    # 4.3: LightGBM Tuning
    print("\n" + "-"*70)
    print("4.3: LIGHTGBM TUNING")
    print("-"*70)
    
    lgb_pipeline = adv_pipelines['lightgbm']
    
    best_lgb, best_lgb_score, best_lgb_params = tune_lightgbm(
        lgb_pipeline, X_train_engineered, y_train,
        cv=cv, n_iter=50, random_state=SEED
    )
    
    tuning_results['lightgbm'] = best_lgb_score
    best_models['lightgbm'] = best_lgb
    
    # Save parameters
    with open('reports/best_lgb_params.json', 'w') as f:
        json.dump(best_lgb_params, f, indent=2)
    
    # =========================================================================
    # STEP 5: Ensemble Methods
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: ENSEMBLE METHODS")
    print("="*70)
    
    # 5.1: Voting Ensemble
    print("\n" + "-"*70)
    print("5.1: VOTING ENSEMBLE")
    print("-"*70)
    
    voting_pipeline = build_voting_ensemble(best_models, preprocessor, voting='soft')
    voting_results = evaluate_all({'voting_soft': voting_pipeline}, 
                                   X_train_engineered, y_train, cv)
    voting_score = voting_results['accuracy'].values[0]
    tuning_results['voting_soft'] = voting_score
    best_models['voting_soft'] = voting_pipeline
    
    print(f"\n✓ Voting Ensemble CV Accuracy: {voting_score:.4f}")
    
    # 5.2: Stacking Ensemble
    print("\n" + "-"*70)
    print("5.2: STACKING ENSEMBLE")
    print("-"*70)
    
    stacking_pipeline = build_stacking_ensemble(best_models, preprocessor)
    stacking_results = evaluate_all({'stacking': stacking_pipeline},
                                     X_train_engineered, y_train, cv)
    stacking_score = stacking_results['accuracy'].values[0]
    tuning_results['stacking'] = stacking_score
    best_models['stacking'] = stacking_pipeline
    
    print(f"\n✓ Stacking Ensemble CV Accuracy: {stacking_score:.4f}")
    
    # =========================================================================
    # STEP 6: Model Selection
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: MODEL SELECTION")
    print("="*70)
    
    # Sort models by score
    sorted_results = sorted(tuning_results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nAll Model Scores (sorted):")
    print("-"*70)
    for rank, (model_name, score) in enumerate(sorted_results, 1):
        status = "✓ EXCEEDS TARGET" if score >= 0.80 else "⚠ Below target"
        print(f"{rank}. {model_name:20s}: {score:.4f} ({score*100:.2f}%) {status}")
    
    # Select best model
    best_model_name, best_model_score = sorted_results[0]
    best_model = best_models[best_model_name]
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   CV Accuracy: {best_model_score:.4f} ({best_model_score*100:.2f}%)")
    print(f"{'='*70}")
    
    # Save summary
    summary = {
        'best_model': best_model_name,
        'best_score': best_model_score,
        'all_scores': tuning_results,
        'improvement_from_baseline': best_model_score - 0.7826,  # From Kaggle score
        'meets_target': best_model_score >= 0.80
    }
    
    with open('reports/advanced_pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # =========================================================================
    # STEP 7: Generate Submission
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: GENERATING SUBMISSION")
    print("="*70)
    
    print(f"\nTraining {best_model_name} on full training set...")
    best_model.fit(X_train_engineered, y_train)
    
    print("Generating predictions on test set...")
    predictions = best_model.predict(X_test_engineered)
    
    # Create submission
    submission_df = sample_submission_df.copy()
    submission_df[target_name] = predictions
    
    # Save submission
    submission_path = 'submission_advanced.csv'
    save_submission(submission_df, submission_path)
    
    print(f"\n✓ Submission saved to: {submission_path}")
    print(f"  - Predictions: {len(predictions)}")
    print(f"  - Class distribution: {pd.Series(predictions).value_counts().to_dict()}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\n📊 Performance Summary:")
    print(f"  - Previous Kaggle Score: 78.26%")
    print(f"  - Best CV Score: {best_model_score*100:.2f}%")
    print(f"  - Expected Improvement: +{(best_model_score - 0.7826)*100:.2f}%")
    print(f"  - Target (80%): {'✓ MET' if best_model_score >= 0.80 else '⚠ NOT MET'}")
    
    print(f"\n🏆 Best Model: {best_model_name}")
    
    print(f"\n📁 Generated Files:")
    print(f"  - {submission_path}")
    print(f"  - reports/best_rf_extensive_params.json")
    print(f"  - reports/best_xgb_params.json")
    print(f"  - reports/best_lgb_params.json")
    print(f"  - reports/advanced_pipeline_summary.json")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Upload {submission_path} to Kaggle")
    print(f"  2. Check leaderboard score")
    print(f"  3. If score < 80%, consider:")
    print(f"     - Running TPOT AutoML (make tpot-fe)")
    print(f"     - Increasing N_ITER for more extensive tuning")
    print(f"     - Trying different feature engineering strategies")
    
    print("\n" + "="*70)
    print("✓ ADVANCED PIPELINE COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

