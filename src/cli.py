"""
Command-line interface for Kaggle competition pipeline.
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.io_utils import load_data, get_target_name, save_submission
from src.features import split_columns, build_preprocessor
from src.modeling import build_pipelines, random_search_rf
from src.evaluation import evaluate_all, assert_min_accuracy
from src.feature_engineering import engineer_all_features
from src.advanced_models import (
    build_advanced_pipelines,
    tune_random_forest_extensive,
    tune_xgboost,
    tune_lightgbm,
    build_stacking_ensemble,
    build_voting_ensemble
)
from src.automl import run_tpot_optimization, compare_with_baseline


def cmd_eda(args):
    """Exploratory Data Analysis summary."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60 + "\n")
    
    # Load data
    train_df, test_df, sample_submission_df = load_data(args.data_dir)
    target_name = get_target_name(sample_submission_df)
    
    print(f"\n{'─'*60}")
    print("DATASET SHAPES")
    print(f"{'─'*60}")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    
    # Missing values
    print(f"\n{'─'*60}")
    print("MISSING VALUES (Train)")
    print(f"{'─'*60}")
    missing = train_df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing)
    else:
        print("No missing values found!")
    
    # Separate features from target
    X_train = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    
    # Column types
    numeric_cols, categorical_cols = split_columns(X_train)
    
    # Categorical cardinality
    if categorical_cols:
        print(f"\n{'─'*60}")
        print("CATEGORICAL COLUMN CARDINALITY")
        print(f"{'─'*60}")
        for col in categorical_cols:
            n_unique = train_df[col].nunique()
            print(f"{col}: {n_unique} unique values")
    
    # Numeric statistics
    if numeric_cols:
        print(f"\n{'─'*60}")
        print("NUMERIC COLUMN STATISTICS")
        print(f"{'─'*60}")
        print(train_df[numeric_cols].describe())
    
    # Target distribution
    print(f"\n{'─'*60}")
    print(f"TARGET DISTRIBUTION ({target_name})")
    print(f"{'─'*60}")
    print(train_df[target_name].value_counts().sort_index())
    print(f"\nClass balance: {train_df[target_name].value_counts(normalize=True).to_dict()}")
    
    print("\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60 + "\n")


def cmd_cv(args):
    """Cross-validation evaluation of all models."""
    print("\n" + "="*60)
    print("CROSS-VALIDATION EVALUATION")
    print("="*60 + "\n")
    
    # Load data
    train_df, test_df, sample_submission_df = load_data(args.data_dir)
    target_name = get_target_name(sample_submission_df)
    
    # Prepare features and target
    X = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y = train_df[target_name]
    
    # Build preprocessor
    numeric_cols, categorical_cols = split_columns(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    
    # Build pipelines
    pipelines = build_pipelines(preprocessor)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    results_df = evaluate_all(pipelines, X, y, cv)
    
    # Display results
    print(f"\n{'─'*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'─'*60}")
    print(results_df.to_string(index=False))
    
    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', 
                exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to '{args.output}'")
    
    # Verify output
    if os.path.exists(args.output):
        saved_df = pd.read_csv(args.output)
        assert len(saved_df) == 3, f"Expected 3 rows, got {len(saved_df)}"
        print(f"✓ Validation passed: {len(saved_df)} models evaluated")
    
    print("\n" + "="*60)
    print("CV EVALUATION COMPLETE")
    print("="*60 + "\n")


def cmd_tune(args):
    """Hyperparameter tuning for Random Forest."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING - RANDOM FOREST")
    print("="*60 + "\n")
    
    # Load data
    train_df, test_df, sample_submission_df = load_data(args.data_dir)
    target_name = get_target_name(sample_submission_df)
    
    # Prepare features and target
    X = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y = train_df[target_name]
    
    # Build preprocessor and RF pipeline
    numeric_cols, categorical_cols = split_columns(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipelines = build_pipelines(preprocessor)
    rf_pipeline = pipelines['rf']
    
    # Hyperparameter tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    best_estimator, best_score, best_params = random_search_rf(rf_pipeline, X, y, cv)
    
    # Display results
    print(f"\n{'─'*60}")
    print("TUNING RESULTS")
    print(f"{'─'*60}")
    print(f"Best RF params: {best_params}")
    print(f"Best RF CV accuracy: {best_score:.4f}")
    
    # Save best parameters
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', 
                exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n✓ Best parameters saved to '{args.output}'")
    
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60 + "\n")


def cmd_train_predict(args):
    """Train final model and generate submission."""
    print("\n" + "="*60)
    print("TRAIN & PREDICT")
    print("="*60 + "\n")
    
    # Load data
    train_df, test_df, sample_submission_df = load_data(args.data_dir)
    target_name = get_target_name(sample_submission_df)
    
    # Prepare features and target
    X_train = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y_train = train_df[target_name]
    X_test = test_df.drop(columns=['id'], errors='ignore')
    
    # Build preprocessor and pipelines
    numeric_cols, categorical_cols = split_columns(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipelines = build_pipelines(preprocessor)
    
    # Select model
    if args.model not in pipelines:
        raise ValueError(f"Invalid model '{args.model}'. Choose from: {list(pipelines.keys())}")
    
    selected_pipeline = pipelines[args.model]
    print(f"\n✓ Selected model: {args.model}")
    
    # Load best RF parameters if requested
    if args.use_best_rf and args.model == 'rf':
        best_params_path = "reports/best_rf_params.json"
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            
            # Update pipeline with best parameters
            for param, value in best_params.items():
                param_name = param.replace('clf__', '')
                setattr(selected_pipeline.named_steps['clf'], param_name, value)
            
            print(f"✓ Loaded best RF parameters from '{best_params_path}'")
        else:
            print(f"⚠ Warning: '{best_params_path}' not found. Using default parameters.")
            print("  Run 'make tune' first to generate best parameters.")
    
    # Train on full training set
    print(f"\nTraining {args.model} on full training set...")
    selected_pipeline.fit(X_train, y_train)
    print("✓ Training complete!")
    
    # Generate predictions
    print("\nGenerating predictions on test set...")
    predictions = selected_pipeline.predict(X_test)
    
    # Create submission DataFrame
    submission_df = sample_submission_df.copy()
    submission_df[target_name] = predictions
    
    # Save submission
    save_submission(submission_df, args.output)
    
    # Validation checks
    print(f"\n{'─'*60}")
    print("SUBMISSION VALIDATION")
    print(f"{'─'*60}")
    
    # Check row count
    assert len(submission_df) == len(test_df), \
        f"Row count mismatch: submission has {len(submission_df)}, test has {len(test_df)}"
    print(f"✓ Row count: {len(submission_df)} (matches test set)")
    
    # Check columns
    expected_cols = sample_submission_df.columns.tolist()
    actual_cols = submission_df.columns.tolist()
    assert actual_cols == expected_cols, \
        f"Column mismatch: expected {expected_cols}, got {actual_cols}"
    print(f"✓ Columns: {actual_cols} (matches sample submission)")
    
    print(f"\n✓ Submission validation passed: {len(submission_df)} predictions generated")
    
    print("\n" + "="*60)
    print("TRAIN & PREDICT COMPLETE")
    print("="*60 + "\n")


def cmd_tune_advanced(args):
    """Advanced hyperparameter tuning for multiple models."""
    print("\n" + "="*60)
    print("ADVANCED HYPERPARAMETER TUNING")
    print("="*60 + "\n")

    # Load data
    train_df, test_df, sample_submission_df = load_data(args.data_dir)
    target_name = get_target_name(sample_submission_df)

    # Prepare features and target
    X = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y = train_df[target_name]

    # Feature engineering if requested
    if args.engineer_features:
        print("\nApplying feature engineering...")
        X = engineer_all_features(X, y=y, create_poly=True, select_features=False)

    # Build preprocessor
    numeric_cols, categorical_cols = split_columns(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # CV strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    results = {}

    # Extensive Random Forest tuning
    if 'rf' in args.models:
        print("\n" + "="*60)
        print("EXTENSIVE RANDOM FOREST TUNING")
        print("="*60)
        pipelines = build_pipelines(preprocessor)
        rf_pipeline = pipelines['rf']
        best_rf, best_rf_score, best_rf_params = tune_random_forest_extensive(
            rf_pipeline, X, y, cv=cv, n_iter=args.n_iter, random_state=args.seed
        )
        results['rf_extensive'] = {
            'score': best_rf_score,
            'params': best_rf_params
        }

        # Save best RF parameters
        os.makedirs('reports', exist_ok=True)
        with open('reports/best_rf_extensive_params.json', 'w') as f:
            json.dump(best_rf_params, f, indent=2)

    # XGBoost tuning
    if 'xgb' in args.models:
        print("\n" + "="*60)
        print("XGBOOST TUNING")
        print("="*60)
        adv_pipelines = build_advanced_pipelines(preprocessor)
        xgb_pipeline = adv_pipelines['xgboost']
        best_xgb, best_xgb_score, best_xgb_params = tune_xgboost(
            xgb_pipeline, X, y, cv=cv, n_iter=args.n_iter, random_state=args.seed
        )
        results['xgboost'] = {
            'score': best_xgb_score,
            'params': best_xgb_params
        }

        # Save best XGB parameters
        with open('reports/best_xgb_params.json', 'w') as f:
            json.dump(best_xgb_params, f, indent=2)

    # LightGBM tuning
    if 'lgb' in args.models:
        print("\n" + "="*60)
        print("LIGHTGBM TUNING")
        print("="*60)
        adv_pipelines = build_advanced_pipelines(preprocessor)
        lgb_pipeline = adv_pipelines['lightgbm']
        best_lgb, best_lgb_score, best_lgb_params = tune_lightgbm(
            lgb_pipeline, X, y, cv=cv, n_iter=args.n_iter, random_state=args.seed
        )
        results['lightgbm'] = {
            'score': best_lgb_score,
            'params': best_lgb_params
        }

        # Save best LGB parameters
        with open('reports/best_lgb_params.json', 'w') as f:
            json.dump(best_lgb_params, f, indent=2)

    # Display summary
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    for model_name, result in sorted(results.items(), key=lambda x: x[1]['score'], reverse=True):
        print(f"{model_name}: {result['score']:.4f}")

    # Save summary
    with open('reports/advanced_tuning_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ All results saved to reports/")
    print("\n" + "="*60)
    print("ADVANCED TUNING COMPLETE")
    print("="*60 + "\n")


def cmd_tpot(args):
    """Run TPOT AutoML optimization."""
    print("\n" + "="*60)
    print("TPOT AUTOML")
    print("="*60 + "\n")

    # Load data
    train_df, test_df, sample_submission_df = load_data(args.data_dir)
    target_name = get_target_name(sample_submission_df)

    # Prepare features and target
    X = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y = train_df[target_name]

    # Feature engineering if requested
    if args.engineer_features:
        print("\nApplying feature engineering...")
        X = engineer_all_features(X, y=y, create_poly=True, select_features=False)

    # Build preprocessor and preprocess data
    numeric_cols, categorical_cols = split_columns(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_preprocessed = preprocessor.fit_transform(X)

    # Run TPOT
    best_pipeline, best_score = run_tpot_optimization(
        X_preprocessed, y,
        generations=args.generations,
        population_size=args.population_size,
        cv=5,
        random_state=args.seed,
        verbosity=2,
        n_jobs=-1,
        output_file='reports/tpot_pipeline.py'
    )

    # Compare with baseline
    if os.path.exists('reports/cv_metrics.csv'):
        baseline_df = pd.read_csv('reports/cv_metrics.csv')
        baseline_scores = dict(zip(baseline_df['model'], baseline_df['accuracy']))
        compare_with_baseline(best_score, baseline_scores)

    print("\n" + "="*60)
    print("TPOT COMPLETE")
    print("="*60 + "\n")


def cmd_ensemble(args):
    """Build and evaluate ensemble models."""
    print("\n" + "="*60)
    print("ENSEMBLE MODELS")
    print("="*60 + "\n")

    # Load data
    train_df, test_df, sample_submission_df = load_data(args.data_dir)
    target_name = get_target_name(sample_submission_df)

    # Prepare features and target
    X = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y = train_df[target_name]

    # Feature engineering if requested
    if args.engineer_features:
        print("\nApplying feature engineering...")
        X = engineer_all_features(X, y=y, create_poly=True, select_features=False)

    # Build preprocessor
    numeric_cols, categorical_cols = split_columns(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Build base models
    print("\nBuilding base models...")
    base_pipelines = build_pipelines(preprocessor)
    adv_pipelines = build_advanced_pipelines(preprocessor)

    # Combine pipelines
    all_pipelines = {**base_pipelines, **adv_pipelines}

    # Train base models
    print("\nTraining base models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for name, pipeline in all_pipelines.items():
        print(f"  Training {name}...")
        pipeline.fit(X, y)

    # Build ensembles
    print("\nBuilding ensemble models...")

    # Voting ensemble
    voting_pipeline = build_voting_ensemble(all_pipelines, preprocessor, voting='soft')

    # Stacking ensemble
    stacking_pipeline = build_stacking_ensemble(all_pipelines, preprocessor)

    # Evaluate ensembles
    print("\nEvaluating ensembles...")
    ensemble_pipelines = {
        'voting_soft': voting_pipeline,
        'stacking': stacking_pipeline
    }

    results_df = evaluate_all(ensemble_pipelines, X, y, cv)

    # Display results
    print(f"\n{'─'*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'─'*60}")
    print(results_df.to_string(index=False))

    # Save results
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/ensemble_metrics.csv', index=False)
    print(f"\n✓ Results saved to 'reports/ensemble_metrics.csv'")

    print("\n" + "="*60)
    print("ENSEMBLE COMPLETE")
    print("="*60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Kaggle Competition CLI - Startup Success Prediction"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # EDA command
    parser_eda = subparsers.add_parser('eda', help='Exploratory Data Analysis')
    parser_eda.add_argument('--data-dir', type=str, default='data',
                           help='Path to data directory (default: data)')
    parser_eda.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    parser_eda.set_defaults(func=cmd_eda)

    # CV command
    parser_cv = subparsers.add_parser('cv', help='Cross-validation evaluation')
    parser_cv.add_argument('--data-dir', type=str, default='data',
                          help='Path to data directory (default: data)')
    parser_cv.add_argument('--seed', type=int, default=42,
                          help='Random seed (default: 42)')
    parser_cv.add_argument('--output', type=str, default='reports/cv_metrics.csv',
                          help='Path to save CV results (default: reports/cv_metrics.csv)')
    parser_cv.set_defaults(func=cmd_cv)

    # Tune command
    parser_tune = subparsers.add_parser('tune', help='Hyperparameter tuning for RF')
    parser_tune.add_argument('--data-dir', type=str, default='data',
                            help='Path to data directory (default: data)')
    parser_tune.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    parser_tune.add_argument('--output', type=str, default='reports/best_rf_params.json',
                            help='Path to save best params (default: reports/best_rf_params.json)')
    parser_tune.set_defaults(func=cmd_tune)

    # Train-predict command
    parser_train = subparsers.add_parser('train-predict',
                                         help='Train model and generate submission')
    parser_train.add_argument('--data-dir', type=str, default='data',
                             help='Path to data directory (default: data)')
    parser_train.add_argument('--seed', type=int, default=42,
                             help='Random seed (default: 42)')
    parser_train.add_argument('--model', type=str, default='rf',
                             choices=['logit', 'rf', 'gb'],
                             help='Model to use (default: rf)')
    parser_train.add_argument('--use-best-rf', action='store_true',
                             help='Use tuned RF parameters from tune command')
    parser_train.add_argument('--output', type=str, default='submission.csv',
                             help='Submission file path (default: submission.csv)')
    parser_train.set_defaults(func=cmd_train_predict)

    # Tune-advanced command
    parser_tune_adv = subparsers.add_parser('tune-advanced',
                                            help='Advanced hyperparameter tuning')
    parser_tune_adv.add_argument('--data-dir', type=str, default='data',
                                help='Path to data directory (default: data)')
    parser_tune_adv.add_argument('--seed', type=int, default=42,
                                help='Random seed (default: 42)')
    parser_tune_adv.add_argument('--models', nargs='+',
                                default=['rf', 'xgb', 'lgb'],
                                choices=['rf', 'xgb', 'lgb'],
                                help='Models to tune (default: rf xgb lgb)')
    parser_tune_adv.add_argument('--n-iter', type=int, default=100,
                                help='Number of iterations for RandomizedSearchCV (default: 100)')
    parser_tune_adv.add_argument('--engineer-features', action='store_true',
                                help='Apply feature engineering before tuning')
    parser_tune_adv.set_defaults(func=cmd_tune_advanced)

    # TPOT command
    parser_tpot = subparsers.add_parser('tpot', help='Run TPOT AutoML')
    parser_tpot.add_argument('--data-dir', type=str, default='data',
                            help='Path to data directory (default: data)')
    parser_tpot.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    parser_tpot.add_argument('--generations', type=int, default=5,
                            help='Number of TPOT generations (default: 5)')
    parser_tpot.add_argument('--population-size', type=int, default=20,
                            help='TPOT population size (default: 20)')
    parser_tpot.add_argument('--engineer-features', action='store_true',
                            help='Apply feature engineering before TPOT')
    parser_tpot.set_defaults(func=cmd_tpot)

    # Ensemble command
    parser_ensemble = subparsers.add_parser('ensemble', help='Build ensemble models')
    parser_ensemble.add_argument('--data-dir', type=str, default='data',
                                help='Path to data directory (default: data)')
    parser_ensemble.add_argument('--seed', type=int, default=42,
                                help='Random seed (default: 42)')
    parser_ensemble.add_argument('--engineer-features', action='store_true',
                                help='Apply feature engineering')
    parser_ensemble.set_defaults(func=cmd_ensemble)

    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

