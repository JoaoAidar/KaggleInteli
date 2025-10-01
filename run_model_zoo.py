"""
Comprehensive Model Zoo Evaluation Script.

Systematically evaluates all models × feature configurations to achieve ≥80% Kaggle accuracy.
"""
import sys
import os
import json
import time
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.io_utils import load_data, get_target_name, save_submission
from src.features import split_columns, build_preprocessor
from src.feature_configs import FEATURE_CONFIGS, apply_feature_config, get_config_info
from src.model_zoo import get_model_zoo, get_param_distributions, get_model_priority


# Configuration
DATA_DIR = 'data'
SEED = 42
N_ITER = 100  # RandomizedSearchCV iterations
CV_FOLDS = 10  # More conservative than 5
RESULTS_DIR = 'reports/model_zoo_results'
BEST_PARAMS_DIR = 'reports/model_zoo_best_params'

# Target CV accuracy (accounting for ~1.5% gap to Kaggle)
TARGET_CV_ACCURACY = 0.815  # To achieve ≥80% on Kaggle


def setup_directories():
    """Create necessary directories."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(BEST_PARAMS_DIR, exist_ok=True)


def save_progress(results_df, filename='progress.csv'):
    """Save intermediate results."""
    filepath = os.path.join(RESULTS_DIR, filename)
    results_df.to_csv(filepath, index=False)
    print(f"  💾 Progress saved to {filepath}")


def evaluate_model_config(model_name, config_name, model_pipeline, X_train, X_test, 
                          y_train, param_dist, cv, test_ids):
    """
    Evaluate a single model × configuration combination.
    
    Returns
    -------
    dict
        Results dictionary with metrics and metadata
    """
    result = {
        'model': model_name,
        'config': config_name,
        'status': 'pending',
        'cv_accuracy': None,
        'cv_precision': None,
        'cv_recall': None,
        'cv_f1': None,
        'cv_std': None,
        'best_params': None,
        'n_features': X_train.shape[1],
        'training_time': None,
        'error': None
    }
    
    try:
        start_time = time.time()
        
        # Hyperparameter tuning
        print(f"    🔧 Tuning hyperparameters ({N_ITER} iterations)...")
        
        random_search = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=param_dist,
            n_iter=N_ITER,
            scoring='accuracy',
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            verbose=0,
            error_score='raise'
        )
        
        random_search.fit(X_train, y_train)
        
        # Get best model
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        # Cross-validation with best model
        print(f"    📊 Cross-validating best model...")
        cv_results = cross_validate(
            best_model, X_train, y_train,
            cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            n_jobs=-1,
            return_train_score=False
        )
        
        # Calculate metrics
        cv_accuracy = cv_results['test_accuracy'].mean()
        cv_precision = cv_results['test_precision'].mean()
        cv_recall = cv_results['test_recall'].mean()
        cv_f1 = cv_results['test_f1'].mean()
        cv_std = cv_results['test_accuracy'].std()
        
        training_time = time.time() - start_time
        
        # Update result
        result.update({
            'status': 'success',
            'cv_accuracy': cv_accuracy,
            'cv_precision': cv_precision,
            'cv_recall': cv_recall,
            'cv_f1': cv_f1,
            'cv_std': cv_std,
            'best_params': best_params,
            'training_time': training_time
        })
        
        # Save best parameters
        params_file = os.path.join(BEST_PARAMS_DIR, f'{model_name}_{config_name}_params.json')
        with open(params_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_params = {}
            for k, v in best_params.items():
                if isinstance(v, (np.integer, np.floating)):
                    serializable_params[k] = v.item()
                else:
                    serializable_params[k] = v
            json.dump(serializable_params, f, indent=2)
        
        # Generate predictions if CV accuracy meets threshold
        if cv_accuracy >= TARGET_CV_ACCURACY:
            print(f"    🎯 CV accuracy {cv_accuracy:.4f} meets target! Generating submission...")
            
            # Train on full training set
            best_model.fit(X_train, y_train)
            
            # Generate predictions
            predictions = best_model.predict(X_test)
            
            # Create submission
            submission_df = pd.DataFrame({
                'id': test_ids,
                'labels': predictions
            })
            
            submission_file = f'submission_{model_name}_{config_name}.csv'
            submission_df.to_csv(submission_file, index=False)
            print(f"    ✅ Submission saved: {submission_file}")
        
        # Print summary
        print(f"    ✅ Success! CV Accuracy: {cv_accuracy:.4f} (±{cv_std:.4f})")
        print(f"       Precision: {cv_precision:.4f}, Recall: {cv_recall:.4f}, F1: {cv_f1:.4f}")
        print(f"       Training time: {training_time:.1f}s")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"    ❌ Failed: {str(e)}")
        # Print traceback for debugging
        traceback.print_exc()
    
    return result


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("MODEL ZOO: COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"\nObjective: Achieve ≥80% accuracy on Kaggle leaderboard")
    print(f"Strategy: Systematic evaluation of all models × feature configurations")
    print(f"Target CV Accuracy: ≥{TARGET_CV_ACCURACY:.1%} (to account for ~1.5% Kaggle gap)")
    print("="*80 + "\n")
    
    # Setup
    setup_directories()
    np.random.seed(SEED)
    
    # Load data
    print("📂 Loading data...")
    train_df, test_df, sample_submission_df = load_data(DATA_DIR)
    target_name = get_target_name(sample_submission_df)
    
    X_train_raw = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y_train = train_df[target_name]
    X_test_raw = test_df.drop(columns=['id'], errors='ignore')
    test_ids = test_df['id'].values
    
    print(f"✅ Data loaded: {len(X_train_raw)} train, {len(X_test_raw)} test samples")
    print(f"   Original features: {X_train_raw.shape[1]}")
    
    # Display feature configurations
    print("\n📋 Feature Configurations:")
    config_info = get_config_info()
    for _, row in config_info.iterrows():
        print(f"   {row['config_id']}: {row['name']} ({row['expected_features']} features)")
    
    # Get model priority
    model_priority = get_model_priority()
    
    # Sort models by priority (highest first)
    sorted_models = sorted(model_priority.items(), key=lambda x: x[1], reverse=True)
    model_order = [m[0] for m in sorted_models]
    
    print(f"\n🤖 Models to evaluate: {len(model_order)}")
    print("   Priority order (5=highest):")
    for model_name, priority in sorted_models:
        print(f"   [{priority}] {model_name}")
    
    # Calculate total combinations
    total_combinations = len(model_order) * len(FEATURE_CONFIGS)
    print(f"\n📊 Total combinations to evaluate: {total_combinations}")
    print(f"   Estimated time: {total_combinations * 3:.0f}-{total_combinations * 10:.0f} minutes")
    
    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    print(f"\n🔄 Cross-validation: {CV_FOLDS}-fold Stratified K-Fold")
    
    # Results storage
    all_results = []
    combination_count = 0
    
    # Start evaluation
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80 + "\n")
    
    start_time_total = time.time()
    
    # Iterate through models (by priority)
    for model_name in model_order:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Get parameter distributions
        param_dists = get_param_distributions()
        
        if model_name not in param_dists:
            print(f"⚠️  No parameter distribution defined for {model_name}, skipping...")
            continue
        
        param_dist = param_dists[model_name]
        
        # Iterate through feature configurations
        for config_name in FEATURE_CONFIGS.keys():
            combination_count += 1
            
            print(f"\n[{combination_count}/{total_combinations}] {model_name} × {config_name}")
            print(f"{'─'*80}")
            
            try:
                # Apply feature configuration
                print(f"  🔧 Applying feature configuration...")
                X_train, X_test, feature_names = apply_feature_config(
                    config_name, X_train_raw, X_test_raw, y_train
                )
                print(f"  ✅ Features: {len(feature_names)}")
                
                # Build preprocessor
                numeric_cols, categorical_cols = split_columns(X_train)
                preprocessor = build_preprocessor(numeric_cols, categorical_cols)
                
                # Get model
                models = get_model_zoo(preprocessor)
                
                if model_name not in models:
                    print(f"  ⚠️  Model {model_name} not available, skipping...")
                    continue
                
                model_pipeline = models[model_name]
                
                # Evaluate
                result = evaluate_model_config(
                    model_name, config_name, model_pipeline,
                    X_train, X_test, y_train,
                    param_dist, cv, test_ids
                )
                
                all_results.append(result)
                
                # Save progress after each combination
                results_df = pd.DataFrame(all_results)
                save_progress(results_df)
                
            except Exception as e:
                print(f"  ❌ Combination failed: {str(e)}")
                traceback.print_exc()
                
                # Record failure
                all_results.append({
                    'model': model_name,
                    'config': config_name,
                    'status': 'failed',
                    'error': str(e),
                    'cv_accuracy': None,
                    'cv_precision': None,
                    'cv_recall': None,
                    'cv_f1': None,
                    'cv_std': None,
                    'best_params': None,
                    'n_features': None,
                    'training_time': None
                })
                
                # Save progress
                results_df = pd.DataFrame(all_results)
                save_progress(results_df)
    
    # Final results
    total_time = time.time() - start_time_total
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Combinations evaluated: {combination_count}/{total_combinations}")
    
    # Create final results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save complete results
    complete_results_file = os.path.join(RESULTS_DIR, 'model_zoo_complete_results.csv')
    results_df.to_csv(complete_results_file, index=False)
    print(f"\n✅ Complete results saved: {complete_results_file}")
    
    # Filter successful results
    successful_results = results_df[results_df['status'] == 'success'].copy()
    
    if len(successful_results) == 0:
        print("\n⚠️  No successful evaluations!")
        return
    
    # Sort by CV accuracy
    successful_results = successful_results.sort_values('cv_accuracy', ascending=False)
    
    # Save summary (top 10)
    summary_file = os.path.join(RESULTS_DIR, 'model_zoo_summary.csv')
    successful_results.head(10).to_csv(summary_file, index=False)
    print(f"✅ Summary (top 10) saved: {summary_file}")
    
    # Display top results
    print("\n" + "="*80)
    print("TOP 10 RESULTS")
    print("="*80)
    
    for idx, row in successful_results.head(10).iterrows():
        meets_target = "✅" if row['cv_accuracy'] >= TARGET_CV_ACCURACY else "⚠️ "
        print(f"\n{meets_target} Rank #{idx+1}: {row['model']} × {row['config']}")
        print(f"   CV Accuracy: {row['cv_accuracy']:.4f} (±{row['cv_std']:.4f})")
        print(f"   Precision: {row['cv_precision']:.4f}, Recall: {row['cv_recall']:.4f}, F1: {row['cv_f1']:.4f}")
        print(f"   Features: {row['n_features']}, Time: {row['training_time']:.1f}s")
    
    # Count models meeting target
    meeting_target = successful_results[successful_results['cv_accuracy'] >= TARGET_CV_ACCURACY]
    print(f"\n📊 Models meeting target (≥{TARGET_CV_ACCURACY:.1%} CV): {len(meeting_target)}/{len(successful_results)}")
    
    if len(meeting_target) > 0:
        print("\n🎯 SUCCESS! Models meeting target:")
        for _, row in meeting_target.iterrows():
            print(f"   ✅ {row['model']} × {row['config']}: {row['cv_accuracy']:.4f}")
            print(f"      → submission_{row['model']}_{row['config']}.csv")
    else:
        print(f"\n⚠️  No models met the target of {TARGET_CV_ACCURACY:.1%} CV accuracy")
        print(f"   Best achieved: {successful_results.iloc[0]['cv_accuracy']:.4f}")
        print(f"   Gap to target: {(TARGET_CV_ACCURACY - successful_results.iloc[0]['cv_accuracy']):.4f}")
    
    print("\n" + "="*80)
    print("✅ MODEL ZOO EVALUATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

