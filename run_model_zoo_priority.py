"""
Priority Model Zoo: Focus on most promising models first.

Evaluates high-priority models (tree-based) with all feature configurations
to quickly find models that meet the ≥80% Kaggle target.
"""
import sys
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.io_utils import load_data, get_target_name
from src.features import split_columns, build_preprocessor
from src.feature_configs import FEATURE_CONFIGS, apply_feature_config
from src.model_zoo import get_model_zoo, get_param_distributions

# Configuration
DATA_DIR = 'data'
SEED = 42
N_ITER = 100
CV_FOLDS = 10
TARGET_CV_ACCURACY = 0.815

# Priority models only (tree-based + gradient boosting)
PRIORITY_MODELS = ['rf', 'xgboost', 'lightgbm', 'extra_trees', 'gb']

# Priority feature configs (most likely to work)
PRIORITY_CONFIGS = ['A_original', 'B_interactions', 'C_polynomials']


def evaluate_single_combination(model_name, config_name, X_train_raw, X_test_raw, 
                                y_train, test_ids, cv):
    """Evaluate a single model × config combination."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name} × {config_name}")
    print(f"{'='*70}")
    
    try:
        # Apply feature configuration
        print("  Applying feature configuration...")
        X_train, X_test, feature_names = apply_feature_config(
            config_name, X_train_raw, X_test_raw, y_train
        )
        print(f"  ✓ Features: {len(feature_names)}")
        
        # Build preprocessor
        numeric_cols, categorical_cols = split_columns(X_train)
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        
        # Get model and parameters
        models = get_model_zoo(preprocessor)
        param_dists = get_param_distributions()
        
        if model_name not in models or model_name not in param_dists:
            print(f"  ✗ Model {model_name} not available")
            return None
        
        model_pipeline = models[model_name]
        param_dist = param_dists[model_name]
        
        # Hyperparameter tuning
        print(f"  Tuning hyperparameters ({N_ITER} iterations)...")
        start_time = time.time()
        
        random_search = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=param_dist,
            n_iter=N_ITER,
            scoring='accuracy',
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        # Cross-validation
        print("  Cross-validating...")
        cv_results = cross_validate(
            best_model, X_train, y_train,
            cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            n_jobs=-1
        )
        
        cv_accuracy = cv_results['test_accuracy'].mean()
        cv_std = cv_results['test_accuracy'].std()
        cv_precision = cv_results['test_precision'].mean()
        cv_recall = cv_results['test_recall'].mean()
        cv_f1 = cv_results['test_f1'].mean()
        
        training_time = time.time() - start_time
        
        # Results
        result = {
            'model': model_name,
            'config': config_name,
            'cv_accuracy': cv_accuracy,
            'cv_std': cv_std,
            'cv_precision': cv_precision,
            'cv_recall': cv_recall,
            'cv_f1': cv_f1,
            'n_features': len(feature_names),
            'training_time': training_time,
            'best_params': random_search.best_params_
        }
        
        # Print results
        meets_target = cv_accuracy >= TARGET_CV_ACCURACY
        status = "✓ MEETS TARGET" if meets_target else "✗ Below target"
        
        print(f"\n  {status}")
        print(f"  CV Accuracy: {cv_accuracy:.4f} (±{cv_std:.4f})")
        print(f"  Precision: {cv_precision:.4f}, Recall: {cv_recall:.4f}, F1: {cv_f1:.4f}")
        print(f"  Training time: {training_time:.1f}s")
        
        # Generate submission if meets target
        if meets_target:
            print(f"\n  Generating submission...")
            best_model.fit(X_train, y_train)
            predictions = best_model.predict(X_test)
            
            submission_df = pd.DataFrame({
                'id': test_ids,
                'labels': predictions
            })
            
            submission_file = f'submission_{model_name}_{config_name}.csv'
            submission_df.to_csv(submission_file, index=False)
            print(f"  ✓ Submission saved: {submission_file}")
            
            # Save best parameters
            os.makedirs('reports/model_zoo_best_params', exist_ok=True)
            params_file = f'reports/model_zoo_best_params/{model_name}_{config_name}_params.json'
            with open(params_file, 'w') as f:
                serializable_params = {}
                for k, v in random_search.best_params_.items():
                    if isinstance(v, (np.integer, np.floating)):
                        serializable_params[k] = v.item()
                    else:
                        serializable_params[k] = v
                json.dump(serializable_params, f, indent=2)
        
        return result
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("PRIORITY MODEL ZOO")
    print("="*70)
    print(f"\nTarget: ≥{TARGET_CV_ACCURACY:.1%} CV accuracy (≥80% Kaggle)")
    print(f"Models: {', '.join(PRIORITY_MODELS)}")
    print(f"Configs: {', '.join(PRIORITY_CONFIGS)}")
    print(f"Total combinations: {len(PRIORITY_MODELS) * len(PRIORITY_CONFIGS)}")
    print("="*70 + "\n")
    
    # Setup
    np.random.seed(SEED)
    os.makedirs('reports/model_zoo_results', exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df, test_df, sample_submission_df = load_data(DATA_DIR)
    target_name = get_target_name(sample_submission_df)
    
    X_train_raw = train_df.drop(columns=[target_name, 'id'], errors='ignore')
    y_train = train_df[target_name]
    X_test_raw = test_df.drop(columns=['id'], errors='ignore')
    test_ids = test_df['id'].values
    
    print(f"✓ Data loaded: {len(X_train_raw)} train, {len(X_test_raw)} test")
    
    # CV strategy
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    
    # Evaluate all combinations
    all_results = []
    combination_count = 0
    total_combinations = len(PRIORITY_MODELS) * len(PRIORITY_CONFIGS)
    
    start_time_total = time.time()
    
    for model_name in PRIORITY_MODELS:
        for config_name in PRIORITY_CONFIGS:
            combination_count += 1
            print(f"\n[{combination_count}/{total_combinations}]")
            
            result = evaluate_single_combination(
                model_name, config_name,
                X_train_raw, X_test_raw, y_train, test_ids, cv
            )
            
            if result is not None:
                all_results.append(result)
                
                # Save progress
                results_df = pd.DataFrame(all_results)
                results_df.to_csv('reports/model_zoo_results/priority_results.csv', index=False)
    
    # Final summary
    total_time = time.time() - start_time_total
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Successful evaluations: {len(all_results)}/{total_combinations}")
    
    if len(all_results) == 0:
        print("\n✗ No successful evaluations!")
        return
    
    # Sort by accuracy
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('cv_accuracy', ascending=False)
    
    # Save final results
    results_df.to_csv('reports/model_zoo_results/priority_summary.csv', index=False)
    
    # Display top results
    print("\n" + "="*70)
    print("TOP RESULTS")
    print("="*70)
    
    for idx, row in results_df.head(5).iterrows():
        meets_target = "✓" if row['cv_accuracy'] >= TARGET_CV_ACCURACY else "✗"
        print(f"\n{meets_target} #{idx+1}: {row['model']} × {row['config']}")
        print(f"   CV Accuracy: {row['cv_accuracy']:.4f} (±{row['cv_std']:.4f})")
        print(f"   Precision: {row['cv_precision']:.4f}, Recall: {row['cv_recall']:.4f}")
        print(f"   Features: {row['n_features']}, Time: {row['training_time']:.1f}s")
    
    # Count models meeting target
    meeting_target = results_df[results_df['cv_accuracy'] >= TARGET_CV_ACCURACY]
    print(f"\n📊 Models meeting target: {len(meeting_target)}/{len(results_df)}")
    
    if len(meeting_target) > 0:
        print("\n✓ SUCCESS! Models meeting target:")
        for _, row in meeting_target.iterrows():
            print(f"   {row['model']} × {row['config']}: {row['cv_accuracy']:.4f}")
            print(f"   → submission_{row['model']}_{row['config']}.csv")
    else:
        print(f"\n✗ No models met target of {TARGET_CV_ACCURACY:.1%}")
        print(f"   Best: {results_df.iloc[0]['cv_accuracy']:.4f}")
        print(f"   Gap: {(TARGET_CV_ACCURACY - results_df.iloc[0]['cv_accuracy']):.4f}")
    
    print("\n" + "="*70)
    print("✓ PRIORITY MODEL ZOO COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

