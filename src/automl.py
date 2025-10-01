"""
AutoML integration using TPOT for automated pipeline optimization.
"""
import os
from tpot import TPOTClassifier
from sklearn.model_selection import StratifiedKFold


def run_tpot_optimization(X, y, generations=5, population_size=20, cv=5, 
                          random_state=42, verbosity=2, n_jobs=-1,
                          output_file='reports/tpot_pipeline.py'):
    """
    Run TPOT AutoML optimization to find the best pipeline.
    
    Parameters
    ----------
    X : pd.DataFrame or array-like
        Training features
    y : pd.Series or array-like
        Training target
    generations : int, default=5
        Number of iterations to run the optimization
    population_size : int, default=20
        Number of individuals to retain in the genetic programming population
    cv : int or cross-validation generator, default=5
        Cross-validation strategy
    random_state : int, default=42
        Random seed for reproducibility
    verbosity : int, default=2
        How much information to print during optimization
    n_jobs : int, default=-1
        Number of processes to use (-1 for all cores)
    output_file : str, default='reports/tpot_pipeline.py'
        Path to save the best pipeline code
        
    Returns
    -------
    tuple of (best_pipeline, best_score)
        - best_pipeline: Fitted TPOT pipeline
        - best_score: Best cross-validation accuracy score
    """
    print("\n" + "="*70)
    print("TPOT AUTOML OPTIMIZATION")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  - Generations: {generations}")
    print(f"  - Population size: {population_size}")
    print(f"  - CV folds: {cv if isinstance(cv, int) else cv.get_n_splits()}")
    print(f"  - Scoring: accuracy")
    print(f"  - Random state: {random_state}")
    print(f"  - Parallel jobs: {n_jobs}")
    
    # Create CV strategy if integer provided
    if isinstance(cv, int):
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_strategy = cv
    
    # Initialize TPOT
    tpot = TPOTClassifier(
        generations=generations,
        population_size=population_size,
        cv=cv_strategy,
        scoring='accuracy',
        random_state=random_state,
        verbosity=verbosity,
        n_jobs=n_jobs,
        config_dict='TPOT light',  # Use lighter config for faster optimization
        early_stop=3,  # Stop if no improvement for 3 generations
        max_time_mins=None,  # No time limit
        max_eval_time_mins=5  # Max 5 minutes per pipeline evaluation
    )
    
    print(f"\nStarting TPOT optimization...")
    print(f"This may take {generations * 2}-{generations * 5} minutes depending on hardware.")
    print(f"Progress will be displayed below:\n")
    
    # Fit TPOT
    tpot.fit(X, y)
    
    # Get best score
    best_score = tpot.score(X, y)
    
    print(f"\n{'='*70}")
    print("TPOT OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest CV Accuracy: {best_score:.4f}")
    print(f"\nBest Pipeline:")
    print(tpot.fitted_pipeline_)
    
    # Export best pipeline
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tpot.export(output_file)
    print(f"\n✓ Best pipeline exported to: {output_file}")
    
    return tpot.fitted_pipeline_, best_score


def compare_with_baseline(tpot_score, baseline_scores):
    """
    Compare TPOT score with baseline model scores.
    
    Parameters
    ----------
    tpot_score : float
        TPOT best CV accuracy
    baseline_scores : dict
        Dictionary of model names to accuracy scores
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    import pandas as pd
    
    # Add TPOT to comparison
    all_scores = baseline_scores.copy()
    all_scores['TPOT AutoML'] = tpot_score
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Model': list(all_scores.keys()),
        'Accuracy': list(all_scores.values())
    }).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    comparison['Rank'] = range(1, len(comparison) + 1)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison.to_string(index=False))
    
    best_model = comparison.iloc[0]['Model']
    best_score = comparison.iloc[0]['Accuracy']
    
    print(f"\n🏆 Best Model: {best_model} ({best_score:.4f})")
    
    return comparison

