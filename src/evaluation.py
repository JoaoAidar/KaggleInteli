"""
Model evaluation and validation utilities.
"""
import pandas as pd
from sklearn.model_selection import cross_validate


def cv_report(pipeline, X, y, cv):
    """
    Generate cross-validation report for a single pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        Fitted or unfitted pipeline
    X : pd.DataFrame or array-like
        Features
    y : pd.Series or array-like
        Target
    cv : cross-validation generator
        CV splitter
        
    Returns
    -------
    dict
        Dictionary with keys: accuracy, precision, recall, f1
        Each value is the mean score across CV folds
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    cv_results = cross_validate(
        pipeline, X, y, 
        cv=cv, 
        scoring=scoring,
        n_jobs=-1
    )
    
    report = {
        'accuracy': cv_results['test_accuracy'].mean(),
        'precision': cv_results['test_precision'].mean(),
        'recall': cv_results['test_recall'].mean(),
        'f1': cv_results['test_f1'].mean()
    }
    
    return report


def evaluate_all(pipelines_dict, X, y, cv):
    """
    Evaluate all pipelines using cross-validation.
    
    Parameters
    ----------
    pipelines_dict : dict
        Dictionary mapping model names to Pipeline objects
    X : pd.DataFrame or array-like
        Features
    y : pd.Series or array-like
        Target
    cv : cross-validation generator
        CV splitter
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: model, accuracy, precision, recall, f1
        Sorted by accuracy (descending)
    """
    print(f"Evaluating {len(pipelines_dict)} models with {cv.get_n_splits()}-fold CV...")
    
    results = []
    
    for model_name, pipeline in pipelines_dict.items():
        print(f"\n  Evaluating {model_name}...")
        report = cv_report(pipeline, X, y, cv)
        
        results.append({
            'model': model_name,
            'accuracy': report['accuracy'],
            'precision': report['precision'],
            'recall': report['recall'],
            'f1': report['f1']
        })
        
        print(f"    ✓ Accuracy: {report['accuracy']:.4f}")
    
    # Create DataFrame and sort by accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    
    print(f"\n✓ Evaluation complete!")
    
    return results_df


def assert_min_accuracy(accuracy, threshold=0.80, raise_error=False):
    """
    Check if accuracy meets minimum threshold.
    
    Parameters
    ----------
    accuracy : float
        Accuracy score to check
    threshold : float, default=0.80
        Minimum required accuracy
    raise_error : bool, default=False
        If True, raises ValueError when threshold not met
        If False, prints warning message
        
    Returns
    -------
    bool
        True if accuracy >= threshold, False otherwise
        
    Raises
    ------
    ValueError
        If raise_error=True and accuracy < threshold
    """
    meets_threshold = accuracy >= threshold
    
    if not meets_threshold:
        message = (
            f"⚠ WARNING: Accuracy {accuracy:.4f} is below threshold {threshold:.4f}. "
            f"Consider additional feature engineering or hyperparameter tuning."
        )
        
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
    else:
        print(f"✓ Accuracy {accuracy:.4f} meets threshold {threshold:.4f}")
    
    return meets_threshold

