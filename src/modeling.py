"""
Model building and hyperparameter tuning utilities.
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint


def build_pipelines(preprocessor):
    """
    Build machine learning pipelines with preprocessor.
    
    Creates three pipelines:
    - "logit": Logistic Regression
    - "rf": Random Forest Classifier
    - "gb": Gradient Boosting Classifier
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline
        
    Returns
    -------
    dict
        Dictionary mapping model names to Pipeline objects
    """
    pipelines = {
        "logit": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=5000, random_state=42))
        ]),
        "rf": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        "gb": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=42))
        ])
    }
    
    print(f"✓ Built {len(pipelines)} pipelines: {list(pipelines.keys())}")
    
    return pipelines


def random_search_rf(pipeline, X, y, cv=None):
    """
    Perform randomized hyperparameter search for Random Forest.
    
    Parameters
    ----------
    pipeline : Pipeline
        Random Forest pipeline to tune
    X : pd.DataFrame or array-like
        Training features
    y : pd.Series or array-like
        Training target
    cv : cross-validation generator, optional
        If None, uses StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    Returns
    -------
    tuple of (best_estimator, best_score, best_params)
        - best_estimator: Fitted pipeline with best parameters
        - best_score: Best cross-validation accuracy score
        - best_params: Dictionary of best hyperparameters
    """
    # Parameter distributions for random search
    param_distributions = {
        'clf__n_estimators': randint(150, 600),
        'clf__max_depth': randint(4, 20),
        'clf__min_samples_split': randint(2, 20),
        'clf__min_samples_leaf': randint(1, 15),
        'clf__max_features': ['sqrt', 'log2', None]
    }
    
    # Default CV strategy
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Starting RandomizedSearchCV for Random Forest...")
    print(f"  - n_iter: 30")
    print(f"  - cv: {cv.get_n_splits()} folds")
    print(f"  - scoring: accuracy")
    
    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        scoring='accuracy',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit
    random_search.fit(X, y)
    
    best_estimator = random_search.best_estimator_
    best_score = random_search.best_score_
    best_params = random_search.best_params_
    
    print(f"\n✓ RandomizedSearchCV complete!")
    print(f"  - Best CV accuracy: {best_score:.4f}")
    print(f"  - Best parameters:")
    for param, value in best_params.items():
        print(f"    * {param}: {value}")
    
    return best_estimator, best_score, best_params

