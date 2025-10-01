"""
Advanced machine learning models for improved performance.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
import xgboost as xgb
import lightgbm as lgb


def build_advanced_pipelines(preprocessor):
    """
    Build advanced machine learning pipelines.
    
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
        "xgboost": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ))
        ]),
        "lightgbm": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", lgb.LGBMClassifier(
                random_state=42,
                verbose=-1
            ))
        ]),
        "extra_trees": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", ExtraTreesClassifier(
                random_state=42,
                n_jobs=-1
            ))
        ])
    }
    
    print(f"✓ Built {len(pipelines)} advanced pipelines: {list(pipelines.keys())}")
    
    return pipelines


def tune_random_forest_extensive(pipeline, X, y, cv=None, n_iter=100, random_state=42):
    """
    Extensive hyperparameter tuning for Random Forest with expanded search space.
    
    Parameters
    ----------
    pipeline : Pipeline
        Random Forest pipeline to tune
    X : pd.DataFrame or array-like
        Training features
    y : pd.Series or array-like
        Training target
    cv : cross-validation generator, optional
        If None, uses StratifiedKFold(n_splits=5)
    n_iter : int, default=100
        Number of parameter settings sampled
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    tuple of (best_estimator, best_score, best_params)
    """
    # Expanded parameter distributions
    param_distributions = {
        'clf__n_estimators': [200, 300, 400, 500, 600, 800, 1000],
        'clf__max_depth': [10, 15, 20, 25, 30, None],
        'clf__min_samples_split': [2, 5, 10, 15, 20],
        'clf__min_samples_leaf': [1, 2, 4, 6, 8],
        'clf__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, None],
        'clf__bootstrap': [True, False],
        'clf__class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    print(f"\nExtensive Random Forest Tuning:")
    print(f"  - n_iter: {n_iter}")
    print(f"  - cv: {cv.get_n_splits()} folds")
    print(f"  - Parameter space size: ~{7*6*5*5*6*2*3:,} combinations")
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X, y)
    
    print(f"\n✓ Extensive tuning complete!")
    print(f"  - Best CV accuracy: {random_search.best_score_:.4f}")
    print(f"  - Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"    * {param}: {value}")
    
    return random_search.best_estimator_, random_search.best_score_, random_search.best_params_


def tune_xgboost(pipeline, X, y, cv=None, n_iter=50, random_state=42):
    """
    Hyperparameter tuning for XGBoost.
    
    Parameters
    ----------
    pipeline : Pipeline
        XGBoost pipeline to tune
    X : pd.DataFrame or array-like
        Training features
    y : pd.Series or array-like
        Training target
    cv : cross-validation generator, optional
    n_iter : int, default=50
        Number of parameter settings sampled
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    tuple of (best_estimator, best_score, best_params)
    """
    param_distributions = {
        'clf__n_estimators': randint(100, 1000),
        'clf__max_depth': randint(3, 15),
        'clf__learning_rate': uniform(0.01, 0.3),
        'clf__subsample': uniform(0.6, 0.4),
        'clf__colsample_bytree': uniform(0.6, 0.4),
        'clf__min_child_weight': randint(1, 10),
        'clf__gamma': uniform(0, 0.5),
        'clf__reg_alpha': uniform(0, 1),
        'clf__reg_lambda': uniform(0, 1)
    }
    
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    print(f"\nXGBoost Tuning:")
    print(f"  - n_iter: {n_iter}")
    print(f"  - cv: {cv.get_n_splits()} folds")
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X, y)
    
    print(f"\n✓ XGBoost tuning complete!")
    print(f"  - Best CV accuracy: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_score_, random_search.best_params_


def tune_lightgbm(pipeline, X, y, cv=None, n_iter=50, random_state=42):
    """
    Hyperparameter tuning for LightGBM.
    
    Parameters
    ----------
    pipeline : Pipeline
        LightGBM pipeline to tune
    X : pd.DataFrame or array-like
        Training features
    y : pd.Series or array-like
        Training target
    cv : cross-validation generator, optional
    n_iter : int, default=50
        Number of parameter settings sampled
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    tuple of (best_estimator, best_score, best_params)
    """
    param_distributions = {
        'clf__n_estimators': randint(100, 1000),
        'clf__num_leaves': randint(20, 150),
        'clf__max_depth': randint(3, 15),
        'clf__learning_rate': uniform(0.01, 0.3),
        'clf__feature_fraction': uniform(0.6, 0.4),
        'clf__bagging_fraction': uniform(0.6, 0.4),
        'clf__bagging_freq': randint(1, 7),
        'clf__min_child_samples': randint(5, 100),
        'clf__reg_alpha': uniform(0, 1),
        'clf__reg_lambda': uniform(0, 1)
    }
    
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    print(f"\nLightGBM Tuning:")
    print(f"  - n_iter: {n_iter}")
    print(f"  - cv: {cv.get_n_splits()} folds")
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X, y)
    
    print(f"\n✓ LightGBM tuning complete!")
    print(f"  - Best CV accuracy: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_score_, random_search.best_params_


def build_stacking_ensemble(base_models, preprocessor, meta_learner=None):
    """
    Build a stacking ensemble classifier.
    
    Parameters
    ----------
    base_models : dict
        Dictionary of model names to fitted pipelines
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    meta_learner : estimator, optional
        Meta-learner for stacking. If None, uses LogisticRegression
        
    Returns
    -------
    Pipeline
        Stacking ensemble pipeline
    """
    if meta_learner is None:
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    # Extract classifiers from pipelines
    estimators = [(name, pipeline.named_steps['clf']) for name, pipeline in base_models.items()]
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    # Wrap in pipeline
    stacking_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", stacking_clf)
    ])
    
    print(f"✓ Built stacking ensemble with {len(estimators)} base models")
    
    return stacking_pipeline


def build_voting_ensemble(base_models, preprocessor, voting='soft'):
    """
    Build a voting ensemble classifier.
    
    Parameters
    ----------
    base_models : dict
        Dictionary of model names to fitted pipelines
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    voting : str, default='soft'
        Voting strategy ('hard' or 'soft')
        
    Returns
    -------
    Pipeline
        Voting ensemble pipeline
    """
    # Extract classifiers from pipelines
    estimators = [(name, pipeline.named_steps['clf']) for name, pipeline in base_models.items()]
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1
    )
    
    # Wrap in pipeline
    voting_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", voting_clf)
    ])
    
    print(f"✓ Built voting ensemble with {len(estimators)} base models (voting={voting})")
    
    return voting_pipeline

