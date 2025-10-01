"""
Comprehensive model zoo with all viable models for classification.
"""
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def get_model_zoo(preprocessor):
    """
    Get all models with their default configurations.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline
        
    Returns
    -------
    dict
        Dictionary of model_name -> Pipeline
    """
    models = {}
    
    # Tree-based models
    models['rf'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    models['extra_trees'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', ExtraTreesClassifier(random_state=42, n_jobs=-1))
    ])
    
    models['gb'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
    
    models['xgboost'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', xgb.XGBClassifier(random_state=42, eval_metric='logloss', 
                                  use_label_encoder=False, n_jobs=-1))
    ])
    
    models['lightgbm'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1))
    ])
    
    if CATBOOST_AVAILABLE:
        models['catboost'] = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', cb.CatBoostClassifier(random_state=42, verbose=False))
        ])
    
    # Linear models
    models['logistic'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
    ])
    
    models['ridge'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RidgeClassifier(random_state=42))
    ])
    
    models['sgd'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', SGDClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Support Vector Machines
    models['svc_rbf'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    
    models['svc_poly'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', SVC(kernel='poly', probability=True, random_state=42))
    ])
    
    models['linear_svc'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LinearSVC(random_state=42, max_iter=2000))
    ])
    
    # Neural Networks
    models['mlp'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', MLPClassifier(random_state=42, max_iter=500))
    ])
    
    # Bagging
    models['bagging_rf'] = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', BaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return models


def get_param_distributions():
    """
    Get hyperparameter distributions for RandomizedSearchCV.
    
    Returns
    -------
    dict
        Dictionary of model_name -> param_distributions
    """
    params = {}
    
    # Random Forest
    params['rf'] = {
        'clf__n_estimators': [200, 300, 400, 500, 600, 800],
        'clf__max_depth': [10, 15, 20, 25, 30, None],
        'clf__min_samples_split': [2, 5, 10, 15, 20],
        'clf__min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'clf__max_features': ['sqrt', 'log2', 0.3, 0.5],
        'clf__class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    # Extra Trees
    params['extra_trees'] = {
        'clf__n_estimators': [200, 300, 400, 500, 600],
        'clf__max_depth': [10, 15, 20, 25, 30, None],
        'clf__min_samples_split': [2, 5, 10, 15],
        'clf__min_samples_leaf': [1, 2, 4, 6, 8],
        'clf__max_features': ['sqrt', 'log2', 0.3, 0.5]
    }
    
    # Gradient Boosting
    params['gb'] = {
        'clf__n_estimators': [100, 200, 300, 400, 500],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'clf__max_depth': [3, 4, 5, 6, 7, 8],
        'clf__min_samples_split': [2, 5, 10, 15],
        'clf__min_samples_leaf': [1, 2, 4, 6],
        'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    # XGBoost
    params['xgboost'] = {
        'clf__n_estimators': randint(100, 800),
        'clf__max_depth': randint(3, 15),
        'clf__learning_rate': uniform(0.01, 0.3),
        'clf__subsample': uniform(0.6, 0.4),
        'clf__colsample_bytree': uniform(0.6, 0.4),
        'clf__min_child_weight': randint(1, 10),
        'clf__gamma': uniform(0, 0.5),
        'clf__reg_alpha': uniform(0, 1),
        'clf__reg_lambda': uniform(0, 1)
    }
    
    # LightGBM
    params['lightgbm'] = {
        'clf__n_estimators': randint(100, 800),
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
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        params['catboost'] = {
            'clf__iterations': [100, 200, 300, 400, 500],
            'clf__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'clf__depth': [4, 5, 6, 7, 8, 9, 10],
            'clf__l2_leaf_reg': [1, 3, 5, 7, 9]
        }
    
    # Logistic Regression
    params['logistic'] = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l1', 'l2', 'elasticnet'],
        'clf__solver': ['saga'],
        'clf__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    # Ridge Classifier
    params['ridge'] = {
        'clf__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }
    
    # SGD Classifier
    params['sgd'] = {
        'clf__loss': ['hinge', 'log_loss', 'modified_huber'],
        'clf__penalty': ['l1', 'l2', 'elasticnet'],
        'clf__alpha': [0.0001, 0.001, 0.01, 0.1],
        'clf__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    # SVC RBF
    params['svc_rbf'] = {
        'clf__C': [0.1, 1, 10, 100],
        'clf__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
    
    # SVC Polynomial
    params['svc_poly'] = {
        'clf__C': [0.1, 1, 10, 100],
        'clf__degree': [2, 3, 4],
        'clf__gamma': ['scale', 'auto']
    }
    
    # Linear SVC
    params['linear_svc'] = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l2'],
        'clf__loss': ['hinge', 'squared_hinge']
    }
    
    # MLP Classifier
    params['mlp'] = {
        'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'clf__activation': ['relu', 'tanh'],
        'clf__alpha': [0.0001, 0.001, 0.01, 0.1],
        'clf__learning_rate': ['constant', 'adaptive']
    }
    
    # Bagging RF
    params['bagging_rf'] = {
        'clf__n_estimators': [5, 10, 15, 20],
        'clf__max_samples': [0.5, 0.7, 0.9, 1.0],
        'clf__max_features': [0.5, 0.7, 0.9, 1.0]
    }
    
    return params


def get_model_priority():
    """
    Get model priority order (higher priority = train first).
    
    Returns
    -------
    dict
        Dictionary of model_name -> priority (1-5, 5 is highest)
    """
    priority = {
        # Priority 5: Tree-based models (most likely to succeed)
        'rf': 5,
        'xgboost': 5,
        'lightgbm': 5,
        'catboost': 5,
        'extra_trees': 5,
        
        # Priority 4: Gradient boosting variants
        'gb': 4,
        
        # Priority 3: Ensemble methods
        'bagging_rf': 3,
        
        # Priority 2: Linear models
        'logistic': 2,
        'ridge': 2,
        'sgd': 2,
        
        # Priority 1: SVMs and Neural Networks
        'svc_rbf': 1,
        'svc_poly': 1,
        'linear_svc': 1,
        'mlp': 1
    }
    
    return priority

