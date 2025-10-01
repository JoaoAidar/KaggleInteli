"""
Feature configuration module for model zoo experiments.
Provides 6 different feature configurations to test.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier


def get_interaction_features(df):
    """Create only interaction features (no polynomials)."""
    df = df.copy()
    
    # Funding-based interactions
    if 'funding_total_usd' in df.columns and 'relationships' in df.columns:
        df['funding_per_relationship'] = df['funding_total_usd'] / (df['relationships'] + 1)
    
    if 'funding_total_usd' in df.columns and 'funding_rounds' in df.columns:
        df['funding_per_round'] = df['funding_total_usd'] / (df['funding_rounds'] + 1)
    
    if 'milestones' in df.columns and 'funding_rounds' in df.columns:
        df['milestone_to_funding_ratio'] = df['milestones'] / (df['funding_rounds'] + 1)
    
    if 'age_last_funding_year' in df.columns and 'age_first_funding_year' in df.columns:
        df['funding_duration'] = df['age_last_funding_year'] - df['age_first_funding_year']
        df['funding_duration'] = df['funding_duration'].fillna(0)
    
    if 'age_last_milestone_year' in df.columns and 'age_first_milestone_year' in df.columns:
        df['milestone_duration'] = df['age_last_milestone_year'] - df['age_first_milestone_year']
        df['milestone_duration'] = df['milestone_duration'].fillna(0)
    
    if 'funding_total_usd' in df.columns and 'milestones' in df.columns:
        df['funding_per_milestone'] = df['funding_total_usd'] / (df['milestones'] + 1)
    
    if 'avg_participants' in df.columns and 'funding_rounds' in df.columns:
        df['total_participants'] = df['avg_participants'] * df['funding_rounds']
    
    if 'has_VC' in df.columns and 'has_angel' in df.columns:
        df['has_both_vc_angel'] = (df['has_VC'] & df['has_angel']).astype(int)
    
    if 'has_roundA' in df.columns and 'has_roundB' in df.columns:
        df['has_multiple_rounds'] = (df['has_roundA'] & df['has_roundB']).astype(int)
    
    location_cols = [col for col in df.columns if col.startswith('is_') and 
                     any(state in col for state in ['CA', 'NY', 'MA', 'TX'])]
    if location_cols:
        df['is_major_hub'] = df[location_cols].sum(axis=1).clip(0, 1)
    
    return df


def get_polynomial_features(df, top_features):
    """Create only polynomial features (no interactions)."""
    from sklearn.preprocessing import PolynomialFeatures
    
    df = df.copy()
    existing_features = [f for f in top_features if f in df.columns]
    
    if not existing_features:
        return df
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df[existing_features])
    poly_feature_names = poly.get_feature_names_out(existing_features)
    
    for i, name in enumerate(poly_feature_names):
        if name not in existing_features:
            df[f'poly_{name}'] = poly_features[:, i]
    
    return df


def config_a_original(X_train, X_test, y_train=None):
    """Config A: Original 31 features only."""
    return X_train.copy(), X_test.copy(), list(X_train.columns)


def config_b_interactions(X_train, X_test, y_train=None):
    """Config B: Original + interaction features (41 features)."""
    X_train_new = get_interaction_features(X_train)
    X_test_new = get_interaction_features(X_test)
    return X_train_new, X_test_new, list(X_train_new.columns)


def config_c_polynomials(X_train, X_test, y_train=None):
    """Config C: Original + polynomial features (46 features)."""
    top_features = ['funding_total_usd', 'relationships', 'funding_rounds', 
                    'avg_participants', 'milestones']
    X_train_new = get_polynomial_features(X_train, top_features)
    X_test_new = get_polynomial_features(X_test, top_features)
    return X_train_new, X_test_new, list(X_train_new.columns)


def config_d_all_engineered(X_train, X_test, y_train=None):
    """Config D: All engineered features (56 features)."""
    # Add interactions
    X_train_new = get_interaction_features(X_train)
    X_test_new = get_interaction_features(X_test)
    
    # Add polynomials
    top_features = ['funding_total_usd', 'relationships', 'funding_rounds', 
                    'avg_participants', 'milestones']
    X_train_new = get_polynomial_features(X_train_new, top_features)
    X_test_new = get_polynomial_features(X_test_new, top_features)
    
    return X_train_new, X_test_new, list(X_train_new.columns)


def config_e_selectkbest(X_train, X_test, y_train):
    """Config E: Top 25 features using SelectKBest."""
    # First get all engineered features
    X_train_eng, X_test_eng, _ = config_d_all_engineered(X_train, X_test, y_train)
    
    # Select top 25 features
    k = min(25, X_train_eng.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train_eng, y_train)
    
    selected_features = X_train_eng.columns[selector.get_support()].tolist()
    
    X_train_selected = X_train_eng[selected_features]
    X_test_selected = X_test_eng[selected_features]
    
    return X_train_selected, X_test_selected, selected_features


def config_f_rf_importance(X_train, X_test, y_train):
    """Config F: Top 20 features using RF feature importance."""
    # First get all engineered features
    X_train_eng, X_test_eng, _ = config_d_all_engineered(X_train, X_test, y_train)
    
    # Train RF to get feature importances
    from src.features import split_columns, build_preprocessor
    
    numeric_cols, categorical_cols = split_columns(X_train_eng)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    
    X_train_preprocessed = preprocessor.fit_transform(X_train_eng)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_preprocessed, y_train)
    
    # Get feature importances
    feature_names = (numeric_cols + 
                     [f'cat_{categorical_cols[0]}_{i}' 
                      for i in range(len(preprocessor.named_transformers_['cat']
                                        .named_steps['onehot'].get_feature_names_out()))])
    
    importances = pd.DataFrame({
        'feature': feature_names[:len(rf.feature_importances_)],
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Map back to original feature names (before preprocessing)
    top_original_features = []
    for feat in importances['feature'].head(30):  # Get more to account for one-hot encoding
        # Remove 'cat_' prefix if present
        if feat.startswith('cat_'):
            base_feat = feat.split('_')[1]
        else:
            base_feat = feat
        
        if base_feat in X_train_eng.columns and base_feat not in top_original_features:
            top_original_features.append(base_feat)
        
        if len(top_original_features) >= 20:
            break
    
    X_train_selected = X_train_eng[top_original_features]
    X_test_selected = X_test_eng[top_original_features]
    
    return X_train_selected, X_test_selected, top_original_features


# Configuration registry
FEATURE_CONFIGS = {
    'A_original': {
        'name': 'Original 31 features',
        'func': config_a_original,
        'expected_features': 31
    },
    'B_interactions': {
        'name': 'Original + Interactions (41 features)',
        'func': config_b_interactions,
        'expected_features': 41
    },
    'C_polynomials': {
        'name': 'Original + Polynomials (46 features)',
        'func': config_c_polynomials,
        'expected_features': 46
    },
    'D_all_engineered': {
        'name': 'All Engineered (56 features)',
        'func': config_d_all_engineered,
        'expected_features': 56
    },
    'E_selectkbest': {
        'name': 'SelectKBest Top 25',
        'func': config_e_selectkbest,
        'expected_features': 25
    },
    'F_rf_importance': {
        'name': 'RF Importance Top 20',
        'func': config_f_rf_importance,
        'expected_features': 20
    }
}


def apply_feature_config(config_name, X_train, X_test, y_train=None):
    """
    Apply a feature configuration.
    
    Parameters
    ----------
    config_name : str
        One of: 'A_original', 'B_interactions', 'C_polynomials', 
                'D_all_engineered', 'E_selectkbest', 'F_rf_importance'
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series, optional
        Training target (required for E and F)
        
    Returns
    -------
    tuple
        (X_train_transformed, X_test_transformed, feature_names)
    """
    if config_name not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(FEATURE_CONFIGS.keys())}")
    
    config = FEATURE_CONFIGS[config_name]
    
    if config_name in ['E_selectkbest', 'F_rf_importance'] and y_train is None:
        raise ValueError(f"Config {config_name} requires y_train")
    
    return config['func'](X_train, X_test, y_train)


def get_config_info():
    """Get information about all available configurations."""
    info = []
    for config_id, config in FEATURE_CONFIGS.items():
        info.append({
            'config_id': config_id,
            'name': config['name'],
            'expected_features': config['expected_features']
        })
    return pd.DataFrame(info)

