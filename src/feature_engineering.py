"""
Advanced feature engineering for improved model performance.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif


def create_interaction_features(df):
    """
    Create interaction features from existing features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional interaction features
    """
    df = df.copy()
    
    # Funding-based interactions
    if 'funding_total_usd' in df.columns and 'relationships' in df.columns:
        df['funding_per_relationship'] = df['funding_total_usd'] / (df['relationships'] + 1)
    
    if 'funding_total_usd' in df.columns and 'funding_rounds' in df.columns:
        df['funding_per_round'] = df['funding_total_usd'] / (df['funding_rounds'] + 1)
    
    if 'milestones' in df.columns and 'funding_rounds' in df.columns:
        df['milestone_to_funding_ratio'] = df['milestones'] / (df['funding_rounds'] + 1)
    
    # Time-based features
    if 'age_last_funding_year' in df.columns and 'age_first_funding_year' in df.columns:
        df['funding_duration'] = df['age_last_funding_year'] - df['age_first_funding_year']
        df['funding_duration'] = df['funding_duration'].fillna(0)
    
    if 'age_last_milestone_year' in df.columns and 'age_first_milestone_year' in df.columns:
        df['milestone_duration'] = df['age_last_milestone_year'] - df['age_first_milestone_year']
        df['milestone_duration'] = df['milestone_duration'].fillna(0)
    
    # Efficiency metrics
    if 'funding_total_usd' in df.columns and 'milestones' in df.columns:
        df['funding_per_milestone'] = df['funding_total_usd'] / (df['milestones'] + 1)
    
    if 'avg_participants' in df.columns and 'funding_rounds' in df.columns:
        df['total_participants'] = df['avg_participants'] * df['funding_rounds']
    
    # Success indicators
    if 'has_VC' in df.columns and 'has_angel' in df.columns:
        df['has_both_vc_angel'] = (df['has_VC'] & df['has_angel']).astype(int)
    
    if 'has_roundA' in df.columns and 'has_roundB' in df.columns:
        df['has_multiple_rounds'] = (df['has_roundA'] & df['has_roundB']).astype(int)
    
    # Location diversity
    location_cols = [col for col in df.columns if col.startswith('is_') and 
                     any(state in col for state in ['CA', 'NY', 'MA', 'TX'])]
    if location_cols:
        df['is_major_hub'] = df[location_cols].sum(axis=1).clip(0, 1)
    
    print(f"✓ Created {len(df.columns) - len(df.columns.intersection(df.columns))} new interaction features")
    
    return df


def create_polynomial_features(df, feature_names, degree=2):
    """
    Create polynomial features for specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_names : list
        List of feature names to create polynomials for
    degree : int, default=2
        Degree of polynomial features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with polynomial features added
    """
    df = df.copy()
    
    # Filter to only existing features
    existing_features = [f for f in feature_names if f in df.columns]
    
    if not existing_features:
        print("⚠ No features found for polynomial expansion")
        return df
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df[existing_features])
    
    # Get feature names
    poly_feature_names = poly.get_feature_names_out(existing_features)
    
    # Add only new features (exclude original features)
    new_features = []
    for i, name in enumerate(poly_feature_names):
        if name not in existing_features:
            df[f'poly_{name}'] = poly_features[:, i]
            new_features.append(f'poly_{name}')
    
    print(f"✓ Created {len(new_features)} polynomial features (degree={degree})")
    
    return df


def select_best_features(X, y, k=50):
    """
    Select k best features using univariate statistical tests.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    k : int, default=50
        Number of top features to select
        
    Returns
    -------
    list
        List of selected feature names
    """
    # Ensure k doesn't exceed number of features
    k = min(k, X.shape[1])
    
    # Select k best features
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Get feature scores
    scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    print(f"✓ Selected top {k} features")
    print(f"  Top 5 features:")
    for idx, row in scores.head(5).iterrows():
        print(f"    - {row['feature']}: {row['score']:.2f}")
    
    return selected_features


def get_top_important_features(feature_names, n=5):
    """
    Get list of likely most important features based on domain knowledge.
    
    Parameters
    ----------
    feature_names : list
        List of all feature names
    n : int, default=5
        Number of top features to return
        
    Returns
    -------
    list
        List of top feature names
    """
    # Priority features based on domain knowledge
    priority_features = [
        'funding_total_usd',
        'relationships',
        'funding_rounds',
        'avg_participants',
        'milestones',
        'has_VC',
        'has_roundA',
        'has_roundB',
        'is_CA',
        'is_NY'
    ]
    
    # Filter to only existing features
    existing_priority = [f for f in priority_features if f in feature_names]
    
    # Return top n
    return existing_priority[:n]


def engineer_all_features(df, y=None, create_poly=True, select_features=False, k_best=50):
    """
    Apply all feature engineering steps.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    y : pd.Series, optional
        Target variable (required if select_features=True)
    create_poly : bool, default=True
        Whether to create polynomial features
    select_features : bool, default=False
        Whether to perform feature selection
    k_best : int, default=50
        Number of features to select if select_features=True
        
    Returns
    -------
    pd.DataFrame or tuple
        If select_features=False: DataFrame with engineered features
        If select_features=True: (DataFrame, list of selected features)
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    # Create interaction features
    print("\n1. Creating interaction features...")
    df_engineered = create_interaction_features(df)
    
    # Create polynomial features for top important features
    if create_poly:
        print("\n2. Creating polynomial features...")
        top_features = get_top_important_features(df.columns.tolist(), n=5)
        df_engineered = create_polynomial_features(df_engineered, top_features, degree=2)
    
    # Feature selection
    if select_features and y is not None:
        print("\n3. Selecting best features...")
        selected_features = select_best_features(df_engineered, y, k=k_best)
        df_selected = df_engineered[selected_features]
        
        print(f"\n✓ Feature engineering complete!")
        print(f"  - Original features: {df.shape[1]}")
        print(f"  - Engineered features: {df_engineered.shape[1]}")
        print(f"  - Selected features: {len(selected_features)}")
        
        return df_selected, selected_features
    
    print(f"\n✓ Feature engineering complete!")
    print(f"  - Original features: {df.shape[1]}")
    print(f"  - Engineered features: {df_engineered.shape[1]}")
    
    return df_engineered

