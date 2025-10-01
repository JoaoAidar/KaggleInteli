"""
Feature engineering and preprocessing utilities.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def split_columns(X):
    """
    Split DataFrame columns into numeric and categorical.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    tuple of (numeric_columns, categorical_columns)
        Two lists of column names
    """
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"✓ Column split complete:")
    print(f"  - Numeric columns: {len(numeric_columns)}")
    print(f"  - Categorical columns: {len(categorical_columns)}")
    
    return numeric_columns, categorical_columns


def build_preprocessor(numeric_columns, categorical_columns):
    """
    Build a ColumnTransformer for preprocessing.
    
    Numeric pipeline:
        - SimpleImputer(strategy="median")
        - StandardScaler()
    
    Categorical pipeline:
        - SimpleImputer(strategy="most_frequent")
        - OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    
    Parameters
    ----------
    numeric_columns : list
        List of numeric column names
    categorical_columns : list
        List of categorical column names
        
    Returns
    -------
    ColumnTransformer
        Configured preprocessor
    """
    # Numeric preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', 
                                  min_frequency=10, 
                                  sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ],
        remainder='drop'
    )
    
    print(f"✓ Preprocessor built:")
    print(f"  - Numeric features: {len(numeric_columns)}")
    print(f"  - Categorical features: {len(categorical_columns)}")
    
    return preprocessor

