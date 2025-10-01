"""
Data loading and saving utilities for Kaggle competition.
"""
import os
import pandas as pd


def load_data(data_dir="data"):
    """
    Load train, test, and sample submission datasets.
    
    Parameters
    ----------
    data_dir : str, default="data"
        Directory containing the CSV files
        
    Returns
    -------
    tuple of (train_df, test_df, sample_submission_df)
        Three pandas DataFrames
        
    Raises
    ------
    FileNotFoundError
        If any required CSV file is missing
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sample_path = os.path.join(data_dir, "sample_submission.csv")
    
    # Check if files exist
    for path, name in [(train_path, "train.csv"), 
                       (test_path, "test.csv"), 
                       (sample_path, "sample_submission.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file '{name}' not found in '{data_dir}' directory. "
                f"Please ensure all data files are present."
            )
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        sample_submission_df = pd.read_csv(sample_path)
        
        print(f"✓ Data loaded successfully from '{data_dir}'")
        print(f"  - Train shape: {train_df.shape}")
        print(f"  - Test shape: {test_df.shape}")
        print(f"  - Sample submission shape: {sample_submission_df.shape}")
        
        return train_df, test_df, sample_submission_df
        
    except Exception as e:
        raise RuntimeError(f"Error loading data files: {str(e)}")


def get_target_name(sample_submission_df):
    """
    Extract target column name from sample submission.
    
    The sample submission format is: [ID_column, target_column]
    This function returns the name of the second column.
    
    Parameters
    ----------
    sample_submission_df : pd.DataFrame
        Sample submission DataFrame
        
    Returns
    -------
    str
        Name of the target column
    """
    columns = sample_submission_df.columns.tolist()
    if len(columns) < 2:
        raise ValueError(
            f"Sample submission must have at least 2 columns (ID and target). "
            f"Found: {columns}"
        )
    
    target_name = columns[1]
    print(f"✓ Target column identified: '{target_name}'")
    return target_name


def save_submission(predictions_df, path="submission.csv"):
    """
    Save predictions DataFrame to CSV in submission format.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions matching sample_submission format
    path : str, default="submission.csv"
        Output file path
    """
    try:
        predictions_df.to_csv(path, index=False)
        print(f"✓ Submission saved to '{path}'")
        print(f"  - Shape: {predictions_df.shape}")
        print(f"  - Columns: {predictions_df.columns.tolist()}")
    except Exception as e:
        raise RuntimeError(f"Error saving submission file: {str(e)}")

