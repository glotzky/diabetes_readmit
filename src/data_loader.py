"""
Data Loading Module for Diabetes 130-US Hospitals Dataset
==========================================================

Clinical Context:
-----------------
This dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals.
Each record represents a diabetic patient encounter with hospital stay of 1-14 days.

The goal is to predict early readmission (within 30 days), which is crucial because:
1. Readmissions are costly for hospitals (Medicare penalties for excess readmissions)
2. They indicate potential gaps in care quality or discharge planning
3. Early identification allows targeted interventions (follow-up calls, home visits)

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings


def load_diabetes_data(
    data_path: Optional[str] = None,
    use_ucimlrepo: bool = True
) -> pd.DataFrame:
    """
    Load the Diabetes 130-US Hospitals dataset.
    
    Parameters
    ----------
    data_path : str, optional
        Path to local CSV file. If None and use_ucimlrepo=True, 
        downloads from UCI ML Repository.
    use_ucimlrepo : bool, default=True
        Whether to use ucimlrepo package to fetch data directly.
    
    Returns
    -------
    pd.DataFrame
        Raw diabetes dataset with all 50+ features.
    
    Clinical Note:
    --------------
    The dataset contains sensitive information (age, gender, race) and 
    should be handled according to HIPAA guidelines in production settings.
    """
    
    if data_path and Path(data_path).exists():
        print(f"Loading data from local file: {data_path}")
        df = pd.read_csv(data_path)
        
    elif use_ucimlrepo:
        print("Fetching data from UCI ML Repository...")
        try:
            from ucimlrepo import fetch_ucirepo
            
            # Fetch the diabetes dataset (ID: 296)
            diabetes_data = fetch_ucirepo(id=296)
            
            # Combine features and targets into single DataFrame
            df = pd.concat([
                diabetes_data.data.features,
                diabetes_data.data.targets
            ], axis=1)
            
            print(f"Successfully loaded {len(df):,} patient encounters")
            
        except ImportError:
            raise ImportError(
                "ucimlrepo package not installed. "
                "Run: pip install ucimlrepo"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from UCI: {str(e)}")
    else:
        raise ValueError(
            "Either provide data_path or set use_ucimlrepo=True"
        )
    
    # Initial data quality report
    _print_data_summary(df)
    
    return df


def _print_data_summary(df: pd.DataFrame) -> None:
    """Print a summary of the loaded dataset."""
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total encounters: {len(df):,}")
    print(f"Total features: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    # Check for readmission distribution
    if 'readmitted' in df.columns:
        print("\nReadmission Distribution:")
        print(df['readmitted'].value_counts())
        
        # Clinical insight
        early_readmit = (df['readmitted'] == '<30').sum()
        total = len(df)
        print(f"\nEarly readmission rate (<30 days): {early_readmit/total*100:.2f}%")
        print("(Industry benchmark: 15-20% for diabetic patients)")
    
    print("="*60 + "\n")


def save_processed_data(
    df: pd.DataFrame,
    output_path: str,
    description: str = "processed_data"
) -> None:
    """
    Save processed DataFrame to CSV with metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed data to save.
    output_path : str
        Path to save the CSV file.
    description : str
        Description for logging purposes.
    """
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved {description}: {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    # Test data loading
    df = load_diabetes_data()
    print(df.head())
    print(df.info())
