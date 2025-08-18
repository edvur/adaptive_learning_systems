"""
data_loader.py - module for data loading and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import logging
from config import get_config

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preprocessing the learning style data"""
    
    def __init__(self, csms_path: str = None, cshs_path: str = None):
        config = get_config()
        self.csms_path = csms_path or config.get_data_path('csms')
        self.cshs_path = cshs_path or config.get_data_path('cshs')
        self.target_cols = ['Perception', 'Input', 'Understanding']
        self.feature_cols = [
            'Course overview', 'Reading file', 'Abstract materiale', 
            'Concrete material', 'Visual Materials', 'Self-assessment',
            'Exercises submit', 'Quiz submitted', 'playing', 
            'paused', 'unstarted', 'buffering'
        ]
        self.mapping = {
            'SEN': 1, 'INT': 0,  # Perception
            'VIS': 1, 'VRB': 0,  # Input
            'SEQ': 1, 'GLO': 0   # Understanding
        }
        
    def load_raw_data(self) -> pd.DataFrame:
        """Loads raw data from Excel files"""
        logger.info("Loading raw data...")
        csms = pd.read_excel(self.csms_path)
        cshs = pd.read_excel(self.cshs_path)
        
        # Combine datasets
        df = pd.concat([csms, cshs], ignore_index=True)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        logger.info(f"Raw data loaded: {len(df)} samples")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cleans data and extracts features/labels"""
        # Remove rows with missing target values 
        df_clean = df.dropna(subset=self.target_cols)
        
        # Extract Features
        X = df_clean[self.feature_cols].fillna(0)
        
        # Convert Labels
        y = df_clean[self.target_cols].copy()
        for col in self.target_cols:
            y[col] = y[col].replace(self.mapping)
        
        logger.info(f"Cleaned data: {len(X)} samples, {X.shape[1]} features")
        
        # Remove duplicates based on features
        duplicate_mask = X.duplicated()
        if duplicate_mask.sum() > 0:
            logger.warning(f"Removing {duplicate_mask.sum()} duplicates")
            X = X[~duplicate_mask]
            y = y[~duplicate_mask]
        
        return X, y
    
    def get_class_distribution(self, y: pd.DataFrame) -> Dict:
        """Calculates class distribution"""
        distribution = {}
        for col in y.columns:
            counts = y[col].value_counts()
            distribution[col] = {
                'counts': counts.to_dict(),
                'ratio': counts.max() / counts.min() if len(counts) > 1 else 1
            }
        return distribution
    
    def create_data_splits(self, X: pd.DataFrame, y: pd.DataFrame, 
                          test_size: float = 0.2, 
                          val_size: float = 0.15,
                          random_state: int = 42) -> Dict:
        """Creates train/validation/test splits"""
        # First train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second train vs validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        
        return splits
    
    def load_and_prepare(self) -> Dict:
        """Main method: loads and prepares all data"""
        # Load raw data
        df = self.load_raw_data()
        
        # Clean data
        X, y = self.clean_data(df)
        
        # Class distribution
        distribution = self.get_class_distribution(y)
        logger.info("Class distribution:")
        for label, info in distribution.items():
            logger.info(f"  {label}: {info['counts']}, Ratio: {info['ratio']:.2f}")
        
        # Create splits
        splits = self.create_data_splits(X, y)
        
        # Add metadata
        splits['class_distribution'] = distribution
        splits['feature_names'] = list(X.columns)
        splits['label_names'] = list(y.columns)
        
        return splits

# Convenience functions for direct import
def load_3label_data():
    """Loads 3-label data"""
    loader = DataLoader()
    return loader.load_and_prepare()

if __name__ == "__main__":
    # Test
    data = load_3label_data()
    print(f"Data loaded successfully!")
    print(f"Features: {len(data['feature_names'])}")
    print(f"Labels: {data['label_names']}")
