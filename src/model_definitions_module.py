"""
model_definitions.py - Optimized model definitions for learning style classification
Focus on the best models for tabular data with class imbalance
"""

from typing import Dict, List, Any
import numpy as np

# Classical models (only the most important ones)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Tree-based models (the best for this dataset)
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

# Imbalanced Learning
from imblearn.ensemble import BalancedRandomForestClassifier

# XGBoost and LightGBM (optional, but recommended)
try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("XGBoost/LightGBM not installed. Using standard models.")

class ModelDefinitions:
    """Optimized model definitions for learning style dataset"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def get_simple_models(self) -> Dict[str, Any]:
        """
        Only 2 baseline models:
        - Logistic Regression: Baseline, interpretable
        - Gaussian Naive Bayes: Fast, good for initial tests
        """
        return {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',  # Important for imbalanced data
                random_state=self.random_state
            ),
            
            'GaussianNB': GaussianNB()
        }
    
    def get_ensemble_models(self) -> Dict[str, Any]:
        """
        Top 3 ensemble models for this dataset:
        1. BalancedRandomForest: Specifically for imbalanced data
        2. HistGradientBoosting: Fast and efficient for tabular data
        3. GradientBoosting: Proven for structured data
        """
        return {
            'BalancedRandomForest': BalancedRandomForestClassifier(
                n_estimators=300,  # Reduced for 983 samples
                max_depth=10,      # Reduced to prevent overfitting
                min_samples_split=10,  # Increased to prevent overfitting
                sampling_strategy='all',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'HistGradientBoosting': HistGradientBoostingClassifier(
                max_iter=200,      # Reduced for small dataset
                max_depth=8,       # Reduced to prevent overfitting
                learning_rate=0.1, # Increased for faster convergence
                l2_regularization=0.2,  # Increased regularization
                random_state=self.random_state
            ),
            
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,  # Reduced for small dataset
                max_depth=4,       # Reduced to prevent overfitting
                learning_rate=0.1, # Increased for faster convergence
                subsample=0.8,
                min_samples_split=15,  # Increased to prevent overfitting
                random_state=self.random_state
            )
        }
    
    def get_advanced_models(self) -> Dict[str, Any]:
        """
        XGBoost and LightGBM - The best for tabular data
        Only if installed
        """
        if not ADVANCED_MODELS_AVAILABLE:
            return {}
        
        return {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300,    # Reduced for small dataset
                max_depth=4,         # Reduced to prevent overfitting
                learning_rate=0.1,   # Increased for faster convergence
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.2,           # Increased regularization
                reg_alpha=0.2,       # Increased L1 regularization
                reg_lambda=1.5,      # Increased L2 regularization
                scale_pos_weight=3,  # For imbalanced classes
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=300,    # Reduced for small dataset
                num_leaves=31,       # Reduced to prevent overfitting
                max_depth=6,         # Reduced to prevent overfitting
                learning_rate=0.1,   # Increased for faster convergence
                feature_fraction=0.8, # Slightly increased
                bagging_fraction=0.8, # Slightly increased
                bagging_freq=5,
                lambda_l1=0.2,       # Increased L1 regularization
                lambda_l2=0.2,       # Increased L2 regularization
                min_child_samples=30, # Increased to prevent overfitting
                class_weight='balanced',
                random_state=self.random_state,
                verbose=-1
            )
        }
    
    def get_all_models(self) -> Dict[str, Any]:
        """Returns all available models"""
        all_models = {}
        all_models.update(self.get_simple_models())
        all_models.update(self.get_ensemble_models())
        all_models.update(self.get_advanced_models())
        return all_models
    
    def get_quick_test_models(self) -> Dict[str, Any]:
        """Fast models for initial tests"""
        return {
            'LogisticRegression': self.get_simple_models()['LogisticRegression'],
            'GaussianNB': self.get_simple_models()['GaussianNB']
        }
    
    def get_best_models_for_imbalanced(self) -> Dict[str, Any]:
        """
        The best models 
        Based on dataset characteristics:
        - Imbalanced classes (Perception 3.35:1, Input 3.70:1)
        - Small sample size (983 samples)
        - Tabular data with 33+ features
        """
        models = {
            # Best for imbalanced data
            'BalancedRandomForest': self.get_ensemble_models()['BalancedRandomForest'],
            
            # Fast and efficient
            'HistGradientBoosting': self.get_ensemble_models()['HistGradientBoosting']
        }
        
        # Add XGBoost/LightGBM if available (very good for tabular data)
        if ADVANCED_MODELS_AVAILABLE:
            models['XGBoost'] = self.get_advanced_models()['XGBoost']
            models['LightGBM'] = self.get_advanced_models()['LightGBM']
        else:
            # Fallback to GradientBoosting
            models['GradientBoosting'] = self.get_ensemble_models()['GradientBoosting']
        
        return models
    
    def get_hyperparameter_space(self, model_name: str) -> Dict[str, Any]:
        """
        Hyperparameter spaces for the most important models
        Reduced to relevant parameters
        """
        
        param_spaces = {
            'BalancedRandomForest': {
                'n_estimators': [200, 300, 400],  # Reduced range
                'max_depth': [6, 8, 10],          # Reduced range
                'min_samples_split': [10, 15, 20], # Increased range
                'sampling_strategy': ['all', 'not majority']
            },
            
            'HistGradientBoosting': {
                'max_iter': [150, 200, 300],      # Reduced range
                'max_depth': [6, 8, 10],          # Reduced range
                'learning_rate': [0.05, 0.1, 0.15], # Adjusted range
                'l2_regularization': [0.1, 0.2, 0.3] # Increased regularization
            },
            
            'XGBoost': {
                'n_estimators': [200, 300, 400],  # Reduced range
                'max_depth': [3, 4, 5],           # Reduced range
                'learning_rate': [0.05, 0.1, 0.15], # Adjusted range
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'scale_pos_weight': [2, 3, 4]    # Tighter range
            },
            
            'LightGBM': {
                'n_estimators': [200, 300, 400],  # Reduced range
                'num_leaves': [20, 31, 40],       # Reduced range
                'max_depth': [4, 6, 8],           # Reduced range
                'learning_rate': [0.05, 0.1, 0.15], # Adjusted range
                'feature_fraction': [0.7, 0.8, 0.9], # Adjusted range
                'bagging_fraction': [0.7, 0.8, 0.9]  # Adjusted range
            }
        }
        
        return param_spaces.get(model_name, {})
    
    def get_model_description(self, model_name: str) -> str:
        """Returns a description for each model"""
        descriptions = {
            'LogisticRegression': "Baseline - Fast, interpretable, linear",
            'GaussianNB': "Probabilistic - Fast, simple assumptions",
            'BalancedRandomForest': "Best choice for imbalanced data - Robust, Feature Importance",
            'HistGradientBoosting': "Fastest boosting - Efficient for large datasets",
            'GradientBoosting': "Classic boosting - Good for structured data",
            'XGBoost': "State-of-the-art - Excellent for tabular data",
            'LightGBM': "Fastest gradient boosting - Very efficient"
        }
        return descriptions.get(model_name, "No description available")

if __name__ == "__main__":
    # Test
    model_defs = ModelDefinitions()
    
    print("=== MODEL OVERVIEW ===\n")
    
    print("1. Baseline Models (2):")
    for name in model_defs.get_simple_models().keys():
        print(f"   - {name}: {model_defs.get_model_description(name)}")
    
    print("\n2. Ensemble Models (3):")
    for name in model_defs.get_ensemble_models().keys():
        print(f"   - {name}: {model_defs.get_model_description(name)}")
    
    print("\n3. Advanced Models (if installed):")
    advanced = model_defs.get_advanced_models()
    if advanced:
        for name in advanced.keys():
            print(f"   - {name}: {model_defs.get_model_description(name)}")
    else:
        print("   - XGBoost/LightGBM not installed")
    
    print("\n4. Recommended Models:")
    for name in model_defs.get_best_models_for_imbalanced().keys():
        print(f"   - {name}")
    
    print(f"\nTotal: {len(model_defs.get_all_models())} models available")