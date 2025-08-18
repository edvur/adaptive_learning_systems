"""
feature_engineering.py - Module for Feature Engineering and Transformations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for Feature Engineering"""
    
    def __init__(self):
        self.scaler = None
        self.selector = None
        self.selected_features = None
        self.feature_names_after_engineering = None
        self.scaler_feature_names = None
        
    def create_advanced_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create advanced Features based on Learning Style Theory"""
        X_enhanced = X.copy()
        
        # Basic aggregations
        X_enhanced['total_engagement'] = X.sum(axis=1)
        X_enhanced['avg_engagement'] = X.mean(axis=1)
        X_enhanced['std_engagement'] = X.std(axis=1)
        X_enhanced['max_activity'] = X.max(axis=1)
        X_enhanced['min_activity'] = X.min(axis=1)
        
        # Activity diversity
        X_enhanced['activities_used'] = (X > 0).sum(axis=1)
        X_enhanced['activity_entropy'] = self._calculate_entropy(X)
        
        # Ratio-Features for Perception (Sensing vs Intuitive)
        X_enhanced['concrete_abstract_ratio'] = (X['Concrete material'] + 1) / (X['Abstract materiale'] + 1)
        X_enhanced['practical_theoretical_ratio'] = (
            (X['Exercises submit'] + X['Quiz submitted'] + 1) / 
            (X['Reading file'] + X['Abstract materiale'] + 1)
        )
        
        # Features for Input (Visual vs Verbal)
        X_enhanced['visual_text_ratio'] = (X['Visual Materials'] + 1) / (X['Reading file'] + 1)
        X_enhanced['video_engagement_rate'] = (
            (X['playing'] + X['paused']) / 
            (X['playing'] + X['paused'] + X['unstarted'] + 1)
        )
        X_enhanced['visual_preference_score'] = (
            X['Visual Materials'] + X['playing'] * 0.5
        ) / (X_enhanced['total_engagement'] + 1)
        
        # Features for Understanding (Sequential vs Global)
        X_enhanced['overview_depth_ratio'] = X['Course overview'] / (X_enhanced['total_engagement'] + 1)
        X_enhanced['completion_rate'] = (
            (X['Exercises submit'] + X['Quiz submitted']) / 
            (X['Course overview'] + 1)
        )
        X_enhanced['structured_learning_score'] = (
            X['Course overview'] * 0.3 + 
            X['Reading file'] * 0.3 + 
            X['Exercises submit'] * 0.4
        ) / (X_enhanced['total_engagement'] + 1)
        
        # Interaction features (polynomial features)
        X_enhanced['reading_visual_product'] = X['Reading file'] * X['Visual Materials']
        X_enhanced['exercise_quiz_product'] = X['Exercises submit'] * X['Quiz submitted']
        X_enhanced['concrete_visual_product'] = X['Concrete material'] * X['Visual Materials']
        
        # Square Features for important variables
        for col in ['Reading file', 'Visual Materials', 'Exercises submit']:
            if col in X.columns:
                X_enhanced[f'{col}_squared'] = X[col] ** 2
        
        # Time-based features (simulated)
        X_enhanced['early_engagement'] = X['Course overview'] + X['Reading file']
        X_enhanced['late_engagement'] = X['Exercises submit'] + X['Quiz submitted']
        X_enhanced['progression_rate'] = (
            X_enhanced['late_engagement'] / 
            (X_enhanced['early_engagement'] + 1)
        )
        
        # Learning style-specific composite scores
        X_enhanced['active_learning_score'] = (
            X['Exercises submit'] * 0.4 + 
            X['Quiz submitted'] * 0.3 + 
            X['Self-assessment'] * 0.3
        )
        
        X_enhanced['reflective_learning_score'] = (
            X['Reading file'] * 0.5 + 
            X['Abstract materiale'] * 0.3 + 
            X['Course overview'] * 0.2
        )
        
        # Log transformations for skewed distributions
        for col in ['total_engagement', 'Reading file', 'Visual Materials']:
            if col in X_enhanced.columns:
                X_enhanced[f'{col}_log'] = np.log1p(X_enhanced[col])
        
        # Save feature names after engineering
        self.feature_names_after_engineering = list(X_enhanced.columns)
        
        logger.info(f"Features extended from {X.shape[1]} to {X_enhanced.shape[1]}")
        
        return X_enhanced
    
    def _calculate_entropy(self, X: pd.DataFrame) -> pd.Series:
        """Calculate Entropy of Activity Distribution"""
        # Normalized rows
        X_norm = X.div(X.sum(axis=1) + 1e-10, axis=0)
        # Calculate Entropy
        entropy = -(X_norm * np.log2(X_norm + 1e-10)).sum(axis=1)
        return entropy
    
    def select_features(self, X: pd.DataFrame, y: pd.DataFrame, 
                       method: str = 'hybrid', 
                       n_features: int = 30) -> pd.DataFrame:
        """Select the best features"""
        
        if method == 'kbest':
            selector = SelectKBest(f_classif, k=n_features)
            
        elif method == 'mutual_info':
            selector = SelectKBest(
                lambda X, y: mutual_info_classif(X, y, random_state=42), 
                k=n_features
            )
            
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            
        elif method == 'hybrid':
            # Combine multiple methods
            return self._hybrid_feature_selection(X, y, n_features)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit selector on first label (for consistency)
        X_selected = selector.fit_transform(X, y.iloc[:, 0])
        
        # Save selected features
        self.selector = selector
        self.selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected features: {len(self.selected_features)}")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def _hybrid_feature_selection(self, X: pd.DataFrame, y: pd.DataFrame, 
                                 n_features: int) -> pd.DataFrame:
        """Hybrid feature selection"""
        scores = {}
        
        # 1. F-Score for every label
        for i, label in enumerate(y.columns):
            f_scores, _ = f_classif(X, y.iloc[:, i])
            scores[f'f_score_{label}'] = f_scores
        
        # 2. Mutual Information for every label
        for i, label in enumerate(y.columns):
            mi_scores = mutual_info_classif(X, y.iloc[:, i], random_state=42)
            scores[f'mi_{label}'] = mi_scores
        
        # 3. Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        for i, label in enumerate(y.columns):
            rf.fit(X, y.iloc[:, i])
            scores[f'rf_{label}'] = rf.feature_importances_
        
        # Combine scores
        score_df = pd.DataFrame(scores, index=X.columns)
        
        # Normalized scores
        score_df_norm = (score_df - score_df.min()) / (score_df.max() - score_df.min())
        
        # Average score
        avg_scores = score_df_norm.mean(axis=1)
        
        # Select top-N features
        top_features = avg_scores.nlargest(n_features).index.tolist()
        self.selected_features = top_features
        
        logger.info(f"Hybrid selection: {len(top_features)} features selected")
        
        return X[top_features]
    
    def scale_features(self, X: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
        """Scale features"""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        self.scaler_feature_names = list(X.columns)
        # Fit and transform the data
        if self.scaler is None:
            raise ValueError("Scaler must be initialized!")
        if self.scaler_feature_names is None:
            raise ValueError("Feature names must be set first!")
        
        X_scaled = self.scaler.fit_transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data with saved transformations"""
        X_transformed = X.copy()
        
        # Feature Selection (if available)
        if self.selector is not None and self.selected_features is not None:
            # Make sure that all expected features are available
            missing_features = set(self.selected_features) - set(X_transformed.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}. "
                               "Make sure that create_advanced_features() "
                               "is called before!")
            X_transformed = X_transformed[self.selected_features]
        
        # Scaling (if available)
        if self.scaler is not None:
            if hasattr(self, 'scaler_feature_names') and self.scaler_feature_names is not None:
                # Make sure that all expected features are available in right order
                X_transformed = X_transformed[self.scaler_feature_names]
            
            X_transformed = pd.DataFrame(
                self.scaler.transform(X_transformed), 
                columns=self.scaler_feature_names, 
                index=X_transformed.index
            )
        
        return X_transformed

class FeaturePipeline:
    """Pipeline for complete feature transformation"""
    
    def __init__(self, 
                 create_features: bool = True,
                 select_features: bool = True,
                 scale_features: bool = True,
                 selection_method: str = 'hybrid',
                 n_features: int = 35,
                 scaling_method: str = 'robust'):
        
        self.create_features = create_features
        self.select_features = select_features
        self.scale_features = scale_features
        self.selection_method = selection_method
        self.n_features = n_features
        self.scaling_method = scaling_method
        
        self.engineer = FeatureEngineer()
        self.is_fitted = False
        
    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        X_transformed = X.copy()
        
        # 1. Feature Engineering
        if self.create_features:
            X_transformed = self.engineer.create_advanced_features(X_transformed)
        
        # 2. Feature Selection
        if self.select_features:
            X_transformed = self.engineer.select_features(
                X_transformed, y, 
                method=self.selection_method,
                n_features=self.n_features
            )
        
        # 3. Feature Scaling
        if self.scale_features:
            X_transformed = self.engineer.scale_features(
                X_transformed, 
                method=self.scaling_method
            )
        
        self.is_fitted = True
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted in fit_transform first!")
        
        X_transformed = X.copy()
        
        # 1. Feature Engineering (MUST be done every time!)
        if self.create_features:
            X_transformed = self.engineer.create_advanced_features(X_transformed)
        
        # 2. Feature selection and scaling (use saved parameters)
        if self.select_features or self.scale_features:
            X_transformed = self.engineer.transform(X_transformed)
        
        return X_transformed

if __name__ == "__main__":
    # Test
    from data_loader_module import load_3label_data
    
    data = load_3label_data()
    
    pipeline = FeaturePipeline()
    X_train_transformed = pipeline.fit_transform(data['X_train'], data['y_train'])
    X_val_transformed = pipeline.transform(data['X_val'])
    
    print(f"Train transformed: {X_train_transformed.shape}")
    print(f"Val transformed: {X_val_transformed.shape}")
    print(f"Features match: {X_train_transformed.shape[1] == X_val_transformed.shape[1]}")