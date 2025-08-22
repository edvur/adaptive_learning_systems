"""
model_training.py - Module for model training and hyperparameter optimization
"""

import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
import joblib
import pickle
import time
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for model training and optimization"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models = {}
        self.training_history = []
    
    def clone_model(self, model):
        """Utility method to clone a model consistently"""
        return model.__class__(**model.get_params())
        
    def train_single_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                          model_name: str = "Model") -> Tuple[Any, float, Dict]:
        """Trains a single model for one label"""
        start_time = time.time()
        
        # Training
        model.fit(X_train, y_train)
        
        # Cross-Validation Score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Training metrics
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        training_time = time.time() - start_time
        
        metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_accuracy,
            'training_time': training_time
        }
        
        logger.info(f"{model_name} - CV: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
        
        return model, metrics['cv_mean'], metrics
    
    def train_with_resampling(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                             resampling_method: str = 'smote') -> Tuple[Any, pd.DataFrame, pd.Series]:
        """Trains model with resampling for imbalanced data"""
        
        if resampling_method == 'smote':
            resampler = SMOTE(random_state=self.random_state)
        elif resampling_method == 'borderline':
            resampler = BorderlineSMOTE(random_state=self.random_state, k_neighbors=5)
        elif resampling_method == 'smotetomek':
            resampler = SMOTETomek(random_state=self.random_state)
        else:
            return model.fit(X_train, y_train), X_train, y_train
        
        # Resampling
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        
        # Training
        model.fit(X_resampled, y_resampled)
        
        logger.info(f"Resampling: {len(X_train)} -> {len(X_resampled)} samples")
        
        return model, X_resampled, y_resampled
    
    def train_multi_output(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                          y_train: pd.DataFrame, use_resampling: Dict[str, bool] = None) -> Dict:
        """Trains models for multi-label classification"""
        
        trained_models = {}
        
        for label in y_train.columns:
            logger.info(f"\nTraining models for {label}...")
            
            label_models = {}
            label_metrics = {}
            
            # Check if label needs resampling
            needs_resampling = False
            if use_resampling and label in use_resampling:
                needs_resampling = use_resampling[label]
            else:
                # Auto-detect based on class distribution
                value_counts = y_train[label].value_counts()
                if len(value_counts) > 1:
                    ratio = value_counts.max() / value_counts.min()
                    needs_resampling = ratio > 2.0
            
            for model_name, model in models.items():
                try:
                    if needs_resampling:
                        trained_model, _, _ = self.train_with_resampling(
                            model, X_train, y_train[label], 'borderline'
                        )
                    else:
                        trained_model, cv_score, metrics = self.train_single_model(
                            model, X_train, y_train[label], f"{model_name}-{label}"
                        )
                    
                    label_models[model_name] = trained_model
                    label_metrics[model_name] = metrics if not needs_resampling else {'cv_mean': 0}
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} for {label}: {str(e)}")
            
            # Select best model for this label
            if label_metrics:
                best_model_name = max(label_metrics.items(), key=lambda x: x[1].get('cv_mean', 0))[0]
                trained_models[label] = {
                    'model': label_models[best_model_name],
                    'model_name': best_model_name,
                    'metrics': label_metrics[best_model_name]
                }
                
                logger.info(f"Best model for {label}: {best_model_name}")
        
        return trained_models
    
    def hyperparameter_tuning(self, model, param_grid: Dict, X_train: pd.DataFrame, 
                             y_train: pd.Series, cv: int = 5, 
                             search_type: str = 'random', n_iter: int = 50) -> Tuple[Any, Dict]:
        """Hyperparameter optimization"""
        
        logger.info(f"Starting hyperparameter tuning ({search_type})...")
        
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, 
                scoring='accuracy', n_jobs=-1, verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv,
                scoring='accuracy', n_jobs=-1, verbose=1,
                random_state=self.random_state
            )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_, {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
    
    def ensemble_training(self, base_models: List[Tuple[str, Any]], 
                         X_train: pd.DataFrame, y_train: pd.DataFrame,
                         ensemble_type: str = 'voting') -> Any:
        """Trains ensemble models"""
        
        if ensemble_type == 'voting':
            from sklearn.ensemble import VotingClassifier
            
            ensemble_models = {}
            
            for label in y_train.columns:
                # Train base models for this label
                estimators = []
                for name, model in base_models:
                    model_clone = self.clone_model(model)
                    model_clone.fit(X_train, y_train[label])
                    estimators.append((name, model_clone))
                
                # Create voting classifier
                voting = VotingClassifier(estimators=estimators, voting='soft')
                voting.fit(X_train, y_train[label])
                
                ensemble_models[label] = voting
            
            return ensemble_models
            
        elif ensemble_type == 'stacking':
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            
            ensemble_models = {}
            
            for label in y_train.columns:
                # Use logistic regression as meta-learner
                stacking = StackingClassifier(
                    estimators=base_models,
                    final_estimator=LogisticRegression(random_state=self.random_state),
                    cv=5,
                )
                stacking.fit(X_train, y_train[label])
                
                ensemble_models[label] = stacking
            
            return ensemble_models
    
    def parallel_training(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                         y_train: pd.DataFrame, n_jobs: int = -1) -> Dict:
        """Parallel training of multiple models"""
        
        def train_model_wrapper(args):
            model_name, model, X, y, label = args
            try:
                trained_model, cv_score, metrics = self.train_single_model(
                    model, X, y, f"{model_name}-{label}"
                )
                return label, model_name, trained_model, metrics
            except Exception as e:
                logger.error(f"Error in parallel training: {str(e)}")
                return None
        
        # Prepare tasks
        tasks = []
        for label in y_train.columns:
            for model_name, model in models.items():
                tasks.append((model_name, model, X_train, y_train[label], label))
        
        # Parallel execution
        results = {}
        with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            futures = [executor.submit(train_model_wrapper, task) for task in tasks]
            
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per task
                    if result:
                        label, model_name, model, metrics = result
                        if label not in results:
                            results[label] = {}
                        results[label][model_name] = {
                            'model': model,
                            'metrics': metrics
                        }
                except TimeoutError:
                    logger.error("Training task timed out (5 minutes)")
                    continue
                except Exception as e:
                    logger.error(f"Training task failed: {e}")
                    continue
        
        return results
    
    def save_models(self, models: Dict, filepath_prefix: str = "models/model"):
        """Saves trained models"""
        for label, model_info in models.items():
            if isinstance(model_info, dict) and 'model' in model_info:
                filepath = f"{filepath_prefix}_{label}.pkl"
                joblib.dump(model_info['model'], filepath)
                logger.info(f"Model saved: {filepath}")
            else:
                filepath = f"{filepath_prefix}_{label}.pkl"
                joblib.dump(model_info, filepath)
    
    def load_models(self, labels: List[str], filepath_prefix: str = "models/model") -> Dict:
        """Loads saved models"""
        models = {}
        for label in labels:
            filepath = f"{filepath_prefix}_{label}.pkl"
            try:
                models[label] = joblib.load(filepath)
                logger.info(f"Model loaded: {filepath}")
            except FileNotFoundError:
                logger.warning(f"Model file not found: {filepath}")
            except (EOFError, pickle.UnpicklingError) as e:
                logger.error(f"Model file corrupted: {filepath} - {e}")
            except Exception as e:
                logger.error(f"Could not load model {filepath}: {e}")
        return models

class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self, feature_pipeline, model_definitions, random_state: int = 42):
        self.feature_pipeline = feature_pipeline
        self.model_definitions = model_definitions
        self.trainer = ModelTrainer(random_state)
        
    def run(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: pd.DataFrame, y_val: pd.DataFrame,
            model_selection: str = 'quick', tune_hyperparameters: bool = False) -> Dict:
        """Runs complete training pipeline"""
        
        # Feature transformation
        logger.info("Transforming features...")
        X_train_transformed = self.feature_pipeline.fit_transform(X_train, y_train)
        X_val_transformed = self.feature_pipeline.transform(X_val)
        
        # Model selection
        if model_selection == 'quick':
            models = self.model_definitions.get_quick_test_models()
        elif model_selection == 'imbalanced':
            models = self.model_definitions.get_best_models_for_imbalanced()
        elif model_selection == 'all':
            models = self.model_definitions.get_all_models()
        else:
            raise ValueError(f"Unknown model selection: {model_selection}")
        
        # Training
        logger.info(f"Training {len(models)} models...")
        
        # Determine which labels need resampling
        use_resampling = {}
        for label in y_train.columns:
            value_counts = y_train[label].value_counts()
            if len(value_counts) > 1:
                ratio = value_counts.max() / value_counts.min()
                use_resampling[label] = ratio > 2.0
        
        # Train models
        trained_models = self.trainer.train_multi_output(
            models, X_train_transformed, y_train, use_resampling
        )
        
        # Hyperparameter tuning if requested
        if tune_hyperparameters:
            logger.info("Starting hyperparameter tuning...")
            for label, model_info in trained_models.items():
                model_name = model_info['model_name']
                param_space = self.model_definitions.get_hyperparameter_space(model_name)
                
                if param_space:
                    tuned_model, tuning_results = self.trainer.hyperparameter_tuning(
                        model_info['model'], param_space, 
                        X_train_transformed, y_train[label],
                        search_type='random', n_iter=30
                    )
                    model_info['model'] = tuned_model
                    model_info['tuning_results'] = tuning_results
        
        return trained_models

if __name__ == "__main__":
    # Test
    from data_loader_module import load_3label_data
    from feature_engineering_module import FeaturePipeline
    from model_definitions_module import ModelDefinitions
    
    # Load data
    data = load_3label_data()
    
    # Setup pipeline
    feature_pipeline = FeaturePipeline()
    model_defs = ModelDefinitions()
    
    training_pipeline = TrainingPipeline(feature_pipeline, model_defs)
    
    # Run training
    trained_models = training_pipeline.run(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        model_selection='quick'
    )
    
    print(f"Training completed! Models trained: {list(trained_models.keys())}")
