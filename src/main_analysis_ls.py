"""
Main Analysis for Learning Style Classification System

This module provides a comprehensive machine learning pipeline for classifying
learning styles across three dimensions: Perception, Input, and Understanding.
It includes feature engineering, model training, ensemble methods, and 
threshold optimization.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import joblib
import logging
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
from config import get_config

# Scientific computing and ML libraries
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.ensemble import (
    VotingClassifier, StackingClassifier, ExtraTreesClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import BorderlineSMOTE
from statsmodels.stats.contingency_tables import mcnemar

# Custom modules
from data_loader_module import DataLoader
from feature_engineering_module import FeaturePipeline
from model_definitions_module import ModelDefinitions
from model_training_module import ModelTrainer

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('learning_style_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Get configuration
config = get_config()
FIGURES_PATH = config.FIGURES_DIR
MODELS_PATH = config.MODELS_DIR  
METRICS_PATH = config.METRICS_DIR
DATA_PATH = config.DATA_DIR


class LearningStyleAnalysis:
    """Main class for comprehensive learning style classification analysis.
    
    This class orchestrates the complete machine learning pipeline for learning
    style classification, including data loading, feature engineering, model 
    training, ensemble creation, and evaluation.
    
    Attributes:
        data_paths (Dict[str, str]): Paths to data files
        random_state (int): Random state for reproducibility
        data (Optional[Dict]): Loaded and processed data
        feature_pipeline (Optional[FeaturePipeline]): Feature engineering pipeline
        results (Dict): Analysis results and metrics
    """
    
    def __init__(self, data_paths: Dict[str, str], random_state: int = 42):
        """Initialize the Learning Style Analysis.
        
        Args:
            data_paths: Dictionary containing paths to CSMS and CSHS data files
            random_state: Random state for reproducible results
        """
        self.data_paths = data_paths
        self.random_state = random_state
        self.data: Optional[Dict] = None
        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.model_definitions: Optional[ModelDefinitions] = None
        self.trainer: Optional[ModelTrainer] = None
        self.results: Dict[str, Any] = {}
        
    def load_data(self):
        """Load and analyze training data from CSMS and CSHS datasets.
        
        Returns:
            dict: Processed data including features, labels, and splits
        """
        print("=" * 60)
        print("1. LOADING AND ANALYZING DATA")
        print("=" * 60)
        
        try:
            data_loader = DataLoader(self.data_paths['csms'], self.data_paths['cshs'])
            self.data = data_loader.load_and_prepare()
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
        
        print(f"\nData loaded!")
        print(f"Features: {len(self.data['feature_names'])}")
        print(f"Labels: {self.data['label_names']}")
        print(f"\nTraining samples: {len(self.data['X_train'])}")
        print(f"Validation samples: {len(self.data['X_val'])}")
        print(f"Test samples: {len(self.data['X_test'])}")
        
        # Class Distribution
        print("\nClass Distribution:")
        for label, info in self.data['class_distribution'].items():
            print(f"  {label}: {info['counts']}, Ratio: {info['ratio']:.2f}")
        
        return self.data
    
    def setup_feature_engineering(self, n_features: int = 40):
        """Set up and apply feature engineering pipeline.
        
        Args:
            n_features (int): Number of features to select
            
        Returns:
            tuple: Transformed training, validation, and test sets
        """
        print("\n" + "=" * 60)
        print("2. FEATURE ENGINEERING")
        print("=" * 60)
        
        self.feature_pipeline = FeaturePipeline(
            create_features=True,
            select_features=True,
            scale_features=True,
            selection_method='hybrid',
            n_features=n_features,
            scaling_method='robust'
        )
        
        # Transform features
        try:
            X_train_transformed = self.feature_pipeline.fit_transform(
                self.data['X_train'], self.data['y_train']
            )
            X_val_transformed = self.feature_pipeline.transform(self.data['X_val'])
            X_test_transformed = self.feature_pipeline.transform(self.data['X_test'])
            logger.info("Feature engineering completed successfully")
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
        
        print(f"\nOriginal features: {self.data['X_train'].shape[1]}")
        print(f"After feature engineering: {X_train_transformed.shape[1]}")
        
        if hasattr(self.feature_pipeline.engineer, 'selected_features') and self.feature_pipeline.engineer.selected_features:
            print(f"\nTop 10 selected features:")
            for i, feat in enumerate(self.feature_pipeline.engineer.selected_features[:10]):
                print(f"  {i+1}. {feat}")
      
        
        self.data['X_train_transformed'] = X_train_transformed
        self.data['X_val_transformed'] = X_val_transformed
        self.data['X_test_transformed'] = X_test_transformed
        
        return X_train_transformed, X_val_transformed, X_test_transformed
    
    def train_baseline_models(self) -> Dict:
        """Train baseline machine learning models.
        
        Returns:
            Dict: Results from baseline model training
        """
        print("\n" + "=" * 60)
        print("3. TRAINING BASELINE MODELS")
        print("=" * 60)
        
        self.model_definitions = ModelDefinitions(self.random_state)
        self.trainer = ModelTrainer(self.random_state)
        
        baseline_models = self.model_definitions.get_quick_test_models()
        print(f"\nTrain {len(baseline_models)} Baseline-Models...")
        
        baseline_results = {}
        
        for label in self.data['label_names']:
            print(f"\nTraining for {label}...")
            label_results = {}
            
            for model_name, model in baseline_models.items():
                # Train model
                trained_model, cv_score, metrics = self.trainer.train_single_model(
                    model, 
                    self.data['X_train_transformed'], 
                    self.data['y_train'][label], 
                    f"{model_name}-{label}"
                )
                
                # Validate
                val_pred = trained_model.predict(self.data['X_val_transformed'])
                val_acc = accuracy_score(self.data['y_val'][label], val_pred)
                val_f1 = f1_score(self.data['y_val'][label], val_pred)
                
                label_results[model_name] = {
                    'model': trained_model,
                    'cv_score': cv_score,
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'metrics': metrics
                }
                
                print(f"  {model_name}: CV={cv_score:.4f}, Val Acc={val_acc:.4f}")
            
            baseline_results[label] = label_results
        
        self.results['baseline'] = baseline_results
        return baseline_results
    
    def train_advanced_models(self) -> Dict:
        """Train advanced machine learning models with imbalanced data handling.
        
        Returns:
            Dict: Results from advanced model training
        """
        print("\n" + "=" * 60)
        print("4. TRAINING ADVANCED MODELS (Tree-based & Boosting)")
        print("=" * 60)
        
        # Advanced Models
        advanced_models = self.model_definitions.get_best_models_for_imbalanced()
        
        # Additional optimized Models - optimized for small dataset
        additional_models = {
            'ExtraTrees_300': ExtraTreesClassifier(
                n_estimators=300, max_depth=10, min_samples_split=15,  # Reduced complexity
                class_weight='balanced', random_state=self.random_state, n_jobs=-1
            ),
            'HistGB_optimized': HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.1, max_depth=8,  # Reduced complexity
                l2_regularization=0.2, random_state=self.random_state  # Increased regularization
            )
        }
        
        # Add XGBoost from model definitions if available
        model_defs_advanced = self.model_definitions.get_advanced_models()
        if 'XGBoost' in model_defs_advanced:
            additional_models['XGBoost_85'] = model_defs_advanced['XGBoost']
        
        advanced_models.update(additional_models)
        print(f"\nTraining {len(advanced_models)} advanced models...")
        
        advanced_results = {}
        
        for label in self.data['label_names']:
            print(f"\nTraining advanced models for {label}...")
            
            # Check if resampling needed
            imbalance_ratio = self.data['class_distribution'][label]['ratio']
            needs_resampling = imbalance_ratio > 2.0
            
            if needs_resampling:
                print(f"  Using BorderlineSMOTE (imbalance ratio: {imbalance_ratio:.2f})")
                resampler = BorderlineSMOTE(random_state=self.random_state, k_neighbors=5)
                X_train_res, y_train_res = resampler.fit_resample(
                    self.data['X_train_transformed'], self.data['y_train'][label]
                )
            else:
                X_train_res = self.data['X_train_transformed']
                y_train_res = self.data['y_train'][label]
            
            label_results = {}
            
            for model_name, model in advanced_models.items():
                try:
                    # Clone model using trainer utility
                    model_clone = self.trainer.clone_model(model)
                    
                    # Train
                    model_clone.fit(X_train_res, y_train_res)
                    
                    # Validate
                    val_pred = model_clone.predict(self.data['X_val_transformed'])
                    val_acc = accuracy_score(self.data['y_val'][label], val_pred)
                    val_f1 = f1_score(self.data['y_val'][label], val_pred)
                    
                    label_results[model_name] = {
                        'model': model_clone,
                        'val_accuracy': val_acc,
                        'val_f1': val_f1
                    }
                    
                    print(f"  {model_name}: Val Acc={val_acc:.4f}, F1={val_f1:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    print(f"  {model_name}: Error - {str(e)}")
            
            advanced_results[label] = label_results
        
        self.results['advanced'] = advanced_results
        return advanced_results
    
    def create_ensemble_models(self) -> Dict:
        """Create ensemble models using voting and stacking methods.
        
        Returns:
            Dict: Results from ensemble model creation
        """
        print("\n" + "=" * 60)
        print("5. CREATING ENSEMBLE MODELS")
        print("=" * 60)
        
        # Select best Models for Ensemble
        best_models_per_label = {}
        
        for label in self.data['label_names']:
            # Combine baseline und advanced results
            all_models = {**self.results['baseline'][label], **self.results['advanced'][label]}
            
            # Sort after Validation Accuracy
            sorted_models = sorted(all_models.items(), 
                                  key=lambda x: x[1]['val_accuracy'], 
                                  reverse=True)
            
            # Select Top 5
            best_models_per_label[label] = sorted_models[:5]
            
            print(f"\nBest Models for {label}:")
            for name, info in best_models_per_label[label]:
                print(f"  {name}: {info['val_accuracy']:.4f}")
        
        # Create Ensembles
        ensemble_results = {}
        
        for label in self.data['label_names']:
            print(f"\nCreate Ensembles for {label}...")
            
            # Extract Models
            estimators = [(name, info['model']) for name, info in best_models_per_label[label]]
            
            # Voting Ensemble (Soft)
            voting_soft = VotingClassifier(estimators=estimators[:3], voting='soft')
            voting_soft.fit(self.data['X_train_transformed'], self.data['y_train'][label])
            
            val_pred_voting = voting_soft.predict(self.data['X_val_transformed'])
            val_acc_voting = accuracy_score(self.data['y_val'][label], val_pred_voting)
            
            # Stacking Ensemble
            stacking = StackingClassifier(
                estimators=estimators[:3],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5
            )
            stacking.fit(self.data['X_train_transformed'], self.data['y_train'][label])
            
            val_pred_stacking = stacking.predict(self.data['X_val_transformed'])
            val_acc_stacking = accuracy_score(self.data['y_val'][label], val_pred_stacking)
            
            ensemble_results[label] = {
                'voting_soft': {'model': voting_soft, 'val_accuracy': val_acc_voting},
                'stacking': {'model': stacking, 'val_accuracy': val_acc_stacking},
                'best_models': best_models_per_label[label]
            }
            
            print(f"  Voting Soft: {val_acc_voting:.4f}")
            print(f"  Stacking: {val_acc_stacking:.4f}")
        
        self.results['ensemble'] = ensemble_results
        return ensemble_results
    
    def optimize_thresholds(self) -> Dict:
        """Optimize decision thresholds for better F1-scores.
        
        Returns:
            Dict: Optimal thresholds for each dimension
        """
        print("\n" + "=" * 60)
        print("6. OPTIMIZING THRESHOLDS")
        print("=" * 60)
        
        optimal_thresholds = {}
        
        for label in self.data['label_names']:
            print(f"\nOptimize Threshold for {label}...")
            
            # Use best Model
            best_model_name, best_model_info = self.results['ensemble'][label]['best_models'][0]
            model = best_model_info['model']
            
            if hasattr(model, 'predict_proba'):
                # Calculate Probabilities
                y_proba = model.predict_proba(self.data['X_val_transformed'])[:, 1]
                
                # Test different Thresholds
                best_threshold = 0.5
                best_f1 = 0
                
                for threshold in np.arange(0.2, 0.8, 0.05):
                    y_pred_threshold = (y_proba > threshold).astype(int)
                    f1 = f1_score(self.data['y_val'][label], y_pred_threshold)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                optimal_thresholds[label] = best_threshold
                
                # New Accuracy with optimal Threshold
                y_pred_optimal = (y_proba > best_threshold).astype(int)
                acc_optimal = accuracy_score(self.data['y_val'][label], y_pred_optimal)
                
                print(f"  Standard (0.5): {best_model_info['val_accuracy']:.4f}")
                print(f"  Optimal ({best_threshold:.2f}): {acc_optimal:.4f}")
                print(f"  Improvement: +{(acc_optimal - best_model_info['val_accuracy'])*100:.1f}%")
        
        self.results['optimal_thresholds'] = optimal_thresholds
        return optimal_thresholds
    
    def evaluate_final_models(self) -> Tuple[Dict, float]:
        """Perform final evaluation on the test set.
        
        Returns:
            Tuple[Dict, float]: Test results and average accuracy
        """
        print("\n" + "=" * 60)
        print("7. FINAL EVALUATION ON TEST SET")
        print("=" * 60)
        
        # Select finale Models
        final_models = {}
        
        for label in self.data['label_names']:
            # Compare all Options
            candidates = [
                ('best_single', self.results['ensemble'][label]['best_models'][0][1]),
                ('voting', self.results['ensemble'][label]['voting_soft']),
                ('stacking', self.results['ensemble'][label]['stacking'])
            ]
            
            # Select based on Validation Accuracy
            best_option = max(candidates, key=lambda x: x[1]['val_accuracy'])
            final_models[label] = best_option
            
            print(f"{label}: {best_option[0]} (Val Acc: {best_option[1]['val_accuracy']:.4f})")
        
        # Test-Evaluation
        test_results = {}
        test_predictions = {}
        
        print("\nFINAL TEST RESULTS:")
        print("="*60)
        
        for label in self.data['label_names']:
            model_type, model_info = final_models[label]
            model = model_info['model']
            
            # Prediction
            if label in self.results['optimal_thresholds'] and hasattr(model, 'predict_proba'):
                # With optimized Threshold
                y_proba = model.predict_proba(self.data['X_test_transformed'])[:, 1]
                y_pred = (y_proba > self.results['optimal_thresholds'][label]).astype(int)
            else:
                y_pred = model.predict(self.data['X_test_transformed'])
            
            test_predictions[label] = y_pred
            
            # Metrics
            accuracy = accuracy_score(self.data['y_test'][label], y_pred)
            f1 = f1_score(self.data['y_test'][label], y_pred)
            
            test_results[label] = {
                'accuracy': accuracy,
                'f1': f1,
                'model_type': model_type
            }
            
            print(f"\n{label} ({model_type}):")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        # Total Results
        avg_accuracy = np.mean([res['accuracy'] for res in test_results.values()])
        avg_f1 = np.mean([res['f1'] for res in test_results.values()])
        
        print("\n" + "="*60)
        print("TOTAL RESULTS:")
        print("="*60)
        print(f"Average Test Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print(f"Average F1-Score: {avg_f1:.4f}")
        
        # Include dimension details in final results
        self.print_performance_metrics()
        
        self.results['test'] = test_results
        self.results['final_models'] = final_models
        self.results['test_predictions'] = test_predictions
        self.results['avg_accuracy'] = avg_accuracy
        self.results['avg_f1'] = avg_f1
        
        return test_results, avg_accuracy
    
    def perform_significance_tests(self):
        """Perform statistical significance tests between best models."""
        print("\n" + "=" * 60)
        print("8. STATISTICAL SIGNIFICANCE TESTING")
        print("=" * 60)
        
        if 'test_predictions' not in self.results or 'ensemble' not in self.results:
            print("Insufficient data for significance testing")
            return
        
        # Perform McNemar's test for each dimension
        significance_results = {}
        
        for label in self.data['label_names']:
            print(f"\nTesting significance for {label} dimension:")
            
            # Compare best individual vs ensemble (if different)
            final_model_type = self.results['test'][label]['model_type']
            
            if final_model_type != 'best_single':
                # Get predictions from best individual model
                best_individual_model = self.results['ensemble'][label]['best_models'][0][1]['model']
                individual_preds = best_individual_model.predict(self.data['X_test_transformed'])
                
                # Get final model predictions
                final_preds = self.results['test_predictions'][label]
                y_true = self.data['y_test'][label]
                
                # Perform significance test
                is_significant, p_value = self.test_model_significance(
                    individual_preds, final_preds, y_true
                )
                
                significance_results[label] = {
                    'comparison': f'best_individual vs {final_model_type}',
                    'is_significant': is_significant,
                    'p_value': p_value
                }
            else:
                print(f"  {label}: Using best individual model - no ensemble comparison needed")
        
        # Store results
        self.results['significance_tests'] = significance_results
        
        if significance_results:
            print("\nSummary of Significance Tests:")
            for label, result in significance_results.items():
                status = "significantly different" if result['is_significant'] else "not significantly different"
                print(f"  {label}: Models are {status} (p = {result['p_value']:.4f})")
    
    def print_performance_metrics(self):
        """Print individual dimension performance with model type."""
        if 'test' not in self.results:
            return
        
        print("\nDimension Performance Details:")
        for label, metrics in self.results['test'].items():
            print(f"  {label}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%) using {metrics['model_type']}")
    
    def test_model_significance(self, model1_preds, model2_preds, y_true):
        """Perform McNemar's test for statistical model comparison.
        
        Args:
            model1_preds: Predictions from first model
            model2_preds: Predictions from second model
            y_true: True labels
            
        Returns:
            Tuple[bool, float]: Significance result and p-value
        """
        print(f"\n  McNemar's Test: {model1_preds.shape[0]} samples")
        print(f"  Comparing predictions between two models...")        
        # Create contingency table
        correct1 = model1_preds == y_true
        correct2 = model2_preds == y_true
        
        a = sum(correct1 & correct2)  # Both correct
        b = sum(correct1 & ~correct2)  # Only model1 correct
        c = sum(~correct1 & correct2)  # Only model2 correct
        d = sum(~correct1 & ~correct2)  # Both wrong
        
        # McNemar's test
        result = mcnemar([[a, b], [c, d]], exact=False, correction=True)
        
        is_significant = result.pvalue < 0.05
        print(f"  McNemar's test: p-value = {result.pvalue:.4f}")
        print(f"  Result: Models are {'significantly' if is_significant else 'not significantly'} different")
        
        return is_significant, result.pvalue
    
    def create_visualizations(self):
        """Create comprehensive visualizations for analysis results."""
        print("\n" + "=" * 60)
        print("9. CREATING VISUALIZATIONS")
        print("=" * 60)
        
        try:
            # Try to import and run the thesis plot generator
            from generate_plots import ThesisPlotGenerator
            
            generator = ThesisPlotGenerator(str(Path(__file__).parent.parent))
            generator.generate_all_plots()
            print("✓ All thesis plots generated successfully!")
                
        except ImportError:
            logger.warning("generate_plots.py not found, creating simple plots...")
            self.create_simple_plots()
        except Exception as e:
            logger.error(f"Failed to generate plots: {str(e)}")
            print("❌ Visualization generation failed, creating simple plots...")
            self.create_simple_plots()
    
    def create_simple_plots(self):
        """Create simple visualization plots."""
        try:
            # Simple confusion matrix plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, label in enumerate(self.data['label_names']):
                if 'test_predictions' in self.results:
                    cm = confusion_matrix(
                        self.data['y_test'][label], 
                        self.results['test_predictions'][label]
                    )
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                    axes[i].set_title(f'{label}\nAccuracy: {self.results["test"][label]["accuracy"]:.3f}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
            
            plt.tight_layout()
            FIGURES_PATH.mkdir(parents=True, exist_ok=True)
            plt.savefig(FIGURES_PATH / 'confusion_matrices_simple.png', dpi=300)
            plt.close()
            
            print("✓ Simple plots created successfully")
        except Exception as e:
            logger.warning(f"Simple plot creation failed: {str(e)}")
            print("⚠️ Plot creation had issues")
    
    def save_results(self):
        """Save all Results and Models"""
        print("\n" + "=" * 60)
        print("10. SAVING RESULTS")
        print("=" * 60)
        
        # Save final Models
        try:
            MODELS_PATH.mkdir(parents=True, exist_ok=True)
            for label in self.data['label_names']:
                model_type, model_info = self.results['final_models'][label]
                model_path = MODELS_PATH / f'final_model_{label}.pkl'
                joblib.dump(model_info['model'], model_path)
                logger.info(f"Model saved for {label} at {model_path}")
                print(f"Model saved for {label}")
        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
            raise
        
        # Save Feature Pipeline
        joblib.dump(self.feature_pipeline, MODELS_PATH / 'feature_pipeline.pkl')
        
        # Save Results Summary
        results_summary = {
            'test_results': self.results['test'],
            'average_accuracy': self.results['avg_accuracy'],
            'average_f1': self.results['avg_f1'],
            'optimal_thresholds': self.results.get('optimal_thresholds', {}),
            'feature_count': self.data['X_train_transformed'].shape[1],
            'selected_features': self.feature_pipeline.engineer.selected_features
        }
        
        METRICS_PATH.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH / 'final_results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        print("\nAll Models and Results are saved!")
    
    def run_complete_analysis(self):
        """Run Complete Analysis"""
        print("\n" + "=" * 80)
        print("LEARNING STYLE CLASSIFICATION - 3 DIMENSIONS")
        print("=" * 80)
        
        # 1. Data loading and analysis
        self.load_data()
        
        # 2. Feature Engineering
        self.setup_feature_engineering(n_features=40)
        
        # 3. Baseline-Models
        self.train_baseline_models()
        
        # 4. Advanced Models
        self.train_advanced_models()
        
        # 5. Ensemble-Models
        self.create_ensemble_models()
        
        # 6. Threshold-Optimization
        self.optimize_thresholds()
        
        # 7. Final Evaluation
        test_results, avg_accuracy = self.evaluate_final_models()
        
        # 8. Statistical Significance Testing
        self.perform_significance_tests()
        
        # 9. Create Visualizations
        self.create_visualizations()
        
        # 10. Save Results
        self.save_results()
        
        return avg_accuracy


def main():
    """Main function to run the complete learning style analysis."""
    logger.info("Starting Learning Style Classification Analysis")
    
    # Setup data paths
    data_paths = {
        'csms': str(config.get_data_path('csms')),
        'cshs': str(config.get_data_path('cshs'))
    }
    
    try:
        # Create and run analysis
        analysis = LearningStyleAnalysis(data_paths, RANDOM_STATE)
        final_accuracy = analysis.run_complete_analysis()
        
        logger.info(f"Analysis completed successfully! Final Accuracy: {final_accuracy:.2%}")
        print(f"\n\nAnalysis completed! Final Accuracy: {final_accuracy:.2%}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()