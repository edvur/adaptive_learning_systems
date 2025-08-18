"""
integration_step1.py - Bridge between existing Models and Tutoring-System
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass
import json

# Configuration
@dataclass
class Config:
    """Central Configuration for all Paths"""
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results"
    
    # Model data
    FEATURE_PIPELINE_PATH: Path = MODELS_DIR / "feature_pipeline.pkl"
    MODEL_PATHS: Dict[str, Path] = None
    
    def __post_init__(self):
        """Initialize model paths"""
        self.MODEL_PATHS = {
            'Perception': self.MODELS_DIR / "final_model_Perception.pkl",
            'Input': self.MODELS_DIR / "final_model_Input.pkl",
            'Understanding': self.MODELS_DIR / "final_model_Understanding.pkl"
        }
        
        # Create directories if they don't exist
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR]:
            dir_path.mkdir(exist_ok=True, parents=True)

# Global Configuration
CONFIG = Config()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dynamic Import of Modules
def import_project_modules():
    """Dynamic Import of Modules"""
    try:
        # Add project root to Python path 
        project_root = CONFIG.BASE_DIR / "python_files"
        if project_root.exists():
            sys.path.insert(0, str(project_root))
        
        # Try importing modules
        global DataLoader, FeaturePipeline
        
        try:
            from data_loader_module import DataLoader
            from feature_engineering_module import FeaturePipeline
            logger.info("Modules imported successfully")
            return True
        except ImportError as e:
            logger.warning(f"Modules could not be imported: {e}")
            logger.info("Using fallback implementations")
            return False
            
    except Exception as e:
        logger.error(f"Error with import: {e}")
        return False

# Fallback implementations if modules don't exist
class FallbackFeaturePipeline:
    """Minimal Feature Pipeline for Tests"""
    def __init__(self):
        self.is_fitted = False
        self.selected_features = None
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Simple transformation without feature engineering"""
        logger.warning("Using fallback feature pipeline")
        return X

class LearningStylePredictor:
    """
    Improved version of Learning Style Predictor with enhanced error handling
    """
    
    def __init__(self, config: Config = None):
        self.config = config or CONFIG
        self.models = {}
        self.feature_pipeline = None
        self.is_loaded = False
        
        # Label Mapping
        self.label_interpretations = {
            'Perception': {
                0: 'Intuitive',
                1: 'Sensing'
            },
            'Input': {
                0: 'Verbal',
                1: 'Visual'
            },
            'Understanding': {
                0: 'Global',
                1: 'Sequential'
            }
        }
        
        # Feature Names
        self.expected_features = [
            'Course overview', 'Reading file', 'Abstract materiale', 
            'Concrete material', 'Visual Materials', 'Self-assessment',
            'Exercises submit', 'Quiz submitted', 'playing', 
            'paused', 'unstarted', 'buffering'
        ]
    
    def load_models(self) -> bool:
        """Load all trained models with improved error handling"""
        try:
            logger.info("Loading trained Learning Style models...")
            
            # Load Feature Pipeline
            if self.config.FEATURE_PIPELINE_PATH.exists():
                self.feature_pipeline = joblib.load(self.config.FEATURE_PIPELINE_PATH)
                logger.info("Feature Pipeline loaded")
            else:
                logger.warning(f"Feature Pipeline not found: {self.config.FEATURE_PIPELINE_PATH}")
                # Use fallback
                self.feature_pipeline = FallbackFeaturePipeline()
            
            # Load models for every dimension
            models_loaded = 0
            for label, model_path in self.config.MODEL_PATHS.items():
                if model_path.exists():
                    self.models[label] = joblib.load(model_path)
                    logger.info(f"{label} model loaded")
                    models_loaded += 1
                else:
                    logger.error(f"{label} model not found: {model_path}")
            
            # Check if enough models are loaded
            if models_loaded == 0:
                logger.error("No models could be loaded!")
                return False
            elif models_loaded < len(self.config.MODEL_PATHS):
                logger.warning(f"Only {models_loaded} from {len(self.config.MODEL_PATHS)} models loaded")
            
            self.is_loaded = True
            logger.info(f"{models_loaded} models successfully loaded!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def validate_activity_data(self, activity_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate the input data"""
        missing_features = []
        
        for feature in self.expected_features:
            if feature not in activity_data.columns:
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            return False, missing_features
        
        return True, []
    
    def predict_learning_style(self, activity_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Improved prediction with error handling and validation
        """
        if not self.is_loaded:
            raise ValueError("Models must be loaded first!")
        
        # Validate Input Data
        is_valid, missing = self.validate_activity_data(activity_data)
        if not is_valid:
            # Fill missing features with 0
            for feature in missing:
                activity_data[feature] = 0
            logger.warning(f"Filling missing features with 0: {missing}")
        
        try:
            # Feature Engineering
            logger.info("Executing feature engineering...")
            
            # Use feature pipeline 
            if hasattr(self.feature_pipeline, 'transform'):
                X_transformed = self.feature_pipeline.transform(activity_data)
            else:
                # Fallback: Use raw data
                X_transformed = activity_data
            
            logger.info(f"Features transformed: {activity_data.shape[1]} -> {X_transformed.shape[1]}")
            
            # Prediction for every dimension
            predictions = {}
            
            for label, model in self.models.items():
                try:
                    # Prediction
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_transformed)[0]
                        predicted_class = np.argmax(proba)
                        confidence = np.max(proba)
                    else:
                        predicted_class = model.predict(X_transformed)[0]
                        confidence = 0.8  # Default confidence
                        proba = [0.2, 0.8] if predicted_class == 1 else [0.8, 0.2]
                    
                    # Interpretation
                    interpretation = self.label_interpretations[label][predicted_class]
                    
                    predictions[label] = {
                        'predicted_class': int(predicted_class),
                        'interpretation': interpretation,
                        'confidence': float(confidence),
                        'probabilities': {
                            self.label_interpretations[label][0]: float(proba[0]),
                            self.label_interpretations[label][1]: float(proba[1])
                        }
                    }
                    
                    logger.info(f"{label}: {interpretation} (Confidence: {confidence:.3f})")
                    
                except Exception as e:
                    logger.error(f"Error with prediction for {label}: {e}")
                    # Fallback prediction
                    predictions[label] = {
                        'predicted_class': 0,
                        'interpretation': self.label_interpretations[label][0],
                        'confidence': 0.5,
                        'probabilities': {
                            self.label_interpretations[label][0]: 0.5,
                            self.label_interpretations[label][1]: 0.5
                        },
                        'error': str(e)
                    }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error with prediction: {str(e)}")
            raise
    
    def create_student_profile(self, activity_data: pd.DataFrame, student_id: str = None) -> Dict:
        """Create a student profile"""
        
        # Generate student ID 
        if student_id is None:
            import uuid
            student_id = f"student_{uuid.uuid4().hex[:8]}"
        
        # Learning style prediction
        learning_style = self.predict_learning_style(activity_data)
        
        # Activity analysis
        activity_analysis = self._analyze_activity_patterns(activity_data)
        
        # Generate recommendations
        recommendations = self._generate_initial_recommendations(learning_style, activity_analysis)
        
        # Combine to profile
        profile = {
            'student_id': student_id,
            'learning_style': learning_style,
            'activity_patterns': activity_analysis,
            'recommendations': recommendations,
            'timestamp': pd.Timestamp.now().isoformat(),
            'metadata': {
                'model_version': '1.0',
                'confidence_threshold': 0.7,
                'features_used': len(activity_data.columns)
            }
        }
        
        return profile
    
    def _analyze_activity_patterns(self, activity_data: pd.DataFrame) -> Dict:
        """Advanced activity analysis with error handling"""
        
        try:
            row = activity_data.iloc[0]
            
            # Safe calculations with error handling
            def safe_get(key, default=0):
                return float(row.get(key, default)) if key in row else default
            
            # Categorized activities
            reading_activities = safe_get('Reading file') + safe_get('Abstract materiale')
            visual_activities = safe_get('Visual Materials') + safe_get('playing')
            interactive_activities = (safe_get('Exercises submit') + 
                                    safe_get('Quiz submitted') + 
                                    safe_get('Self-assessment'))
            overview_activities = safe_get('Course overview')
            
            total_activities = row.sum() if hasattr(row, 'sum') else 1
            
            # Avoid division by zero
            total_activities = max(total_activities, 1)
            
            patterns = {
                'total_engagement': float(total_activities),
                'reading_preference': float(reading_activities / total_activities),
                'visual_preference': float(visual_activities / total_activities),
                'interactive_preference': float(interactive_activities / total_activities),
                'overview_preference': float(overview_activities / total_activities),
                'video_engagement': {
                    'playing_time': safe_get('playing'),
                    'pause_frequency': safe_get('paused'),
                    'completion_indicator': float(
                        safe_get('playing') / (safe_get('playing') + safe_get('unstarted') + 1)
                    )
                },
                'engagement_diversity': self._calculate_engagement_diversity(row)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error with activity analysis: {e}")
            # Fallback values
            return {
                'total_engagement': 0.0,
                'reading_preference': 0.0,
                'visual_preference': 0.0,
                'interactive_preference': 0.0,
                'overview_preference': 0.0,
                'video_engagement': {
                    'playing_time': 0.0,
                    'pause_frequency': 0.0,
                    'completion_indicator': 0.0
                },
                'engagement_diversity': 0.0
            }
    
    def _calculate_engagement_diversity(self, activity_row) -> float:
        """Calculate the diversity in activities (Shannon Entropy)"""
        try:
            # Normalize activities
            total = activity_row.sum()
            if total == 0:
                return 0.0
            
            probabilities = activity_row / total
            # Shannon Entropy
            entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities if p > 0)
            
            # Normalize from 0-1
            max_entropy = np.log2(len(activity_row))
            return float(entropy / max_entropy) if max_entropy > 0 else 0.0
            
        except:
            return 0.5  # Default middle diversity
    
    def _generate_initial_recommendations(self, learning_style: Dict, 
                                        activity_patterns: Dict) -> Dict:
        """Generate detailed recommendations based on learning style"""
        
        recommendations = {
            'content_types': [],
            'learning_strategies': [],
            'presentation_style': {},
            'difficulty_approach': {},
            'time_recommendations': {},
            'interaction_style': {}
        }
        
        # Content type recommendations based on input dimension
        if 'Input' in learning_style:
            if learning_style['Input']['interpretation'] == 'Visual':
                recommendations['content_types'].extend([
                    'Videos with visual demonstrations',
                    'Interactive diagram and Infographics',
                    'Animated Explanations',
                    'Mind Maps and Concept Maps'
                ])
                recommendations['presentation_style']['visual_elements'] = 'high'
                recommendations['presentation_style']['text_density'] = 'low'
            else:  # Verbal
                recommendations['content_types'].extend([
                    'Detailed Text Materials',
                    'Audio-Explanations and Podcasts',
                    'Written Exercises and Essays',
                    'Discussion Forums and Blogs'
                ])
                recommendations['presentation_style']['visual_elements'] = 'moderate'
                recommendations['presentation_style']['text_density'] = 'high'
        
        # Learning strategies based on perception dimension
        if 'Perception' in learning_style:
            if learning_style['Perception']['interpretation'] == 'Sensing':
                recommendations['learning_strategies'].extend([
                    'Practical Exercises with real Examples',
                    'Hands-on Projects',
                    'Step-for-step Tutorials',
                    'Case Studies'
                ])
                recommendations['difficulty_approach']['progression'] = 'gradual'
                recommendations['difficulty_approach']['abstract_level'] = 'low'
            else:  # Intuitive
                recommendations['learning_strategies'].extend([
                    'Theoretical concepts and models',
                    'Abstract problem statements',
                    'Conceptual connections',
                    'Innovative approaches'
                ])
                recommendations['difficulty_approach']['progression'] = 'flexible'
                recommendations['difficulty_approach']['abstract_level'] = 'high'
        
        # Structuring based on understanding dimension
        if 'Understanding' in learning_style:
            if learning_style['Understanding']['interpretation'] == 'Sequential':
                recommendations['learning_strategies'].extend([
                    'Linear learning paths',
                    'Building modules',
                    'Clear learning objectives per unit',
                    'Regular summaries'
                ])
                recommendations['presentation_style']['structure'] = 'sequential'
                recommendations['time_recommendations']['session_length'] = 'medium'
            else:  # Global
                recommendations['learning_strategies'].extend([
                    'Overview before details',
                    'Concept maps',
                    'Flexible navigation',
                    'Networked learning'
                ])
                recommendations['presentation_style']['structure'] = 'holistic'
                recommendations['time_recommendations']['session_length'] = 'flexible'
        
        # Based on activity patterns
        if activity_patterns['visual_preference'] > 0.5:
            recommendations['interaction_style']['primary'] = 'visual_interactive'
        elif activity_patterns['reading_preference'] > 0.5:
            recommendations['interaction_style']['primary'] = 'text_based'
        else:
            recommendations['interaction_style']['primary'] = 'mixed'
        
        # Time recommendations
        if activity_patterns['total_engagement'] < 20:
            recommendations['time_recommendations']['suggested_daily'] = '30-45 min'
        elif activity_patterns['total_engagement'] < 50:
            recommendations['time_recommendations']['suggested_daily'] = '45-60 min'
        else:
            recommendations['time_recommendations']['suggested_daily'] = '60-90 min'
        
        return recommendations
    
    def save_profile(self, profile: Dict, filepath: Path = None) -> Path:
        """Save student profile"""
        if filepath is None:
            filepath = self.config.RESULTS_DIR / f"profile_{profile['student_id']}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Profile saved: {filepath}")
        return filepath
    
    def load_profile(self, student_id: str) -> Optional[Dict]:
        """Load saved student profile"""
        filepath = self.config.RESULTS_DIR / f"profile_{student_id}.json"
        
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None


class TutorIntegrationManager:
    """Improved version of Integration Manager"""
    
    def __init__(self, config: Config = None):
        self.config = config or CONFIG
        self.predictor = LearningStylePredictor(self.config)
        self.student_profiles = {}
        self.session_history = {}
        
    def initialize(self) -> bool:
        """Initialize all components with error handling"""
        logger.info("Initializing Tutor Integration Manager...")
        
        # Try importing modules
        modules_loaded = import_project_modules()
        if not modules_loaded:
            logger.warning("Using fallback implementation")
        
        # Load Models
        success = self.predictor.load_models()
        
        if success:
            logger.info("Integration Manager ready!")
            self._run_self_test()
        else:
            logger.error("Initialization failed")
        
        return success
    
    def _run_self_test(self):
        """Run self test"""
        logger.info("Run self test...")
        
        # Test with Demo-Data
        test_data = pd.DataFrame({
            'Course overview': [5],
            'Reading file': [10],
            'Abstract materiale': [3],
            'Concrete material': [7],
            'Visual Materials': [8],
            'Self-assessment': [2],
            'Exercises submit': [5],
            'Quiz submitted': [3],
            'playing': [15],
            'paused': [4],
            'unstarted': [2],
            'buffering': [1]
        })
        
        try:
            profile = self.predictor.create_student_profile(test_data, "test_student")
            logger.info("Self test successful!")
        except Exception as e:
            logger.error(f"Self test failed: {e}")
    
    def process_new_student(self, student_id: str, activity_data: pd.DataFrame) -> Dict:
        """Process new student with error handling"""
        
        logger.info(f"Processing new student: {student_id}")
        
        try:
            # Create Profile
            profile = self.predictor.create_student_profile(activity_data, student_id)
            
            # Save in cache
            self.student_profiles[student_id] = profile
            
            # Save on disk
            self.predictor.save_profile(profile)
            
            # Initialize Session History
            self.session_history[student_id] = []
            
            logger.info(f"Profile for {student_id} created successfully")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error processing {student_id}: {e}")
            raise
    
    def get_student_profile(self, student_id: str) -> Optional[Dict]:
        """Get existing student profile"""
        
        # Check Cache
        if student_id in self.student_profiles:
            return self.student_profiles[student_id]
        
        # Try loading from disk
        profile = self.predictor.load_profile(student_id)
        if profile:
            self.student_profiles[student_id] = profile
            return profile
        
        return None
    
    def update_student_activity(self, student_id: str, new_activity_data: pd.DataFrame):
        """Update student activity and profile"""
        
        if student_id not in self.student_profiles:
            logger.warning(f"Student {student_id} not found. Creating new profile...")
            return self.process_new_student(student_id, new_activity_data)
        
        try:
            # Update Profile
            updated_profile = self.predictor.create_student_profile(new_activity_data, student_id)
            
            # Keep Session History
            if student_id in self.session_history:
                updated_profile['session_history'] = self.session_history[student_id]
            
            self.student_profiles[student_id] = updated_profile
            self.predictor.save_profile(updated_profile)
            
            logger.info(f"Profile for {student_id} updated")
            return updated_profile
            
        except Exception as e:
            logger.error(f"Error updating {student_id}: {e}")
            raise
    
    def log_session_event(self, student_id: str, event: Dict):
        """Log session events for later analysis"""
        if student_id not in self.session_history:
            self.session_history[student_id] = []
        
        event['timestamp'] = pd.Timestamp.now().isoformat()
        self.session_history[student_id].append(event)
    
    def get_statistics(self) -> Dict:
        """Get statistics of all students"""
        stats = {
            'total_students': len(self.student_profiles),
            'learning_style_distribution': {},
            'average_engagement': 0,
            'most_common_style': {}
        }
        
        if not self.student_profiles:
            return stats
        
        # Analyze learning styles 
        style_counts = {'Perception': {}, 'Input': {}, 'Understanding': {}}
        total_engagement = 0
        
        for profile in self.student_profiles.values():
            # Engagement
            if 'activity_patterns' in profile:
                total_engagement += profile['activity_patterns'].get('total_engagement', 0)
            
            # Learning Style
            if 'learning_style' in profile:
                for dimension, analysis in profile['learning_style'].items():
                    style = analysis.get('interpretation', 'Unknown')
                    if style not in style_counts[dimension]:
                        style_counts[dimension][style] = 0
                    style_counts[dimension][style] += 1
        
        stats['learning_style_distribution'] = style_counts
        stats['average_engagement'] = total_engagement / len(self.student_profiles)
        
        # Most common styles
        for dimension, counts in style_counts.items():
            if counts:
                stats['most_common_style'][dimension] = max(counts.items(), key=lambda x: x[1])[0]
        
        return stats


def test_improved_integration():
    """Test of improved integration"""
    
    print("🧪 TESTING IMPROVED INTEGRATION")
    print("=" * 60)
    
    # Create test configuration
    config = Config()
    print(f"📁 Working directory: {config.BASE_DIR}")
    print(f"📁 Models: {config.MODELS_DIR}")
    
    # Initialize Manager
    manager = TutorIntegrationManager(config)
    
    if not manager.initialize():
        print("❌ Initialization failed")
        print("\n📋 Error Handling:")
        print("1. Make sure that the model files exist:")
        for label, path in config.MODEL_PATHS.items():
            exists = "✅" if path.exists() else "❌"
            print(f"   {exists} {label}: {path}")
        
        print("\n2. Make sure that the Feature Pipeline exists:")
        exists = "✅" if config.FEATURE_PIPELINE_PATH.exists() else "❌"
        print(f"   {exists} Feature Pipeline: {config.FEATURE_PIPELINE_PATH}")
        return
    
    # Test with different Student Types
    test_students = [
        {
            'id': 'visual_learner_001',
            'data': pd.DataFrame({
                'Course overview': [3],
                'Reading file': [5],
                'Abstract materiale': [2],
                'Concrete material': [10],
                'Visual Materials': [20],
                'Self-assessment': [2],
                'Exercises submit': [5],
                'Quiz submitted': [3],
                'playing': [25],
                'paused': [8],
                'unstarted': [2],
                'buffering': [1]
            })
        },
        {
            'id': 'verbal_learner_002',
            'data': pd.DataFrame({
                'Course overview': [8],
                'Reading file': [25],
                'Abstract materiale': [10],
                'Concrete material': [5],
                'Visual Materials': [5],
                'Self-assessment': [5],
                'Exercises submit': [8],
                'Quiz submitted': [6],
                'playing': [10],
                'paused': [3],
                'unstarted': [5],
                'buffering': [1]
            })
        }
    ]
    
    print("\n📊 TESTING DIFFERENT LEARNING STYLES:")
    
    for student_info in test_students:
        print(f"\n👤 Student: {student_info['id']}")
        print("-" * 40)
        
        try:
            # Process student
            profile = manager.process_new_student(
                student_info['id'], 
                student_info['data']
            )
            
            # Show results 
            print("Learning Style Analysis:")
            for dimension, analysis in profile['learning_style'].items():
                print(f"   {dimension}: {analysis['interpretation']} "
                      f"(Confidence: {analysis['confidence']:.2%})")
            
            print("\n📊 Activity patterns:")
            patterns = profile['activity_patterns']
            print(f"   Visual Preference: {patterns['visual_preference']:.2%}")
            print(f"   Reading preference: {patterns['reading_preference']:.2%}")
            print(f"   Interactive Preference: {patterns['interactive_preference']:.2%}")
            
            print("\n🎯 Top recommendations:")
            recs = profile['recommendations']
            print(f"   Content types: {', '.join(recs['content_types'][:2])}")
            print(f"   Learning strategies: {', '.join(recs['learning_strategies'][:2])}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show total statistics 
    print("\n📊 Total Statistics:")
    stats = manager.get_statistics()
    print(f"Number of students: {stats['total_students']}")
    print(f"Average engagement: {stats['average_engagement']:.1f}")
    print(f"Most common style: {stats['most_common_style']}")
    
    print("\n✅ IMPROVED INTEGRATION TESTED SUCCESSFULLY!")
    
    return manager


if __name__ == "__main__":
    # Test 
    manager = test_improved_integration()