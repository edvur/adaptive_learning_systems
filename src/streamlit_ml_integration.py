# ml_integration.py - Integration ML-Modelle in die Streamlit App
"""
ML Model Integration Module
Bridge trained models with the streamlit interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitModelLoader:
    """Load and cache ML-Models for Streamlit"""
    
    @staticmethod
    @st.cache_resource
    def load_learning_style_models(model_dir: str = "models") -> Dict:
        """Load learning style models"""
        models = {}
        model_path = Path(model_dir)
        
        try:
            # Feature Pipeline
            pipeline_path = model_path / "feature_pipeline.pkl"
            if pipeline_path.exists():
                models['feature_pipeline'] = joblib.load(pipeline_path)
                logger.info("Feature pipeline loaded")
            
            # Classification models for each dimension
            for dimension in ['Perception', 'Input', 'Understanding']:
                model_file = model_path / f"final_model_{dimension}.pkl"
                if model_file.exists():
                    models[dimension] = joblib.load(model_file)
                    logger.info(f"{dimension} model loaded")
            
            return models
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return {}
    
    @staticmethod
    @st.cache_resource
    def load_dqn_agent(model_path: str = "models/best_dqn_model.pth") -> Optional[torch.nn.Module]:
        """Load trained DQN-Model"""
        try:
            # Define the DQN-Architecture
            class DuelingDQN(nn.Module):
                def __init__(self, state_dim: int, action_dim: int):
                    super().__init__()
                    self.shared = nn.Sequential(
                        nn.Linear(state_dim, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                    
                    self.value_stream = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1)
                    )
                    
                    self.advantage_stream = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_dim)
                    )
                
                def forward(self, x):
                    features = self.shared(x)
                    value = self.value_stream(features)
                    advantage = self.advantage_stream(features)
                    q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
                    return q_values
            
            # Load model 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DuelingDQN(state_dim=30, action_dim=9)
            
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=device)
                if 'q_network_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['q_network_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                logger.info("DQN model loaded successfully")
                return model
            else:
                logger.warning(f"DQN model not found at {model_path}")
                return None
                
        except Exception as e:
            st.error(f"Error loading DQN model: {str(e)}")
            return None


class StreamlitLearningStylePredictor:
    """Wrapper for learning style prediction in streamlit"""
    
    def __init__(self, models: Dict):
        self.models = models
        self.feature_pipeline = models.get('feature_pipeline')
        self.dimension_models = {
            'Perception': models.get('Perception'),
            'Input': models.get('Input'),
            'Understanding': models.get('Understanding')
        }
    
    def predict(self, activities: pd.DataFrame) -> Dict:
        """Predict for learning styles"""
        
        if not all(self.dimension_models.values()):
            st.error("Not all models are loaded!")
            return {}
        
        try:
            # Feature Engineering
            if self.feature_pipeline:
                X_transformed = self.feature_pipeline.transform(activities)
            else:
                X_transformed = activities
            
            predictions = {}
            
            for dimension, model in self.dimension_models.items():
                if model is None:
                    continue
                
                # Prediction
                pred_class = model.predict(X_transformed)[0]
                pred_proba = model.predict_proba(X_transformed)[0]
                
                # Interpretation
                if dimension == 'Perception':
                    interpretations = {0: 'Intuitive', 1: 'Sensing'}
                elif dimension == 'Input':
                    interpretations = {0: 'Verbal', 1: 'Visual'}
                else:  # Understanding
                    interpretations = {0: 'Global', 1: 'Sequential'}
                
                predictions[dimension] = {
                    'predicted_class': int(pred_class),
                    'interpretation': interpretations[pred_class],
                    'confidence': float(np.max(pred_proba)),
                    'probabilities': {
                        interpretations[0]: float(pred_proba[0]),
                        interpretations[1]: float(pred_proba[1])
                    }
                }
            
            return predictions
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return {}


class StreamlitDQNAgent:
    """Wrapper for DQN-Agent in Streamlit"""
    
    def __init__(self, model: Optional[torch.nn.Module], content_library: List[Dict]):
        self.model = model
        self.content_library = content_library
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def select_content(self, student_state: Dict, learning_style: Dict) -> Dict:
        """Select next content based on DQN"""
        
        if self.model is None:
            # Fallback: random choice 
            return np.random.choice(self.content_library)
        
        try:
            # Create State Vector (30 Features)
            state_vector = self._create_state_vector(student_state, learning_style)
            
            # DQN prediction
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            # Map Q-values zu Content-Auswahl
            content_scores = self._calculate_content_scores(q_values, learning_style)
            
            # Select best Content
            best_content_idx = np.argmax(content_scores)
            selected_content = self.content_library[best_content_idx % len(self.content_library)]
            
            return selected_content
            
        except Exception as e:
            logger.error(f"Content selection error: {str(e)}")
            return np.random.choice(self.content_library)
    
    def _create_state_vector(self, student_state: Dict, learning_style: Dict) -> np.ndarray:
        """Create 30-dimensional State Vector"""
        
        features = []
        
        # Learning Style Features (6)
        for dim in ['Perception', 'Input', 'Understanding']:
            if dim in learning_style:
                probs = learning_style[dim]['probabilities']
                features.extend(list(probs.values()))
            else:
                features.extend([0.5, 0.5])
        
        # Knowledge Features (4)
        knowledge_levels = student_state.get('knowledge_levels', {})
        if knowledge_levels:
            features.append(np.mean(list(knowledge_levels.values())))
            features.append(np.var(list(knowledge_levels.values())))
            features.append(np.min(list(knowledge_levels.values())))
            features.append(np.max(list(knowledge_levels.values())))
        else:
            features.extend([0.3, 0.1, 0.1, 0.5])
        
        # Performance Features (3)
        perf_history = student_state.get('performance_history', [])
        if len(perf_history) >= 3:
            features.append(np.mean(perf_history))
            features.append(np.var(perf_history))
            features.append(np.mean(np.diff(perf_history)) if len(perf_history) > 1 else 0)
        else:
            features.extend([0.5, 0.1, 0.0])
        
        # Current State Features (6)
        features.append(student_state.get('engagement_level', 0.7))
        features.append(student_state.get('fatigue_level', 0.0))
        features.append(student_state.get('motivation_level', 0.7))
        features.append(student_state.get('frustration_level', 0.0))
        features.append(student_state.get('satisfaction_score', 0.5))
        features.append(student_state.get('attention_span', 0.7))
        
        # Session Features (2)
        features.append(min(student_state.get('session_time', 0) / 60.0, 1.0))
        features.append(min(student_state.get('session_time', 0) / 45.0, 1.0))
        
        # Activity Pattern Features (3)
        features.append(student_state.get('reading_preference', 0.5))
        features.append(student_state.get('visual_preference', 0.5))
        features.append(student_state.get('interactive_preference', 0.5))
        
        # Meta-Learning Features (4)
        features.append(student_state.get('learning_velocity', 0.5))
        features.append(student_state.get('retention_rate', 0.8))
        features.append(student_state.get('difficulty_preference', 0.5))
        features.append(student_state.get('error_rate', 0.2))
        
        # Additional Features (2)
        features.append(min(len(student_state.get('completed_content', [])) / 20.0, 1.0))
        features.append(min(student_state.get('help_requests', 0) / 10.0, 1.0))
        
        # Ensure exactly 30 features
        features = features[:30]
        while len(features) < 30:
            features.append(0.5)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_content_scores(self, q_values: np.ndarray, learning_style: Dict) -> np.ndarray:
        """Calculate scores for every content based on Q-values and learning style"""
        
        scores = []
        
        for content in self.content_library:
            # Basic score of Q-values
            base_score = q_values[0] if len(q_values) > 0 else 0.5
            
            # Learning Style Fit
            style_fit = 1.0
            
            # Input dimension
            if learning_style['Input']['interpretation'] == 'Visual':
                if content['type'] in ['video', 'interactive']:
                    style_fit *= 1.2
                elif content['type'] == 'reading':
                    style_fit *= 0.8
            else:  # Verbal
                if content['type'] == 'reading':
                    style_fit *= 1.2
                elif content['type'] == 'video':
                    style_fit *= 0.8
            
            # Perception dimension
            if learning_style['Perception']['interpretation'] == 'Sensing':
                if content.get('practical', False):
                    style_fit *= 1.1
            else:  # Intuitive
                if content.get('theoretical', False):
                    style_fit *= 1.1
            
            final_score = base_score * style_fit
            scores.append(final_score)
        
        return np.array(scores)


# Content Library für Demo
DEMO_CONTENT_LIBRARY = [
    {
        'id': 'js_intro_video',
        'title': 'JavaScript Basics - Video Introduction',
        'type': 'video',
        'topic': 'JavaScript Basics',
        'difficulty': 'Easy',
        'estimated_time': 15,
        'practical': True,
        'theoretical': False,
        'description': 'Visual introduction to JavaScript fundamentals with live coding examples.'
    },
    {
        'id': 'js_variables_reading',
        'title': 'Variables and Data Types - Comprehensive Guide',
        'type': 'reading',
        'topic': 'JavaScript Basics',
        'difficulty': 'Easy',
        'estimated_time': 20,
        'practical': False,
        'theoretical': True,
        'description': 'Detailed text explanation of JavaScript variables, scoping, and data types.'
    },
    {
        'id': 'react_components_interactive',
        'title': 'Building React Components - Interactive Tutorial',
        'type': 'interactive',
        'topic': 'React Components',
        'difficulty': 'Medium',
        'estimated_time': 30,
        'practical': True,
        'theoretical': False,
        'description': 'Hands-on coding exercise to create your first React components.'
    },
    {
        'id': 'react_state_quiz',
        'title': 'React State Management - Knowledge Check',
        'type': 'quiz',
        'topic': 'React Components',
        'difficulty': 'Medium',
        'estimated_time': 10,
        'practical': False,
        'theoretical': True,
        'description': 'Test your understanding of React state and props concepts.'
    },
    {
        'id': 'api_basics_video',
        'title': 'REST API Fundamentals - Visual Guide',
        'type': 'video',
        'topic': 'API Integration',
        'difficulty': 'Medium',
        'estimated_time': 25,
        'practical': False,
        'theoretical': True,
        'description': 'Animated explanation of REST API concepts and HTTP methods.'
    },
    {
        'id': 'api_fetch_exercise',
        'title': 'Fetching Data with JavaScript - Practice',
        'type': 'interactive',
        'topic': 'API Integration',
        'difficulty': 'Hard',
        'estimated_time': 35,
        'practical': True,
        'theoretical': False,
        'description': 'Build a real application that fetches and displays API data.'
    }
]


# helpfunctions for Streamlit
def initialize_ml_models():
    """Initialize all ml models for the Streamlit App"""
    
    with st.spinner("Loading machine learning models..."):
        # Load Learning Style Models
        models = StreamlitModelLoader.load_learning_style_models()
        
        if models:
            predictor = StreamlitLearningStylePredictor(models)
        else:
            predictor = None
            st.warning("Learning style models could not be loaded. Using demo mode.")
        
        # Load DQN-Agent
        dqn_model = StreamlitModelLoader.load_dqn_agent()
        dqn_agent = StreamlitDQNAgent(dqn_model, DEMO_CONTENT_LIBRARY)
        
        return predictor, dqn_agent


def create_demo_predictor():
    """Create Demo-Predictor for Tests"""
    
    class DemoPredictor:
        def predict(self, activities):
            # Simulate realistic predictions based on activities
            row = activities.iloc[0]
            
            # Simple heuristics für Demo
            visual_score = (row.get('Visual Materials', 0) + row.get('playing', 0)) / 2
            verbal_score = (row.get('Reading file', 0) + row.get('Abstract materiale', 0)) / 2
            
            sensing_score = row.get('Concrete material', 0)
            intuitive_score = row.get('Abstract materiale', 0)
            
            sequential_score = row.get('Course overview', 0)
            global_score = 20 - sequential_score
            
            return {
                'Perception': {
                    'predicted_class': 1 if sensing_score > intuitive_score else 0,
                    'interpretation': 'Sensing' if sensing_score > intuitive_score else 'Intuitive',
                    'confidence': 0.7 + np.random.random() * 0.25,
                    'probabilities': {
                        'Sensing': sensing_score / (sensing_score + intuitive_score + 1),
                        'Intuitive': intuitive_score / (sensing_score + intuitive_score + 1)
                    }
                },
                'Input': {
                    'predicted_class': 1 if visual_score > verbal_score else 0,
                    'interpretation': 'Visual' if visual_score > verbal_score else 'Verbal',
                    'confidence': 0.65 + np.random.random() * 0.3,
                    'probabilities': {
                        'Visual': visual_score / (visual_score + verbal_score + 1),
                        'Verbal': verbal_score / (visual_score + verbal_score + 1)
                    }
                },
                'Understanding': {
                    'predicted_class': 1 if sequential_score > global_score else 0,
                    'interpretation': 'Sequential' if sequential_score > global_score else 'Global',
                    'confidence': 0.6 + np.random.random() * 0.35,
                    'probabilities': {
                        'Sequential': sequential_score / (sequential_score + global_score + 1),
                        'Global': global_score / (sequential_score + global_score + 1)
                    }
                }
            }
    
    return DemoPredictor()


# Session State Management
def get_or_create_student_state():
    """Get or create Student State"""
    
    if 'student_state' not in st.session_state:
        st.session_state.student_state = {
            'knowledge_levels': {
                'JavaScript Basics': 0.3,
                'React Components': 0.1,
                'API Integration': 0.0,
                'Testing': 0.0
            },
            'performance_history': [0.5, 0.6, 0.55],
            'engagement_level': 0.7,
            'fatigue_level': 0.0,
            'motivation_level': 0.7,
            'frustration_level': 0.0,
            'satisfaction_score': 0.5,
            'attention_span': 0.7,
            'session_time': 0,
            'reading_preference': 0.5,
            'visual_preference': 0.5,
            'interactive_preference': 0.5,
            'learning_velocity': 0.5,
            'retention_rate': 0.8,
            'difficulty_preference': 0.5,
            'error_rate': 0.2,
            'completed_content': [],
            'help_requests': 0
        }
    
    return st.session_state.student_state


def update_student_state(performance: float, content_type: str, engagement: float):
    """Update the Student State after an Interaktion"""
    
    state = get_or_create_student_state()
    
    # Update performance history
    state['performance_history'].append(performance)
    if len(state['performance_history']) > 10:
        state['performance_history'].pop(0)
    
    # Update engagement
    state['engagement_level'] = 0.8 * state['engagement_level'] + 0.2 * engagement
    
    # Update fatigue
    state['fatigue_level'] = min(1.0, state['fatigue_level'] + 0.05)
    
    # Update session time
    state['session_time'] += 5
    
    # Update preferences based on performance
    if content_type == 'video' and performance > 0.7:
        state['visual_preference'] = min(1.0, state['visual_preference'] + 0.05)
    elif content_type == 'reading' and performance > 0.7:
        state['reading_preference'] = min(1.0, state['reading_preference'] + 0.05)
    elif content_type == 'interactive' and performance > 0.7:
        state['interactive_preference'] = min(1.0, state['interactive_preference'] + 0.05)
    
    return state