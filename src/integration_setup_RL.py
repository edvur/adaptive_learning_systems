"""
integration_setup_RL.py - RL Environment for Adaptive Tutor
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import random


# Import Integration Manager and Configuration
from integration_LS_AT import TutorIntegrationManager, Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Available content types"""
    VIDEO = "video"
    READING = "reading"
    INTERACTIVE = "interactive"
    QUIZ = "quiz"
    EXERCISE = "exercise"
    OVERVIEW = "overview"
    DISCUSSION = "discussion"
    PROJECT = "project"
    SUMMARY = "summary"

class DifficultyLevel(Enum):
    """Improved difficulty levels"""
    BEGINNER = 0.1
    EASY = 0.3
    MEDIUM = 0.5
    HARD = 0.7
    EXPERT = 0.9

@dataclass
class LearningContent:
    """Extended content definition with more attributes"""
    id: str
    title: str
    content_type: ContentType
    topic: str
    difficulty: float
    estimated_time: int  # minutes
    prerequisites: List[str] = field(default_factory=list)
    
    # Learning style fits (0-1 scale)
    sensing_fit: float = 0.5
    intuitive_fit: float = 0.5
    visual_fit: float = 0.5
    verbal_fit: float = 0.5
    sequential_fit: float = 0.5
    global_fit: float = 0.5
    
    # Additional properties
    engagement_factor: float = 0.5
    cognitive_load: float = 0.5
    retention_boost: float = 0.5
    
    # New properties
    interactive_elements: int = 0  # Number of interactive elements
    media_richness: float = 0.5  # How multimedia-rich is the content
    feedback_quality: float = 0.5  # Quality of feedback
    
    def get_fit_score(self, learning_style: Dict[str, Dict]) -> float:
        """Calculates weighted fit score for a specific learning style"""
        
        try:
            # Extract probabilities instead of just interpretations
            perception_probs = learning_style['Perception']['probabilities']
            input_probs = learning_style['Input']['probabilities']
            understanding_probs = learning_style['Understanding']['probabilities']
            
            # Weighted sum based on probabilities
            perception_fit = (
                self.sensing_fit * perception_probs.get('Sensing', 0.5) +
                self.intuitive_fit * perception_probs.get('Intuitive', 0.5)
            )
            
            input_fit = (
                self.visual_fit * input_probs.get('Visual', 0.5) +
                self.verbal_fit * input_probs.get('Verbal', 0.5)
            )
            
            understanding_fit = (
                self.sequential_fit * understanding_probs.get('Sequential', 0.5) +
                self.global_fit * understanding_probs.get('Global', 0.5)
            )
            
            # Adjusted weighting
            weights = {
                'perception': 0.3,
                'input': 0.4,  # Input has higher weight
                'understanding': 0.3
            }
            
            total_fit = (
                perception_fit * weights['perception'] +
                input_fit * weights['input'] +
                understanding_fit * weights['understanding']
            )
            
            # Bonus for high confidence
            avg_confidence = np.mean([
                learning_style['Perception']['confidence'],
                learning_style['Input']['confidence'],
                learning_style['Understanding']['confidence']
            ])
            
            if avg_confidence > 0.8:
                total_fit *= 1.1  # 10% bonus for high confidence
            
            return np.clip(total_fit, 0, 1)
            
        except Exception as e:
            logger.error(f"Error in fit score calculation: {e}")
            return 0.5  # Default medium fit

@dataclass
class StudentState:
    """Improved student state with more realistic attributes"""
    student_id: str
    
    # Learning style data
    learning_style_analysis: Dict[str, Dict]
    activity_patterns: Dict[str, float]
    
    # Current knowledge
    current_knowledge: Dict[str, float] = field(default_factory=dict)
    skill_levels: Dict[str, float] = field(default_factory=dict)
    
    # Current states
    engagement_level: float = 0.7
    fatigue_level: float = 0.0
    motivation_level: float = 0.7
    frustration_level: float = 0.0
    satisfaction_score: float = 0.5
    
    # Learning parameters
    difficulty_preference: float = 0.5
    learning_velocity: float = 0.5
    retention_rate: float = 0.8
    
    # Session data
    session_time: int = 0
    recent_performance: List[float] = field(default_factory=list)
    completed_content: List[str] = field(default_factory=list)
    
    # New attributes
    attention_span: float = 0.7
    error_rate: float = 0.2
    help_requests: int = 0
    streak_days: int = 0
    preferred_session_length: int = 45
    
    def to_observation_vector(self) -> np.ndarray:
        """Converts state to RL observation vector (30 features)"""
    
        try:
            # Learning style features (6 values)
            style_features = []
            for dim in ['Perception', 'Input', 'Understanding']:
                probs = self.learning_style_analysis[dim]['probabilities']
                # Ensure 2 values per dimension
                for key in sorted(probs.keys()):
                    style_features.append(float(probs[key]))  # Ensure float
            
            # Ensure exactly 6 style features
            while len(style_features) < 6:
                style_features.append(0.5)
            style_features = style_features[:6]
            
            # Knowledge Features (4)
            knowledge_values = list(self.current_knowledge.values()) or [0]
            avg_knowledge = float(np.mean(knowledge_values))
            knowledge_variance = float(np.var(knowledge_values)) if len(knowledge_values) > 1 else 0.0
            min_knowledge = float(np.min(knowledge_values)) if knowledge_values else 0.0
            max_knowledge = float(np.max(knowledge_values)) if knowledge_values else 0.0
            
            # Performance Features (3)
            if self.recent_performance:
                avg_performance = float(np.mean(self.recent_performance))
                performance_variance = float(np.var(self.recent_performance))
                performance_trend = float(self._calculate_trend(self.recent_performance))
            else:
                avg_performance = 0.5
                performance_variance = 0.0
                performance_trend = 0.0
            
            # Session Features (2)
            normalized_session_time = float(np.tanh(self.session_time / 60.0))
            session_progress = float(self.session_time / self.preferred_session_length)
            
            # Activity Pattern Features (3)
            reading_pref = float(self.activity_patterns.get('reading_preference', 0))
            visual_pref = float(self.activity_patterns.get('visual_preference', 0))
            interactive_pref = float(self.activity_patterns.get('interactive_preference', 0))
            
            # Current State Features (6)
            current_state_features = [
                float(self.engagement_level),
                float(self.fatigue_level),
                float(self.motivation_level),
                float(self.frustration_level),
                float(self.satisfaction_score),
                float(self.attention_span)
            ]
            
            # Meta-Learning Features (4)
            meta_features = [
                float(self.learning_velocity),
                float(self.retention_rate),
                float(self.difficulty_preference),
                float(self.error_rate)
            ]
            
            # Additional Features to reach 30 (2)
            additional_features = [
                float(len(self.completed_content)) / 20.0,  # Normalized completed content
                float(self.help_requests) / 10.0  # Normalized help requests
            ]
            
            # Combine all features into a single list
            all_features = []
            all_features.extend(style_features)  # 6
            all_features.append(avg_knowledge)  # 1
            all_features.append(knowledge_variance)  # 1
            all_features.append(min_knowledge)  # 1
            all_features.append(max_knowledge)  # 1
            all_features.append(avg_performance)  # 1
            all_features.append(performance_variance)  # 1
            all_features.append(performance_trend)  # 1
            all_features.extend(current_state_features)  # 6
            all_features.append(normalized_session_time)  # 1
            all_features.append(session_progress)  # 1
            all_features.append(reading_pref)  # 1
            all_features.append(visual_pref)  # 1
            all_features.append(interactive_pref)  # 1
            all_features.extend(meta_features)  # 4
            all_features.extend(additional_features)  # 2
            
            # Convert to numpy array
            observation = np.array(all_features, dtype=np.float32)
            
            # Ensure exactly 30 features
            if len(observation) < 30:
                # Pad with zeros if needed
                observation = np.pad(observation, (0, 30 - len(observation)), constant_values=0)
            elif len(observation) > 30:
                # Truncate if too many
                observation = observation[:30]
            
            # Clip to valid range
            return np.clip(observation, -1, 1)
            
        except Exception as e:
            logger.error(f"Error in observation vector creation: {e}")
            # Fallback: Return zero vector with correct size
            return np.zeros(30, dtype=np.float32)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculates trend of recent values"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Normalize x for numerical stability
        x_norm = (x - x.mean()) / (x.std() + 1e-8)
        
        # Calculate slope
        slope = np.cov(x_norm, y)[0, 1] / (np.var(x_norm) + 1e-8)
        
        return np.tanh(slope)  # Tanh for limiting to [-1, 1]

class TutorAction:
    """Structured tutor action"""
    
    def __init__(self, 
                 content_id: str,
                 explanation_depth: float = 0.5,
                 interaction_level: float = 0.5,
                 encouragement_type: int = 0,
                 break_suggestion: bool = False,
                 difficulty_adjustment: float = 0.0,
                 recap_previous: bool = False,
                 provide_hint: bool = False,
                 adaptive_feedback: float = 0.5):
        
        self.content_id = content_id
        self.explanation_depth = np.clip(explanation_depth, 0, 1)
        self.interaction_level = np.clip(interaction_level, 0, 1)
        self.encouragement_type = encouragement_type
        self.break_suggestion = break_suggestion
        self.difficulty_adjustment = np.clip(difficulty_adjustment, -0.5, 0.5)
        self.recap_previous = recap_previous
        self.provide_hint = provide_hint
        self.adaptive_feedback = np.clip(adaptive_feedback, 0, 1)
    
    def to_dict(self) -> Dict:
        """Converts to dictionary"""
        return {
            'content_id': self.content_id,
            'explanation_depth': self.explanation_depth,
            'interaction_level': self.interaction_level,
            'encouragement_type': self.encouragement_type,
            'break_suggestion': self.break_suggestion,
            'difficulty_adjustment': self.difficulty_adjustment,
            'recap_previous': self.recap_previous,
            'provide_hint': self.provide_hint,
            'adaptive_feedback': self.adaptive_feedback
        }

class AdaptiveLearningEnvironment(gym.Env):
    """RL environment for adaptive learning tutor"""
    
    def __init__(self, 
                 integration_manager: TutorIntegrationManager,
                 content_library: List[LearningContent],
                 config: Config = None,
                 max_session_time: int = 60,
                 topics: List[str] = None,
                 reward_config: Dict[str, float] = None):
        
        super().__init__()
        
        self.integration_manager = integration_manager
        self.content_library = content_library
        self.config = config or Config()
        self.max_session_time = max_session_time
        self.topics = topics or ["JavaScript", "React", "APIs", "Testing", "Deployment"]
        
        # Reward Configuration
        self.reward_config = reward_config or {
            'performance_weight': 0.3,
            'engagement_weight': 0.25,
            'learning_weight': 0.25,
            'efficiency_weight': 0.1,
            'satisfaction_weight': 0.1
        }
        
        # Action space: continuous for more flexibility
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),  # Extended for more action parameters
            dtype=np.float32
        )
        
        # Observation Space: 30 Features
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(30,),
            dtype=np.float32
        )
        
        # State
        self.current_student: Optional[StudentState] = None
        self.step_count = 0
        self.episode_reward = 0
        self.episode_info = {
            'learning_gains': [],
            'performances': [],
            'engagement_levels': []
        }
        
        # Content mapping for faster access
        self.content_by_id = {c.id: c for c in content_library}
        self.content_by_topic = self._organize_content_by_topic()
        
        logger.info("🎮 Adaptive Learning Environment initialized")
        logger.info(f"   Content Library: {len(content_library)} Content Items")
        logger.info(f"   Topics: {topics}")
        logger.info(f"   Max Session: {max_session_time} min")
    
    def _organize_content_by_topic(self) -> Dict[str, List[LearningContent]]:
        """Organize content by topics"""
        content_by_topic = {}
        for content in self.content_library:
            if content.topic not in content_by_topic:
                content_by_topic[content.topic] = []
            content_by_topic[content.topic].append(content)
        return content_by_topic
    
    def reset(self, student_id: str = None) -> np.ndarray:
        """Reset for new student or new session"""
        
        if student_id is None:
            student_id = f"student_{random.randint(1000, 9999)}"
        
        # Get or create student profile
        profile = self.integration_manager.get_student_profile(student_id)
        
        if profile is None:
            # Generate demo activity data
            demo_activity = self._generate_demo_activity_data()
            profile = self.integration_manager.process_new_student(student_id, demo_activity)
        
        # Create student state
        self.current_student = self._create_student_state(profile)
        
        # Reset episode variables
        self.step_count = 0
        self.episode_reward = 0
        self.episode_info = {
            'learning_gains': [],
            'performances': [],
            'engagement_levels': []
        }
        
        logger.info(f"Environment reset for Student: {student_id}")
        
        return self.current_student.to_observation_vector()
    
    def _create_student_state(self, profile: Dict) -> StudentState:
        """Create student state from profile"""
        
        # Initial knowledge based on activity patterns
        initial_knowledge = {}
        for topic in self.topics:
            # Base knowledge on total engagement
            base_knowledge = min(0.3, profile['activity_patterns']['total_engagement'] / 100)
            initial_knowledge[topic] = base_knowledge + random.uniform(-0.1, 0.1)
        
        # Calculate initial parameters from profile
        activity_total = profile['activity_patterns']['total_engagement']
        
        return StudentState(
            student_id=profile['student_id'],
            learning_style_analysis=profile['learning_style'],
            activity_patterns=profile['activity_patterns'],
            current_knowledge=initial_knowledge,
            skill_levels={},
            engagement_level=min(0.9, 0.5 + activity_total / 50),
            fatigue_level=0.0,
            motivation_level=0.7 + random.uniform(-0.1, 0.1),
            difficulty_preference=0.4 + random.uniform(-0.1, 0.1),
            session_time=0,
            recent_performance=[0.5 + random.uniform(-0.1, 0.1) for _ in range(3)],
            completed_content=[],
            learning_velocity=0.5 + random.uniform(-0.2, 0.2),
            retention_rate=0.7 + random.uniform(-0.1, 0.1),
            attention_span=0.7 + random.uniform(-0.1, 0.1),
            preferred_session_length=45 + random.randint(-15, 15)
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute a tutor action"""
        
        if self.current_student is None:
            raise ValueError("Environment must be initialized with reset() first!")
        
        # Parse Action
        tutor_action = self._parse_action(action)
        
        # Validate content
        if tutor_action.content_id not in self.content_by_id:
            logger.warning(f"Invalid content ID: {tutor_action.content_id}")
            # Choose random valid content
            tutor_action.content_id = random.choice(list(self.content_by_id.keys()))
        
        # Simulate student response
        response = self._simulate_student_response(tutor_action)
        
        # Update Student State
        self._update_student_state(tutor_action, response)
        
        # Calculate reward
        reward = self._calculate_reward(tutor_action, response)
        
        # Check if episode done
        done = self._is_episode_done()
        
        # Collect episode info
        self.episode_info['learning_gains'].append(response.get('knowledge_gain', 0))
        self.episode_info['performances'].append(response.get('performance', 0))
        self.episode_info['engagement_levels'].append(self.current_student.engagement_level)
        
        # Info for debugging and analysis
        info = {
            'student_id': self.current_student.student_id,
            'action': tutor_action.to_dict(),
            'response': response,
            'state': {
                'engagement': self.current_student.engagement_level,
                'fatigue': self.current_student.fatigue_level,
                'motivation': self.current_student.motivation_level,
                'avg_knowledge': np.mean(list(self.current_student.current_knowledge.values()))
            },
            'reward_components': response.get('reward_components', {}),
            'step_count': self.step_count
        }
        
        # Log Session Event
        self.integration_manager.log_session_event(
            self.current_student.student_id,
            {
                'step': self.step_count,
                'content': tutor_action.content_id,
                'performance': response.get('performance', 0),
                'reward': reward
            }
        )
        
        self.step_count += 1
        self.episode_reward += reward
        
        return (
            self.current_student.to_observation_vector(),
            reward,
            done,
            info
        )
    
    def _parse_action(self, action: np.ndarray) -> TutorAction:
        """Convert RL action to TutorAction"""
        
        # Ensure action has correct shape
        if len(action) < 9:
            action = np.pad(action, (0, 9 - len(action)), constant_values=0.5)
        
        # Content selection based on multiple factors
        content_score = action[0]
        
        # Intelligent content selection
        available_content = [
            c for c in self.content_library 
            if c.id not in self.current_student.completed_content[-3:]  # Don't repeat last 3
        ]
        
        if not available_content:
            available_content = self.content_library
        
        # Weight content by fit score and action
        content_weights = []
        for content in available_content:
            fit_score = content.get_fit_score(self.current_student.learning_style_analysis)
            difficulty_match = 1 - abs(content.difficulty - self.current_student.difficulty_preference)
            
            weight = (fit_score * 0.5 + difficulty_match * 0.3 + content_score * 0.2)
            content_weights.append(weight)
        
        # Choose content probabilistically
        content_weights = np.array(content_weights)
        content_weights = content_weights / content_weights.sum()
        
        selected_idx = np.random.choice(len(available_content), p=content_weights)
        selected_content = available_content[selected_idx]
        
        return TutorAction(
            content_id=selected_content.id,
            explanation_depth=action[1],
            interaction_level=action[2],
            encouragement_type=int(action[3] * 3),  # 0-2
            break_suggestion=action[4] > 0.5,
            difficulty_adjustment=(action[5] - 0.5),  # -0.5 bis 0.5
            recap_previous=action[6] > 0.5,
            provide_hint=action[7] > 0.5,
            adaptive_feedback=action[8]
        )
    
    def _simulate_student_response(self, action: TutorAction) -> Dict[str, float]:
        """Improved simulation of student reaction"""
        
        content = self.content_by_id[action.content_id]
        
        # Base performance factors
        content_fit = content.get_fit_score(self.current_student.learning_style_analysis)
        
        # Difficulty adjustment
        adjusted_difficulty = content.difficulty + action.difficulty_adjustment
        difficulty_delta = abs(adjusted_difficulty - self.current_student.difficulty_preference)
        difficulty_match = np.exp(-2 * difficulty_delta)  # Exponential decay
        
        # Engagement-based modifiers
        engagement_modifier = self.current_student.engagement_level * 0.3 + 0.7
        
        # Fatigue penalty
        fatigue_penalty = 1 - (self.current_student.fatigue_level * 0.4)
        
        # Motivation boost
        motivation_boost = 1 + (self.current_student.motivation_level - 0.5) * 0.2
        
        # Interaction bonus
        interaction_bonus = 1.0
        if action.interaction_level > 0.7 and content.interactive_elements > 0:
            interaction_bonus = 1.15
        
        # Explanation depth effect
        explanation_effect = 1.0
        if self.current_student.error_rate > 0.3 and action.explanation_depth > 0.7:
            explanation_effect = 1.1
        elif self.current_student.error_rate < 0.1 and action.explanation_depth < 0.3:
            explanation_effect = 1.05  # Faster learners benefit from less explanation
        
        # Calculate final performance
        base_performance = (
            content_fit * 0.25 +
            difficulty_match * 0.25 +
            engagement_modifier * 0.2 +
            content.engagement_factor * 0.15 +
            random.uniform(0.1, 0.3) * 0.15  # Randomness
        )
        
        performance = np.clip(
            base_performance * fatigue_penalty * motivation_boost * 
            interaction_bonus * explanation_effect,
            0, 1
        )
        
        # Engagement Change
        engagement_change = 0
        
        # Positive factors
        if content_fit > 0.7:
            engagement_change += 0.03
        if performance > 0.8:
            engagement_change += 0.04
        if action.encouragement_type > 0:
            engagement_change += 0.02 * action.encouragement_type
        if action.interaction_level > 0.6:
            engagement_change += 0.02
        
        # Negative factors
        if performance < 0.4:
            engagement_change -= 0.05
        if difficulty_delta > 0.3:
            engagement_change -= 0.03
        if self.current_student.fatigue_level > 0.7:
            engagement_change -= 0.04
        
        # Fatigue Change
        base_fatigue = content.estimated_time / 100.0
        cognitive_load_factor = content.cognitive_load * 0.5
        
        fatigue_change = base_fatigue * (1 + cognitive_load_factor)
        
        # Break reduces fatigue
        if action.break_suggestion and self.current_student.fatigue_level > 0.5:
            fatigue_change = -0.1
        
        # Knowledge Gain
        knowledge_gain = (
            performance * 
            self.current_student.learning_velocity * 
            content.retention_boost *
            0.1  # Scaling factor
        )
        
        # Motivation Change
        motivation_change = 0
        if performance > 0.7:
            motivation_change += 0.03
        if self.current_student.streak_days > 3:
            motivation_change += 0.02
        if performance < 0.3:
            motivation_change -= 0.04
        
        # Error Rate Update
        error_rate_change = 0
        if performance > 0.8:
            error_rate_change = -0.02
        elif performance < 0.4:
            error_rate_change = 0.03
        
        # Reward components for detailed tracking
        reward_components = {
            'performance_reward': performance * self.reward_config['performance_weight'],
            'engagement_reward': engagement_change * 10 * self.reward_config['engagement_weight'],
            'learning_reward': knowledge_gain * 10 * self.reward_config['learning_weight'],
            'efficiency_reward': (performance / (content.estimated_time / 30)) * self.reward_config['efficiency_weight'],
            'satisfaction_reward': (performance - 0.5) * 2 * self.reward_config['satisfaction_weight']
        }
        
        return {
            'performance': performance,
            'engagement_change': engagement_change,
            'fatigue_change': fatigue_change,
            'knowledge_gain': knowledge_gain,
            'motivation_change': motivation_change,
            'error_rate_change': error_rate_change,
            'time_spent': content.estimated_time,
            'satisfaction_change': (performance - 0.5) * 0.1,
            'reward_components': reward_components
        }
    
    def _update_student_state(self, action: TutorAction, response: Dict[str, float]):
        """Updates student state with improved logic"""
        
        # Performance History
        self.current_student.recent_performance.append(response['performance'])
        if len(self.current_student.recent_performance) > 10:  # Longer history
            self.current_student.recent_performance.pop(0)
        
        # Engagement with momentum
        current_engagement = self.current_student.engagement_level
        new_engagement = current_engagement + response['engagement_change']
        # Smoothing for more stable updates
        self.current_student.engagement_level = np.clip(
            0.8 * current_engagement + 0.2 * new_engagement, 0, 1
        )
        
        # Fatigue
        new_fatigue = self.current_student.fatigue_level + response['fatigue_change']
        self.current_student.fatigue_level = np.clip(new_fatigue, 0, 1)
        
        # Knowledge Update
        content = self.content_by_id[action.content_id]
        topic = content.topic
        
        if topic in self.current_student.current_knowledge:
            # Direct topic improvement
            current = self.current_student.current_knowledge[topic]
            gain = response['knowledge_gain']
            
            # Consider retention
            retained = current * self.current_student.retention_rate
            new_knowledge = retained + gain
            
            self.current_student.current_knowledge[topic] = min(1.0, new_knowledge)
        
        # Spillover to related topics
        for other_topic in self.current_student.current_knowledge:
            if other_topic != topic:
                spillover = response['knowledge_gain'] * 0.1
                current = self.current_student.current_knowledge[other_topic]
                self.current_student.current_knowledge[other_topic] = min(1.0, current + spillover)
        
        # Session Time
        self.current_student.session_time += response['time_spent']
        
        # Motivation
        new_motivation = self.current_student.motivation_level + response.get('motivation_change', 0)
        self.current_student.motivation_level = np.clip(new_motivation, 0, 1)
        
        # Error Rate
        new_error_rate = self.current_student.error_rate + response.get('error_rate_change', 0)
        self.current_student.error_rate = np.clip(new_error_rate, 0, 0.5)
        
        # Satisfaction (Exponential Moving Average)
        satisfaction_change = response.get('satisfaction_change', 0)
        self.current_student.satisfaction_score = np.clip(
            0.9 * self.current_student.satisfaction_score + 0.1 * (0.5 + satisfaction_change),
            0, 1
        )
        
        # Frustration
        if response['performance'] < 0.3:
            self.current_student.frustration_level = min(1.0, self.current_student.frustration_level + 0.1)
        else:
            self.current_student.frustration_level = max(0.0, self.current_student.frustration_level - 0.05)
        
        # Help Requests
        if action.provide_hint:
            self.current_student.help_requests += 1
        
        # Completed Content
        self.current_student.completed_content.append(action.content_id)
        
        # Update learning velocity based on performance trend
        if len(self.current_student.recent_performance) >= 5:
            trend = self.current_student._calculate_trend(self.current_student.recent_performance)
            if trend > 0.1:
                self.current_student.learning_velocity = min(1.0, self.current_student.learning_velocity + 0.02)
            elif trend < -0.1:
                self.current_student.learning_velocity = max(0.1, self.current_student.learning_velocity - 0.02)
    
    def _calculate_reward(self, action: TutorAction, response: Dict[str, float]) -> float:
        """Enhanced educational reward system focused on learning outcomes"""
        
        # Core Educational Metrics (weighted heavily)
        performance = response.get('performance', 0)
        knowledge_gain = response.get('knowledge_gain', 0)
        engagement_change = response.get('engagement_change', 0)
        
        # 1. LEARNING EFFECTIVENESS (40% of reward)
        learning_reward = 0
        
        # Performance quality with progressive scaling
        if performance > 0.8:
            learning_reward += 2.0  # Excellent performance
        elif performance > 0.6:
            learning_reward += 1.0  # Good performance  
        elif performance > 0.4:
            learning_reward += 0.3  # Acceptable performance
        else:
            learning_reward -= 0.5  # Poor performance penalty
        
        # Knowledge gain multiplier
        learning_reward *= (1.0 + knowledge_gain * 2.0)
        
        # 2. ADAPTIVE PERSONALIZATION (25% of reward)
        content = self.content_by_id[action.content_id]
        content_fit = content.get_fit_score(self.current_student.learning_style_analysis)
        
        personalization_reward = 0
        if content_fit > 0.8:
            personalization_reward += 1.0  # Excellent fit
        elif content_fit > 0.6:
            personalization_reward += 0.5  # Good fit
        elif content_fit < 0.3:
            personalization_reward -= 0.8  # Poor fit penalty
        
        # Difficulty appropriateness
        current_knowledge = self.current_student.current_knowledge.get(content.topic, 0.5)
        difficulty_gap = abs(content.difficulty - current_knowledge)
        if difficulty_gap < 0.2:  # Just right
            personalization_reward += 0.8
        elif difficulty_gap > 0.5:  # Too hard/easy
            personalization_reward -= 0.5
        
        # 3. STUDENT WELLBEING (20% of reward)
        wellbeing_reward = 0
        
        # Engagement management
        if engagement_change > 0:
            wellbeing_reward += engagement_change * 2.0
        elif engagement_change < -0.1:
            wellbeing_reward -= abs(engagement_change) * 3.0
        
        # Fatigue management
        if action.break_suggestion and self.current_student.fatigue_level > 0.7:
            wellbeing_reward += 1.0  # Appropriate break
        elif action.break_suggestion and self.current_student.fatigue_level < 0.3:
            wellbeing_reward -= 0.5  # Unnecessary break
        
        # Frustration prevention
        if self.current_student.frustration_level > 0.8:
            wellbeing_reward -= 2.0  # High frustration penalty
        elif self.current_student.frustration_level < 0.3:
            wellbeing_reward += 0.3  # Low frustration bonus
        
        # 4. PEDAGOGICAL QUALITY (15% of reward)
        pedagogical_reward = 0
        
        # Appropriate help provision
        if action.provide_hint and self.current_student.error_rate > 0.4:
            pedagogical_reward += 0.8  # Good help timing
        elif action.provide_hint and self.current_student.error_rate < 0.1:
            pedagogical_reward -= 0.3  # Unnecessary help
        
        # Explanation depth appropriateness
        if action.explanation_depth > 0.7 and self.current_student.error_rate > 0.3:
            pedagogical_reward += 0.6  # Good detailed explanation
        elif action.explanation_depth < 0.3 and self.current_student.error_rate > 0.5:
            pedagogical_reward -= 0.4  # Insufficient explanation
        
        # Learning progression tracking
        if len(self.current_student.recent_performance) >= 3:
            trend = self._calculate_performance_trend()
            if trend > 0.1:
                pedagogical_reward += 1.0  # Improving trend
            elif trend < -0.1:
                pedagogical_reward -= 0.6  # Declining trend
        
        # 5. COMBINE REWARDS WITH WEIGHTS
        total_reward = (
            learning_reward * 0.40 +           # Learning effectiveness
            personalization_reward * 0.25 +   # Adaptive personalization  
            wellbeing_reward * 0.20 +          # Student wellbeing
            pedagogical_reward * 0.15          # Pedagogical quality
        )
        
        # 6. BONUS/PENALTY MODIFIERS
        
        # Learning streak bonus
        if len(self.current_student.recent_performance) >= 3:
            if all(p > 0.7 for p in self.current_student.recent_performance[-3:]):
                total_reward += 0.8  # Sustained excellence
        
        # Content diversity bonus (encourage exploration)
        recent_types = [self.content_by_id[cid].content_type 
                       for cid in self.current_student.completed_content[-5:] 
                       if cid in self.content_by_id]
        if len(set(recent_types)) >= 3:
            total_reward += 0.4
        
        # Session length management
        if self.current_student.session_time > 60:
            total_reward -= (self.current_student.session_time - 60) * 0.02
        
        # Critical state emergency penalties
        if self.current_student.engagement_level < 0.15:
            total_reward -= 3.0  # Emergency: student disengaging
        
        return total_reward
    
    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend from recent history"""
        if len(self.current_student.recent_performance) < 3:
            return 0
        
        performances = self.current_student.recent_performance[-5:]  # Last 5 episodes
        x = np.arange(len(performances))
        try:
            slope = np.polyfit(x, performances, 1)[0]
            return slope
        except:
            return 0
        
        return reward
    
    def _is_episode_done(self) -> bool:
        """Smarter episode end conditions"""
        
        # Hard limits
        if self.current_student.session_time >= self.max_session_time:
            return True
        
        if self.step_count >= 100:  # Maximum steps
            return True
        
        # Critical states
        if self.current_student.engagement_level < 0.15:
            return True
        
        if self.current_student.frustration_level > 0.9:
            return True
        
        if self.current_student.fatigue_level > 0.95:
            return True
        
        # Positive termination
        avg_knowledge = np.mean(list(self.current_student.current_knowledge.values()))
        if avg_knowledge > 0.85 and self.current_student.satisfaction_score > 0.8:
            return True  # Learning goals achieved
        
        # Minimum session length
        if self.step_count < 5:
            return False
        
        # Natural end
        if (self.current_student.session_time >= self.current_student.preferred_session_length and
            self.current_student.engagement_level < 0.5):
            return True
        
        return False
    
    def _generate_demo_activity_data(self) -> pd.DataFrame:
        """Generates more realistic demo activity data"""
        
        # Define learner type profiles
        learner_profiles = {
            'visual_active': {
                'Course overview': (2, 5),
                'Reading file': (3, 8),
                'Abstract materiale': (1, 3),
                'Concrete material': (8, 15),
                'Visual Materials': (15, 30),
                'Self-assessment': (2, 5),
                'Exercises submit': (8, 15),
                'Quiz submitted': (5, 10),
                'playing': (20, 40),
                'paused': (5, 15),
                'unstarted': (1, 3),
                'buffering': (0, 2)
            },
            'verbal_reflective': {
                'Course overview': (5, 12),
                'Reading file': (20, 40),
                'Abstract materiale': (10, 20),
                'Concrete material': (3, 8),
                'Visual Materials': (3, 10),
                'Self-assessment': (5, 10),
                'Exercises submit': (3, 8),
                'Quiz submitted': (3, 7),
                'playing': (5, 15),
                'paused': (2, 5),
                'unstarted': (3, 8),
                'buffering': (0, 2)
            },
            'balanced': {
                'Course overview': (3, 8),
                'Reading file': (10, 20),
                'Abstract materiale': (5, 10),
                'Concrete material': (8, 15),
                'Visual Materials': (10, 20),
                'Self-assessment': (3, 7),
                'Exercises submit': (5, 12),
                'Quiz submitted': (4, 8),
                'playing': (12, 25),
                'paused': (3, 8),
                'unstarted': (2, 5),
                'buffering': (0, 3)
            }
        }
        
        # Choose random profile
        profile_name = random.choice(list(learner_profiles.keys()))
        profile = learner_profiles[profile_name]
        
        # Generate data with variance
        activity_data = {}
        for activity, (min_val, max_val) in profile.items():
            # Normal distribution around midpoint
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 4
            value = max(min_val, min(max_val, int(np.random.normal(mean, std))))
            activity_data[activity] = value
        
        logger.info(f"Generated demo data for profile: {profile_name}")
        
        return pd.DataFrame([activity_data])
    
    def get_student_summary(self) -> Dict:
        """Extended summary of current student"""
        
        if self.current_student is None:
            return {}
        
        # Calculate aggregated metrics
        avg_knowledge = np.mean(list(self.current_student.current_knowledge.values()))
        knowledge_progress = avg_knowledge - np.mean([0.3] * len(self.current_student.current_knowledge))
        
        avg_performance = np.mean(self.current_student.recent_performance) if self.current_student.recent_performance else 0
        
        # Content analysis
        completed_types = []
        for content_id in self.current_student.completed_content:
            if content_id in self.content_by_id:
                completed_types.append(self.content_by_id[content_id].content_type.value)
        
        type_distribution = {}
        for ctype in set(completed_types):
            type_distribution[ctype] = completed_types.count(ctype)
        
        return {
            'student_id': self.current_student.student_id,
            'learning_style': {
                dim: analysis['interpretation'] 
                for dim, analysis in self.current_student.learning_style_analysis.items()
            },
            'state_metrics': {
                'engagement': round(self.current_student.engagement_level, 3),
                'fatigue': round(self.current_student.fatigue_level, 3),
                'motivation': round(self.current_student.motivation_level, 3),
                'frustration': round(self.current_student.frustration_level, 3),
                'satisfaction': round(self.current_student.satisfaction_score, 3)
            },
            'learning_metrics': {
                'avg_knowledge': round(avg_knowledge, 3),
                'knowledge_progress': round(knowledge_progress, 3),
                'avg_performance': round(avg_performance, 3),
                'error_rate': round(self.current_student.error_rate, 3),
                'learning_velocity': round(self.current_student.learning_velocity, 3)
            },
            'session_info': {
                'session_time': self.current_student.session_time,
                'completed_content': len(self.current_student.completed_content),
                'help_requests': self.current_student.help_requests,
                'content_type_distribution': type_distribution
            },
            'episode_summary': {
                'total_reward': round(self.episode_reward, 2),
                'avg_learning_gain': round(np.mean(self.episode_info['learning_gains']), 3) if self.episode_info['learning_gains'] else 0,
                'avg_performance': round(np.mean(self.episode_info['performances']), 3) if self.episode_info['performances'] else 0
            }
        }
    
    def render(self, mode='human'):
        """Optional: Visualization of current state"""
        if mode == 'human':
            summary = self.get_student_summary()
            print("\n" + "="*50)
            print(f"Student: {summary['student_id']}")
            print(f"Step: {self.step_count}, Time: {summary['session_info']['session_time']} min")
            print(f"Engagement: {summary['state_metrics']['engagement']:.2f}")
            print(f"Knowledge: {summary['learning_metrics']['avg_knowledge']:.2f}")
            print(f"Performance: {summary['learning_metrics']['avg_performance']:.2f}")
            print(f"Reward: {self.episode_reward:.2f}")
            print("="*50)


def create_enhanced_content_library() -> List[LearningContent]:
    """Creates an enhanced content library"""
    
    contents = []
    
    # JavaScript basics
    contents.extend([
        LearningContent(
            id="js_intro_video",
            title="JavaScript Introduction - Interactive Video",
            content_type=ContentType.VIDEO,
            topic="JavaScript",
            difficulty=0.2,
            estimated_time=15,
            prerequisites=[],
            visual_fit=0.9, verbal_fit=0.3, sequential_fit=0.8, global_fit=0.4,
            sensing_fit=0.3, intuitive_fit=0.7, engagement_factor=0.8,
            cognitive_load=0.3, retention_boost=0.7,
            interactive_elements=3, media_richness=0.9, feedback_quality=0.8
        ),
        
        LearningContent(
            id="js_variables_reading",
            title="Variables and Data Types - Detailed Guide",
            content_type=ContentType.READING,
            topic="JavaScript",
            difficulty=0.3,
            estimated_time=20,
            prerequisites=["js_intro_video"],
            visual_fit=0.2, verbal_fit=0.9, sequential_fit=0.9, global_fit=0.2,
            sensing_fit=0.7, intuitive_fit=0.3, engagement_factor=0.4,
            cognitive_load=0.5, retention_boost=0.8,
            interactive_elements=0, media_richness=0.2, feedback_quality=0.3
        ),
        
        LearningContent(
            id="js_functions_interactive",
            title="Creating Functions - Interactive Exercise",
            content_type=ContentType.INTERACTIVE,
            topic="JavaScript",
            difficulty=0.5,
            estimated_time=30,
            prerequisites=["js_variables_reading"],
            visual_fit=0.7, verbal_fit=0.3, sequential_fit=0.6, global_fit=0.4,
            sensing_fit=0.9, intuitive_fit=0.1, engagement_factor=0.9,
            cognitive_load=0.7, retention_boost=0.9,
            interactive_elements=8, media_richness=0.7, feedback_quality=0.9
        ),
        
        LearningContent(
            id="js_project_calculator",
            title="Mini Project: Build Calculator",
            content_type=ContentType.PROJECT,
            topic="JavaScript",
            difficulty=0.6,
            estimated_time=45,
            prerequisites=["js_functions_interactive"],
            visual_fit=0.6, verbal_fit=0.4, sequential_fit=0.5, global_fit=0.5,
            sensing_fit=0.9, intuitive_fit=0.1, engagement_factor=0.95,
            cognitive_load=0.8, retention_boost=0.95,
            interactive_elements=12, media_richness=0.6, feedback_quality=0.7
        )
    ])
    
    # React components
    contents.extend([
        LearningContent(
            id="react_intro_overview",
            title="React Overview - The Big Picture",
            content_type=ContentType.OVERVIEW,
            topic="React",
            difficulty=0.4,
            estimated_time=10,
            prerequisites=["js_functions_interactive"],
            visual_fit=0.8, verbal_fit=0.5, sequential_fit=0.2, global_fit=0.9,
            sensing_fit=0.3, intuitive_fit=0.7, engagement_factor=0.6,
            cognitive_load=0.4, retention_boost=0.5,
            interactive_elements=2, media_richness=0.7, feedback_quality=0.5
        ),
        
        LearningContent(
            id="react_components_video",
            title="Understanding Components - Visual Tutorial",
            content_type=ContentType.VIDEO,
            topic="React",
            difficulty=0.5,
            estimated_time=20,
            prerequisites=["react_intro_overview"],
            visual_fit=0.9, verbal_fit=0.4, sequential_fit=0.7, global_fit=0.3,
            sensing_fit=0.6, intuitive_fit=0.4, engagement_factor=0.8,
            cognitive_load=0.6, retention_boost=0.7,
            interactive_elements=4, media_richness=0.9, feedback_quality=0.7
        ),
        
        LearningContent(
            id="react_state_exercise",
            title="State Management - Practical Exercises",
            content_type=ContentType.EXERCISE,
            topic="React",
            difficulty=0.7,
            estimated_time=35,
            prerequisites=["react_components_video"],
            visual_fit=0.5, verbal_fit=0.5, sequential_fit=0.8, global_fit=0.2,
            sensing_fit=0.8, intuitive_fit=0.2, engagement_factor=0.85,
            cognitive_load=0.8, retention_boost=0.85,
            interactive_elements=10, media_richness=0.5, feedback_quality=0.9
        ),
        
        LearningContent(
            id="react_hooks_quiz",
            title="React Hooks - Knowledge Test",
            content_type=ContentType.QUIZ,
            topic="React",
            difficulty=0.6,
            estimated_time=15,
            prerequisites=["react_state_exercise"],
            visual_fit=0.4, verbal_fit=0.6, sequential_fit=0.9, global_fit=0.1,
            sensing_fit=0.7, intuitive_fit=0.3, engagement_factor=0.7,
            cognitive_load=0.6, retention_boost=0.8,
            interactive_elements=15, media_richness=0.3, feedback_quality=0.95
        )
    ])
    
    # APIs
    contents.extend([
        LearningContent(
            id="api_basics_reading",
            title="REST API Basics - Comprehensive Text",
            content_type=ContentType.READING,
            topic="APIs",
            difficulty=0.5,
            estimated_time=25,
            prerequisites=["js_functions_interactive"],
            visual_fit=0.3, verbal_fit=0.8, sequential_fit=0.8, global_fit=0.2,
            sensing_fit=0.5, intuitive_fit=0.5, engagement_factor=0.5,
            cognitive_load=0.7, retention_boost=0.7,
            interactive_elements=0, media_richness=0.3, feedback_quality=0.4
        ),
        
        LearningContent(
            id="api_fetch_interactive",
            title="Fetching Data with Fetch - Live Coding",
            content_type=ContentType.INTERACTIVE,
            topic="APIs",
            difficulty=0.7,
            estimated_time=40,
            prerequisites=["api_basics_reading", "react_state_exercise"],
            visual_fit=0.7, verbal_fit=0.3, sequential_fit=0.7, global_fit=0.3,
            sensing_fit=0.9, intuitive_fit=0.1, engagement_factor=0.9,
            cognitive_load=0.8, retention_boost=0.9,
            interactive_elements=12, media_richness=0.7, feedback_quality=0.85
        )
    ])
    
    # Testing
    contents.extend([
        LearningContent(
            id="testing_intro_discussion",
            title="Why Testing is Important - Discussion",
            content_type=ContentType.DISCUSSION,
            topic="Testing",
            difficulty=0.3,
            estimated_time=15,
            prerequisites=[],
            visual_fit=0.3, verbal_fit=0.7, sequential_fit=0.4, global_fit=0.6,
            sensing_fit=0.4, intuitive_fit=0.6, engagement_factor=0.7,
            cognitive_load=0.3, retention_boost=0.6,
            interactive_elements=5, media_richness=0.2, feedback_quality=0.7
        ),
        
        LearningContent(
            id="testing_jest_tutorial",
            title="Jest Testing Framework - Hands-on Tutorial",
            content_type=ContentType.INTERACTIVE,
            topic="Testing",
            difficulty=0.6,
            estimated_time=45,
            prerequisites=["testing_intro_discussion", "js_functions_interactive"],
            visual_fit=0.6, verbal_fit=0.4, sequential_fit=0.8, global_fit=0.2,
            sensing_fit=0.8, intuitive_fit=0.2, engagement_factor=0.85,
            cognitive_load=0.7, retention_boost=0.85,
            interactive_elements=15, media_richness=0.6, feedback_quality=0.9
        )
    ])
    
    logger.info(f"Enhanced content library created with {len(contents)} items")
    
    return contents


def test_improved_rl_environment():
    """Test RL Environment"""
    
    print(" TESTING RL ENVIRONMENT")
    print("=" * 60)
    
    # Setup
    config = Config()
    
    # Initialize integration manager
    integration_manager = TutorIntegrationManager(config)
    if not integration_manager.initialize():
        print(" Integration Manager Initialization failed")
        return
    
    # Create Content Library
    content_library = create_enhanced_content_library()
    print(f"📚 Content Library: {len(content_library)} items")
    
    # Initialize environment
    env = AdaptiveLearningEnvironment(
        integration_manager=integration_manager,
        content_library=content_library,
        config=config,
        max_session_time=60,
        topics=["JavaScript", "React", "APIs", "Testing"]
    )
    
    print(f"🎮 Environment initialized")
    print(f"   Action Space: {env.action_space.shape}")
    print(f"   Observation Space: {env.observation_space.shape}")
    
    # Test 1: Reset and Observation
    print(f"\n TEST 1: Environment Reset")
    obs = env.reset("test_student_001")
    print(f"   Observation Shape: {obs.shape}")
    print(f"   Observation Range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test 2: Action Execution
    print(f"\n🎯 TEST 2: Action Execution")
    
    total_reward = 0
    for step in range(5):
        # Sample action
        action = env.action_space.sample()
        
        # Execute step
        next_obs, reward, done, info = env.step(action)
        
        total_reward += reward
        
        print(f"\n   Step {step + 1}:")
        print(f"   Content: {info['action']['content_id']}")
        print(f"   Performance: {info['response']['performance']:.3f}")
        print(f"   Reward: {reward:.3f}")
        print(f"   State: E={info['state']['engagement']:.2f}, "
              f"F={info['state']['fatigue']:.2f}, "
              f"K={info['state']['avg_knowledge']:.2f}")
        
        if done:
            print(f"   📊 Episode ended!")
            break
    
    # Test 3: Student Summary
    print(f"\n TEST 3: Student Summary")
    summary = env.get_student_summary()
    
    print(f"\n Learning Style:")
    for dim, style in summary['learning_style'].items():
        print(f"   {dim}: {style}")
    
    print(f"\n State Metrics:")
    for metric, value in summary['state_metrics'].items():
        print(f"   {metric}: {value}")
    
    print(f"\nLearning Metrics:")
    for metric, value in summary['learning_metrics'].items():
        print(f"   {metric}: {value}")
    
    print(f"\n⏱ Session Info:")
    print(f"   Time: {summary['session_info']['session_time']} min")
    print(f"   Completed Content: {summary['session_info']['completed_content']}")
    
    # Test 4: Different Student Types
    print(f"\n👥 TEST 4: Different Student Types")
    
    student_types = ["visual_learner", "verbal_learner", "balanced_learner"]
    
    for i, stype in enumerate(student_types):
        obs = env.reset(f"{stype}_{i+1}")
        
        rewards = []
        performances = []
        
        for _ in range(3):
            action = env.action_space.sample()
            _, reward, done, info = env.step(action)
            
            rewards.append(reward)
            performances.append(info['response']['performance'])
            
            if done:
                break
        
        print(f"\n   {stype}:")
        print(f"   Avg Reward: {np.mean(rewards):.3f}")
        print(f"   Avg Performance: {np.mean(performances):.3f}")
    
    # Test 5: Content Fit Analysis
    print(f"\n TEST 5: Content Fit Analysis")
    
    student_style = env.current_student.learning_style_analysis
    
    print(f"Current Student Learning Style:")
    for dim, analysis in student_style.items():
        print(f"   {dim}: {analysis['interpretation']} ({analysis['confidence']:.2%})")
    
    print(f"\n Top 5 Content Fits:")
    content_fits = []
    for content in content_library:
        fit = content.get_fit_score(student_style)
        content_fits.append((content, fit))
    
    content_fits.sort(key=lambda x: x[1], reverse=True)
    
    for content, fit in content_fits[:5]:
        print(f"   {content.title}: {fit:.3f}")
        print(f"     Type: {content.content_type.value}, Difficulty: {content.difficulty}")
    
    print(f"\n IMPROVED RL ENVIRONMENT TEST SUCCESSFUL!")
    
    return env


if __name__ == "__main__":
    # Run test
    env = test_improved_rl_environment()