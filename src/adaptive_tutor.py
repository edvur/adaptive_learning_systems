# adaptive_tutor_page.py - Adaptive tutor page with Deep RL integration

import streamlit as st
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional

# Import your RL components
from integration_LS_AT import TutorIntegrationManager, Config
from integration_setup_RL import AdaptiveLearningEnvironment, create_enhanced_content_library
from deep_rl_training import ImprovedDQNAgent

class AdaptiveTutorRL:
    """Integration of Deep RL model into Streamlit app"""
    
    def __init__(self):
        self.config = Config()
        self.integration_manager = None
        self.env = None
        self.agent = None
        self.content_library = None
        self.current_state = None
        self.session_history = []

    # def initialize_content_library(self):
    #     """Erstellt eine Demo Content Library falls RL nicht verfügbar"""
    #     self.content_library = [
    #         {
    #             'id': 'js_basics_1',
    #             'title': 'JavaScript Variablen und Datentypen',
    #             'content': '''
    #             ## JavaScript Variablen
                
    #             In JavaScript gibt es drei Möglichkeiten, Variablen zu deklarieren:
    #             - `let`: Für Variablen, die sich ändern können
    #             - `const`: Für konstante Werte
    #             - `var`: Veraltet, sollte vermieden werden
                
    #             ### Beispiel:
    #             ```javascript
    #             let name = "Max";
    #             const age = 25;
    #             ```
    #             ''',
    #             'difficulty': 1,
    #             'exercises': [
    #                 "// Deklariere eine Variable für deinen Namen\nlet myName = ____;",
    #                 "// Erstelle eine Konstante für PI\nconst PI = ____;"
    #             ],
    #             'hints': ["Verwende Anführungszeichen für Strings", "PI ist ungefähr 3.14159"],
    #             'content_type': 'reading',
    #             'estimated_time': 10
    #         },
    #         {
    #             'id': 'js_functions_1',
    #             'title': 'Funktionen in JavaScript',
    #             'content': '''
    #             ## Funktionen definieren
                
    #             Funktionen sind wiederverwendbare Codeblöcke:
                
    #             ```javascript
    #             function greet(name) {
    #                 return "Hallo " + name + "!";
    #             }
                
    #             // Arrow Function
    #             const add = (a, b) => a + b;
    #             ```
    #             ''',
    #             'difficulty': 2,
    #             'exercises': [
    #                 "// Schreibe eine Funktion, die zwei Zahlen multipliziert\nfunction multiply(a, b) {\n    return ____;\n}",
    #                 "// Konvertiere zu einer Arrow Function\nconst divide = ____ => ____;"
    #             ],
    #             'hints': ["Verwende den * Operator", "Arrow Functions: (params) => expression"],
    #             'content_type': 'interactive',
    #             'estimated_time': 15
    #         },
    #         {
    #             'id': 'react_components_1',
    #             'title': 'React Components Grundlagen',
    #             'content': '''
    #             ## React Components
                
    #             Components sind die Bausteine von React Apps:
                
    #             ```javascript
    #             function Welcome(props) {
    #                 return <h1>Hallo, {props.name}!</h1>;
    #             }
                
    #             // Mit useState Hook
    #             function Counter() {
    #                 const [count, setCount] = useState(0);
    #                 return (
    #                     <button onClick={() => setCount(count + 1)}>
    #                         Clicks: {count}
    #                     </button>
    #                 );
    #             }
    #             ```
    #             ''',
    #             'difficulty': 3,
    #             'exercises': [
    #                 "// Erstelle eine Komponente, die einen Namen anzeigt\nfunction Greeting(____) {\n    return <div>____</div>;\n}"
    #             ],
    #             'hints': ["Props werden als Parameter übergeben", "JSX verwendet geschweifte Klammern für JavaScript"],
    #             'content_type': 'interactive',
    #             'estimated_time': 20
    #         }
    #     ]
        
    @st.cache_resource
    def load_rl_model(_self, model_path: str = "improved_models_standard"):
        """Load the trained DQN model"""
        try:
            # Initialize components
            _self.integration_manager = TutorIntegrationManager(_self.config)
            _self.integration_manager.initialize()
            
            # Create content library
            _self.content_library = create_enhanced_content_library()
            
            # Create environment
            _self.env = AdaptiveLearningEnvironment(
                integration_manager=_self.integration_manager,
                content_library=_self.content_library,
                config=_self.config,
                max_session_time=60
            )
            
            # Load trained agent
            state_dim = _self.env.observation_space.shape[0]
            action_dim = _self.env.action_space.shape[0]
            
            _self.agent = ImprovedDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                config={'learning_rate': 1e-4}
            )
            
            # Load trained weights
            model_file = Path(model_path) / "models" / "best_model.pth"
            if model_file.exists():
                checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
                _self.agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                _self.agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                st.success("Deep RL Model successfully loaded!")
                return True
            else:
                st.warning("⚠️ No trained model found. Using random policy.")
                return False
                
        except Exception as e:
            st.error(f"Error loading RL model: {e}")
            return False
    
    def get_adaptive_recommendation(self, student_profile: Dict) -> tuple:
        """Generates adaptive recommendations based on the RL agent"""
        
        # Reset environment with student profile
        self.env.current_student_profile = student_profile
        state = self.env.reset()
        
        # Get action from agent
        with torch.no_grad():
            action = self.agent.act(state, epsilon=0.0)  # No exploration during inference
        
        # Decode action to recommendation
        recommendation = self._decode_action(action)
        
        # Store current state
        self.current_state = state
        
        return recommendation, state
    
    def _decode_action(self, action: np.ndarray) -> Dict:
        """Decodes RL action to concrete tutor recommendations"""
        
        # Action dimensions with safety checks
        content_index = int(action[0] * len(self.content_library))
        content_index = min(max(0, content_index), len(self.content_library) - 1)
        
        selected_content = self.content_library[content_index]
        
        # Interpret action values
        recommendation = {
            'content': selected_content,
            'explanation_depth': float(np.clip(action[1], 0, 1)),
            'interaction_level': float(np.clip(action[2], 0, 1)),
            'encouragement_type': int(action[3] * 3) % 3,  # 0-2
            'break_suggestion': bool(action[4] > 0.5),
            'difficulty_adjustment': float(np.clip(action[5] - 0.5, -0.5, 0.5)),
            'provide_hint': bool(action[6] > 0.5),
            'adaptive_feedback': float(np.clip(action[7], 0, 1))
        }
        
        return recommendation
    
    def update_learning_progress(self, student_response: Dict) -> tuple:
        """Updates learning progress based on student response"""
        
        if self.current_state is None:
            return 0, {}
        
        # Convert response to action format for environment
        action = np.array([
            student_response.get('completed', 0),
            student_response.get('accuracy', 0),
            student_response.get('time_spent', 0),
            student_response.get('help_used', 0),
            student_response.get('engagement', 0.5)
        ])
        
        # Step environment
        next_state, reward, done, info = self.env.step(action)
        
        # Store in session history
        self.session_history.append({
            'timestamp': datetime.now(),
            'state': self.current_state,
            'action': action,
            'reward': reward,
            'info': info
        })
        
        # Update current state
        self.current_state = next_state
        
        return reward, info

def render_adaptive_tutor_page():
    """Renders the adaptive tutor page with RL integration"""
    
    st.title("Adaptive AI Tutor")
    st.markdown("Personalized Learning with Deep Reinforcement Learning")
    
    # Initialize RL system
    if 'rl_tutor' not in st.session_state:
        with st.spinner("Loading AI Tutor System..."):
            st.session_state.rl_tutor = AdaptiveTutorRL()
            model_loaded = st.session_state.rl_tutor.load_rl_model()
            st.session_state.model_loaded = model_loaded
    
    # Initialize session state
    if 'current_exercise' not in st.session_state:
        st.session_state.current_exercise = None
    if 'exercise_start_time' not in st.session_state:
        st.session_state.exercise_start_time = None
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {
            'exercises_completed': 0,
            'total_score': 0,
            'time_spent': 0,
            'hints_used': 0
        }
    
    # Check for learning style data
    if 'user_data' not in st.session_state or st.session_state.user_data.get('learning_style') is None:
        st.warning("⚠️ Please complete the learning style test first!")
        if st.button("Go to Learning Style Test", type="primary"):
            st.session_state.page = "learning_style_test"
            st.rerun()
        return
    
    # Sidebar for learning settings
    with st.sidebar:
        st.header("Learning Settings")
        
        # Select learning goal
        learning_goal = st.selectbox(
            "What would you like to learn today?",
            ["JavaScript Basics", "React Components", "API Integration", "Testing", "TypeScript"]
        )
        
        # Time budget
        time_budget = st.slider("Available Time (Minutes)", 15, 120, 45)
        
        # Difficulty preference
        difficulty_pref = st.select_slider(
            "Preferred Difficulty",
            ["Very Easy", "Easy", "Medium", "Hard", "Very Hard"],
            "Medium"
        )
        
        st.divider()
        
        # Show current learning style
        st.subheader("Your Learning Style")
        learning_style = st.session_state.user_data['learning_style']
        
        style_colors = {
            "visual": "🎨",
            "verbal": "📝",
            "active": "🏃",
            "reflective": "🤔"
        }
        
        for dimension, data in learning_style.items():
            if isinstance(data, dict):
                icon = style_colors.get(dimension, "📊")
                st.metric(
                    f"{icon} {dimension.capitalize()}", 
                    data.get('interpretation', 'N/A'), 
                    f"{data.get('confidence', 0):.0%}"
                )
        
        st.divider()
        
        # Session Stats
        st.subheader("Session Statistics")
        st.metric("Exercises Completed", st.session_state.session_stats['exercises_completed'])
        st.metric("Total Score", f"{st.session_state.session_stats['total_score']:.0f}")
        st.metric("Time Invested", f"{st.session_state.session_stats['time_spent']:.0f} Min")
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Learning Content")
        
        # Create student profile for RL
        student_profile = {
            'learning_style': learning_style,
            'knowledge_levels': st.session_state.user_data.get('knowledge_levels', {}),
            'performance_history': st.session_state.user_data.get('performance_history', []),
            'engagement_level': 0.7,
            'fatigue_level': min(st.session_state.session_stats['time_spent'] / 60, 1.0),
            'preferences': {
                'goal': learning_goal,
                'time_budget': time_budget,
                'difficulty': difficulty_pref
            }
        }
        
        # Get RL recommendation
        if st.session_state.current_exercise is None:
            recommendation, state = st.session_state.rl_tutor.get_adaptive_recommendation(student_profile)
            st.session_state.current_exercise = recommendation
            st.session_state.exercise_start_time = time.time()
        else:
            recommendation = st.session_state.current_exercise
        
        # Show recommended content
        content = recommendation['content']
        
        # Content card with adaptive styling
        with st.container():
            # Header with difficulty display
            col_h1, col_h2 = st.columns([3, 1])
            with col_h1:
                st.subheader(f"{content.title}")
            with col_h2:
                difficulty_display = "⭐" * content.difficulty
                st.markdown(f"**Difficulty:** {difficulty_display}")
            
            # Adaptive explanation depth
            if recommendation['explanation_depth'] > 0.7:
                st.info("**Detailed Explanation**")
                st.markdown(content.content)
                
                # Additional examples with high explanation depth
                if content.exercises:
                    with st.expander("View Examples", expanded=True):
                        for i, exercise in enumerate(content.exercises[:2]):
                            st.markdown(f"**Example {i+1}:**")
                            st.code(exercise, language="javascript")
            else:
                st.info("**Compact Overview**")
                # Show shortened version
                short_content = content.content[:min(len(content.content)//2, 500)] + "..."
                st.markdown(short_content)
                
                if st.button("Show More Details"):
                    recommendation['explanation_depth'] = 1.0
                    st.rerun()
            
            # Break recommendation
            if recommendation['break_suggestion']:
                st.warning("☕ **Break Recommendation:** You've been learning for a while. A short break might be helpful!")
            
            # Interactive elements based on RL recommendation
            if recommendation['interaction_level'] > 0.5:
                st.subheader("Interactive Exercise")
                
                # Exercise selection based on difficulty
                exercise_index = int(recommendation['difficulty_adjustment'] + 0.5)
                exercise_index = max(0, min(exercise_index, len(content.exercises) - 1))
                
                if content.exercises:
                    current_exercise = content.exercises[exercise_index]
                    
                    # Task description
                    st.markdown("**Your Task:**")
                    st.markdown(f"Complete the following code:")
                    
                    # Hint system
                    if recommendation['provide_hint']:
                        with st.expander("Need a hint?"):
                            st.markdown(content.hints[0] if content.hints else "Think about the basic concepts!")
                            st.session_state.session_stats['hints_used'] += 1
                    
                    # Code editor
                    user_code = st.text_area(
                        "Write your code:",
                        value=current_exercise,
                        height=200,
                        key=f"code_editor_{content.id}"
                    )
                    
                    col_a, col_b, col_c = st.columns([1, 1, 2])
                    
                    with col_a:
                        if st.button("Check Solution", type="primary"):
                            # Calculate time
                            time_spent = (time.time() - st.session_state.exercise_start_time) / 60
                            
                            # Simulate evaluation (in real app, code evaluation would happen here)
                            accuracy = np.random.uniform(0.6, 1.0)  # Placeholder
                            
                            # Update Progress
                            student_response = {
                                'completed': 1,
                                'accuracy': accuracy,
                                'time_spent': time_spent,
                                'help_used': 1 if st.session_state.session_stats['hints_used'] > 0 else 0,
                                'engagement': 0.8
                            }
                            
                            reward, info = st.session_state.rl_tutor.update_learning_progress(student_response)
                            
                            # Update Stats
                            st.session_state.session_stats['exercises_completed'] += 1
                            st.session_state.session_stats['total_score'] += accuracy * 100
                            st.session_state.session_stats['time_spent'] += time_spent
                            
                            # Feedback
                            if accuracy > 0.9:
                                st.success(f"Excellent! Accuracy: {accuracy:.0%}")
                            elif accuracy > 0.7:
                                st.info(f"Well done! Accuracy: {accuracy:.0%}")
                            else:
                                st.warning(f"Keep practicing! Accuracy: {accuracy:.0%}")
                            
                            # Reset for next exercise
                            st.session_state.current_exercise = None
                            time.sleep(2)
                            st.rerun()
                    
                    with col_b:
                        if st.button("⏭Skip"):
                            st.session_state.current_exercise = None
                            st.rerun()
                    
                    with col_c:
                        # Show adaptive feedback
                        if recommendation['adaptive_feedback'] > 0.7:
                            encouragement = ["Great approach!", 
                                           "You're making great progress!",
                                           "Your approach is very good!"]
                            st.info(f"{encouragement[recommendation['encouragement_type']]}")
            
            else:
                # Passive learning materials for low interaction levels
                st.subheader("Learning Materials")
                
                # Video recommendations
                if hasattr(content, 'video_url') and content.video_url:
                    st.video(content.video_url)
                
                # Additional resources
                with st.expander("Additional Resources"):
                    st.markdown("""
                    - [MDN Web Docs](https://developer.mozilla.org/)
                    - [JavaScript.info](https://javascript.info/)
                    - [React Documentation](https://react.dev/)
                    """)
                
                if st.button("Ready for an Exercise?"):
                    recommendation['interaction_level'] = 1.0
                    st.rerun()
    
    with col2:
        st.header("🎯 Learning Progress")
        
        # Progress visualization
        progress_data = {
            'Knowledge': st.session_state.user_data.get('knowledge_levels', {}).get(learning_goal, 0.5),
            'Engagement': 0.8,
            'Speed': min(st.session_state.session_stats['exercises_completed'] / 10, 1.0),
            'Accuracy': st.session_state.session_stats['total_score'] / max(st.session_state.session_stats['exercises_completed'] * 100, 1)
        }
        
        # Radar chart for progress
        fig = go.Figure()
        
        categories = list(progress_data.keys())
        values = list(progress_data.values())
        values.append(values[0])  # Close the polygon
        categories.append(categories[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Status',
            fillcolor='rgba(135, 206, 250, 0.3)',
            line=dict(color='rgba(135, 206, 250, 1)')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Recommendations")
        
        # RL-based recommendations
        if st.session_state.model_loaded:
            if st.session_state.session_stats['exercises_completed'] > 2:
                avg_accuracy = st.session_state.session_stats['total_score'] / (st.session_state.session_stats['exercises_completed'] * 100)
                
                if avg_accuracy < 0.6:
                    st.info("📉 Consider easier exercises")
                elif avg_accuracy > 0.9:
                    st.info("📈 You're ready for harder tasks!")
                else:
                    st.info("The difficulty fits your level well")
            
            # Time-based recommendations
            if st.session_state.session_stats['time_spent'] > 30:
                st.warning("Consider taking a break!")
        
        # Achievements
        st.subheader("Achievements")
        achievements = []
        
        if st.session_state.session_stats['exercises_completed'] >= 5:
            achievements.append("5 exercises mastered!")
        if st.session_state.session_stats['total_score'] >= 400:
            achievements.append("400 points reached!")
        if st.session_state.session_stats['hints_used'] == 0 and st.session_state.session_stats['exercises_completed'] > 0:
            achievements.append("Solved without hints!")
        
        if achievements:
            for achievement in achievements:
                st.success(achievement)
        else:
            st.info("Complete exercises to unlock achievements!")
        
        # End session
        st.divider()
        if st.button("End Session & Show Statistics"):
            render_session_summary()


def render_session_summary():
    """Shows a summary of the learning session"""
    
    st.balloons()
    
    # Modal-like container
    with st.container():
        st.header("Session Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Exercises Completed",
                st.session_state.session_stats['exercises_completed'],
                "✅"
            )
        
        with col2:
            avg_score = st.session_state.session_stats['total_score'] / max(st.session_state.session_stats['exercises_completed'], 1)
            st.metric(
                "Average Score",
                f"{avg_score:.0f}%",
                "📈" if avg_score > 70 else "📉"
            )
        
        with col3:
            st.metric(
                "Time Invested",
                f"{st.session_state.session_stats['time_spent']:.0f} Min",
                "⏱️"
            )
        
        # Detailed analysis
        st.subheader("🔍 Detailed Analysis")
        
        # Learning efficiency
        efficiency = (st.session_state.session_stats['total_score'] / 100) / max(st.session_state.session_stats['time_spent'], 1)
        
        st.progress(min(efficiency, 1.0))
        st.caption(f"Learning Efficiency: {efficiency:.2f} points per minute")
        
        # Recommendations for next session
        st.subheader("🎯 Recommendations for Next Session")
        
        recommendations = []
        
        if avg_score < 60:
            recommendations.append("• Review the basics")
            recommendations.append("• Use more hints")
        elif avg_score > 85:
            recommendations.append("• Increase difficulty")
            recommendations.append("• Try new topics")
        
        if st.session_state.session_stats['time_spent'] < 20:
            recommendations.append("• Plan longer learning sessions")
        elif st.session_state.session_stats['time_spent'] > 60:
            recommendations.append("• Take more frequent breaks")
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Save session data
        if 'performance_history' not in st.session_state.user_data:
            st.session_state.user_data['performance_history'] = []
        
        st.session_state.user_data['performance_history'].append({
            'date': datetime.now().isoformat(),
            'stats': st.session_state.session_stats.copy()
        })
        
        # Reset for new session
        if st.button("🔄 Start New Session", type="primary"):
            st.session_state.session_stats = {
                'exercises_completed': 0,
                'total_score': 0,
                'time_spent': 0,
                'hints_used': 0
            }
            st.session_state.current_exercise = None
            st.rerun()


# Helper functions for integration

def get_difficulty_mapping() -> Dict[str, int]:
    """Maps difficulty preferences to numerical values"""
    return {
        "Very Easy": 1,
        "Easy": 2,
        "Medium": 3,
        "Hard": 4,
        "Very Hard": 5
    }


def calculate_engagement_score(session_stats: Dict) -> float:
    """Calculates engagement score based on session statistics"""
    if session_stats['exercises_completed'] == 0:
        return 0.5
    
    # Factors for engagement
    completion_rate = min(session_stats['exercises_completed'] / 10, 1.0)
    accuracy_rate = session_stats['total_score'] / max(session_stats['exercises_completed'] * 100, 1)
    time_factor = min(session_stats['time_spent'] / 30, 1.0)  # 30 min as reference
    
    # Weighted average
    engagement = (completion_rate * 0.3 + accuracy_rate * 0.4 + time_factor * 0.3)
    
    return float(np.clip(engagement, 0, 1))


def format_learning_path(content_list: List[Any]) -> str:
    """Formats a list of learning contents as a learning path"""
    path_str = "### Your Personalized Learning Path:\n\n"
    
    for i, content in enumerate(content_list[:5]):  # Show max 5 next steps
        difficulty_stars = "⭐" * getattr(content, 'difficulty', 3)
        path_str += f"{i+1}. **{getattr(content, 'title', 'Unknown')}** {difficulty_stars}\n"
        path_str += f"   - Estimated Time: {getattr(content, 'estimated_time', 15)} Min\n"
        path_str += f"   - Type: {getattr(content, 'content_type', 'Exercise')}\n\n"
    
    return path_str


# Export function for main app
if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="Adaptive Tutor",
        page_icon="🤖",
        layout="wide"
    )
    
    # Mock user data for testing
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {
            'learning_style': {
                'visual': {'score': 0.8, 'confidence': 0.9, 'interpretation': 'Stark visuell'},
                'verbal': {'score': 0.3, 'confidence': 0.7, 'interpretation': 'Wenig verbal'},
                'active': {'score': 0.7, 'confidence': 0.8, 'interpretation': 'Aktiv'},
                'reflective': {'score': 0.4, 'confidence': 0.6, 'interpretation': 'Moderat reflektiv'}
            },
            'knowledge_levels': {
                'JavaScript Basics': 0.6,
                'React Components': 0.4,
                'API Integration': 0.3
            }
        }
    
    render_adaptive_tutor_page()