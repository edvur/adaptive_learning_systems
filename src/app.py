# app.py - Main application for Adaptive Learning System
"""
Adaptive Learning System with Learning Style Classification and DQN-based Tutoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import time
from streamlit_ml_integration import StreamlitModelLoader, StreamlitLearningStylePredictor
from learning_style_test import (
    init_database, 
    render_learning_style_test, 
    render_study_dashboard,  
)
from adaptive_tutor import render_adaptive_tutor_page


# Initialize database
# This function should be defined in learning_style_test.py
init_database()

# Load ML models
@st.cache_resource
def load_ml_models():
    try:
        models = StreamlitModelLoader.load_learning_style_models("models")
        if models and len(models) >= 4:  # feature_pipeline + 3 models
            return StreamlitLearningStylePredictor(models)
        else:
            st.warning("Some models missing")
            return None
    except Exception as e:
        st.error(f"Could not load ML models: {e}")
        return None

def calculate_learning_streak(session_history):
    """Calculate consecutive days of learning"""
    if not session_history:
        return 0
    
    dates = [s.get('date', datetime.now().date()) for s in session_history]
    dates = sorted(set(dates), reverse=True)
    
    streak = 1
    for i in range(1, len(dates)):
        if (dates[i-1] - dates[i]).days == 1:
            streak += 1
        else:
            break
    
    return streak

def check_achievements(user_data):
    """Check which achievements are unlocked"""
    achievements = {
        "First Steps": user_data.get('learning_style') is not None,
        "Getting Started": user_data.get('total_sessions', 0) >= 1,
        "Regular Learner": user_data.get('total_sessions', 0) >= 5,
        "Dedicated Student": user_data.get('total_sessions', 0) >= 10,
        "High Performer": user_data.get('avg_performance', 0) >= 0.8,
        "Perfectionist": user_data.get('avg_performance', 0) >= 0.95,
        "Knowledge Seeker": len(user_data.get('knowledge_levels', {})) >= 3,
        "Master Learner": len(user_data.get('knowledge_levels', {})) >= 5,
        "Consistency King": calculate_learning_streak(user_data.get('session_history', [])) >= 7
    }
    
    return achievements

def check_learning_style_status():
    """Debug function to check status"""
    if 'user_data' in st.session_state:
        if 'learning_style' in st.session_state.user_data:
            if st.session_state.user_data['learning_style'] is not None:
                return True, "Learning Style available"
            else:
                return False, "Learning Style is None"
        else:
            return False, "No learning_style key in user_data"
    else:
        return False, "No user_data in session_state"

# Page configuration
st.set_page_config(
    page_title="Adaptive Learning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better design
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'student_id': f'student_{np.random.randint(1000, 9999)}',
        'learning_style': None,
        'current_session': None,
        'session_history': [],
        'total_sessions': 0,
        'avg_performance': 0.0
    }

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.predictor = None
    st.session_state.dqn_agent = None
    st.session_state.content_library = None

if 'predictor' not in st.session_state or st.session_state.predictor is None:
    st.session_state.predictor = load_ml_models()

# Sidebar for navigation
with st.sidebar:
    #st.image("/Users/edavurmaz/Uni/Bachelorarbeit/adaptive_tutoring_system/src/assets/347215-sepik.jpg", use_column_width=True)
    st.markdown("---")
    
    st.markdown(f"**Student ID:** `{st.session_state.user_data['student_id']}`")
    
    if st.session_state.user_data['learning_style']:
        st.success("Learning Style Analyzed")

        with st.expander("Your Learning Style", expanded=False):
            ls = st.session_state.user_data['learning_style']
            for dim, data in ls.items():
                st.caption(f"{dim}: **{data['interpretation']}**")
    else:
        st.info("ℹ️ Please complete learning style analysis")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Home", "Learning Style Analysis", "Tutoring Session", "Progress & Analytics", "ℹ️ About"]
    )

# Main content based on navigation
if page == "Home":

    # if st.session_state.predictor:
    #     st.success("✅ ML Models loaded successfully!")
    #     st.write("Models available:", st.session_state.predictor.dimension_models.keys())
    # else:
    #     st.warning("⚠️ ML Models not loaded - using demo mode")

    st.title("Welcome to the Adaptive Learning System")
    st.markdown("### Personalized Learning Based on Your Learning Style")
    
    
    # Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "✅ Complete" if st.session_state.user_data['learning_style'] else "⏳ Pending"
        st.metric("Learning Style", status)
    
    with col2:
        sessions = st.session_state.user_data.get('total_sessions', 0)
        st.metric("Sessions Completed", sessions)
    
    with col3:
        avg_perf = st.session_state.user_data.get('avg_performance', 0)
        st.metric("Avg Performance", f"{avg_perf:.1%}")
    
    with col4:
        # Check if RL model is available
        model_status = "✅ Ready" if Path("improved_models_standard/best_model.pth").exists() else "⚠️ Not Found"
        st.metric("AI Tutor", model_status)
    
    st.markdown("---")
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3> Learning Style Analysis</h3>
        <p>Discover your unique learning style across three dimensions:</p>
        <ul>
        <li><b>Perception</b>: Sensing vs Intuitive</li>
        <li><b>Input</b>: Visual vs Verbal</li>
        <li><b>Understanding</b>: Sequential vs Global</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3> AI-Powered Tutoring</h3>
        <p>Get personalized content recommendations based on:</p>
        <ul>
        <li>Your learning style</li>
        <li>Current knowledge level</li>
        <li>Real-time performance</li>
        <li>Engagement patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3> Track Your Progress</h3>
        <p>Monitor your learning journey with:</p>
        <ul>
        <li>Performance analytics</li>
        <li>Knowledge growth</li>
        <li>Personalized recommendations</li>
        <li>Achievement tracking</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    if st.session_state.user_data['learning_style']:
        st.subheader("Your Learning Profile")
        ls = st.session_state.user_data['learning_style']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Perception", ls['Perception']['interpretation'], 
                     f"Confidence: {ls['Perception']['confidence']:.1%}")
        with col2:
            st.metric("Input", ls['Input']['interpretation'],
                     f"Confidence: {ls['Input']['confidence']:.1%}")
        with col3:
            st.metric("Understanding", ls['Understanding']['interpretation'],
                     f"Confidence: {ls['Understanding']['confidence']:.1%}")
    else:
        st.info(" Start by completing your Learning Style Analysis")
        if st.button("Go to Analysis", key="home_analysis"):
            st.experimental_rerun()

elif page == "Learning Style Analysis":
    st.title("Learning Style Analysis")
    #st.write("Debug: Page loaded successfully")
    st.markdown("Complete the activity assessment to determine your learning style")
    
    # Load models if not already loaded
    if not st.session_state.models_loaded:
        with st.spinner("Loading ML models..."):
    #         # Here you would load your real models
    #         # from your_modules import LearningStylePredictor
    #         # st.session_state.predictor = LearningStylePredictor()
              st.session_state.models_loaded = True
              time.sleep(1)  # Simulate loading time
    
    #st.write("Debug: Page loaded successfully")
    
    try:
        # Import check
        #st.write("Debug: Checking imports...")
        from learning_style_test import (
            render_learning_style_test,
            render_study_dashboard
        )
        #st.success("✅ Imports successful")

        # Tabs for different input methods
        tab1, tab2, tab3 = st.tabs(["Cognitive Assessment", "Manual Acitivty Input", "Study Dashboard"])
        
        with tab1:
            #st.write("Debug: Tab 1 loaded")
            try:
                render_learning_style_test()
            except Exception as e:
                st.error(f"Error in render_learning_style_test: {str(e)}")
                st.exception(e)

        with tab2:
            #st.write("Debug: Tab 2 loaded")
            st.markdown("### Enter Your Learning Activities")
            st.markdown("Rate your engagement with different learning activities (0-100)")
            
            with st.form("activity_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Content Interaction")
                    course_overview = st.slider("Course Overview", 0, 100, 5, 
                                            help="How often do you check course overviews?")
                    reading_file = st.slider("Reading Files", 0, 100, 10,
                                        help="Time spent reading text materials")
                    abstract_material = st.slider("Abstract Materials", 0, 100, 3,
                                                help="Engagement with theoretical content")
                    concrete_material = st.slider("Concrete Materials", 0, 100, 7,
                                                help="Working with practical examples")
                
                with col2:
                    st.markdown("#### Visual & Interactive")
                    visual_materials = st.slider("Visual Materials", 0, 100, 8,
                                            help="Using diagrams, charts, videos")
                    self_assessment = st.slider("Self-Assessment", 0, 100, 2,
                                            help="Taking self-evaluation tests")
                    exercises_submit = st.slider("Exercises Submitted", 0, 100, 5,
                                            help="Completing practice exercises")
                    quiz_submitted = st.slider("Quizzes Submitted", 0, 100, 3,
                                            help="Taking graded quizzes")
                
                with col3:
                    st.markdown("#### Video Engagement")
                    playing = st.slider("Videos Played", 0, 100, 15,
                                    help="Time watching educational videos")
                    paused = st.slider("Videos Paused", 0, 100, 4,
                                    help="Frequency of pausing videos")
                    unstarted = st.slider("Videos Unstarted", 0, 100, 2,
                                        help="Videos added but not watched")
                    buffering = st.slider("Buffering Events", 0, 100, 1,
                                        help="Technical interruptions")
                
                submit_button = st.form_submit_button("Analyze My Learning Style", 
                                                    use_container_width=True)
                
                if submit_button:
                    # Create activity DataFrame
                    activities = pd.DataFrame([{
                        'Course overview': course_overview,
                        'Reading file': reading_file,
                        'Abstract materiale': abstract_material,
                        'Concrete material': concrete_material,
                        'Visual Materials': visual_materials,
                        'Self-assessment': self_assessment,
                        'Exercises submit': exercises_submit,
                        'Quiz submitted': quiz_submitted,
                        'playing': playing,
                        'paused': paused,
                        'unstarted': unstarted,
                        'buffering': buffering
                    }])
                    
                    with st.spinner("Analyzing your learning style..."):
                        # Simulate ML prediction
                        # In the real app: predictions = st.session_state.predictor.predict(activities)
                        time.sleep(2)
                        
                        # Simulated results
                        predictions = {
                            'Perception': {
                                'predicted_class': 1 if concrete_material > abstract_material else 0,
                                'interpretation': 'Sensing' if concrete_material > abstract_material else 'Intuitive',
                                'confidence': 0.75 + np.random.random() * 0.2,
                                'probabilities': {
                                    'Sensing': 0.6 if concrete_material > abstract_material else 0.4,
                                    'Intuitive': 0.4 if concrete_material > abstract_material else 0.6
                                }
                            },
                            'Input': {
                                'predicted_class': 1 if visual_materials > reading_file else 0,
                                'interpretation': 'Visual' if visual_materials > reading_file else 'Verbal',
                                'confidence': 0.7 + np.random.random() * 0.25,
                                'probabilities': {
                                    'Visual': 0.7 if visual_materials > reading_file else 0.3,
                                    'Verbal': 0.3 if visual_materials > reading_file else 0.7
                                }
                            },
                            'Understanding': {
                                'predicted_class': 1 if course_overview > 10 else 0,
                                'interpretation': 'Sequential' if course_overview > 10 else 'Global',
                                'confidence': 0.65 + np.random.random() * 0.3,
                                'probabilities': {
                                    'Sequential': 0.65 if course_overview > 10 else 0.35,
                                    'Global': 0.35 if course_overview > 10 else 0.65
                                }
                            }
                        }
                        
                        # Save to session state
                        st.session_state.user_data['learning_style'] = predictions
                        st.session_state.user_data['activities'] = activities.to_dict('records')[0]
                    
                    st.success("Learning Style Analysis Complete!")
                    
                    # Show results
                    st.markdown("### Your Learning Style Profile")
                    
                    for dimension, result in predictions.items():
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Gauge chart for confidence
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = result['confidence'] * 100,
                                title = {'text': f"{dimension}<br><b>{result['interpretation']}</b>"},
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkgreen"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"}],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90}
                                }
                            ))
                            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"#### {dimension} Dimension")
                            
                            # Probability Bar Chart
                            probs_df = pd.DataFrame([result['probabilities']])
                            fig = px.bar(probs_df.T, orientation='h',
                                        labels={'index': 'Style', 'value': 'Probability'},
                                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
                            fig.update_layout(height=200, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Interpretation
                            if dimension == 'Perception':
                                if result['interpretation'] == 'Sensing':
                                    st.info("You prefer concrete, practical information and hands-on experiences")
                                else:
                                    st.info("You prefer theoretical concepts and abstract thinking")
                            elif dimension == 'Input':
                                if result['interpretation'] == 'Visual':
                                    st.info("You learn best through visual representations and demonstrations")
                                else:
                                    st.info("You learn best through written and spoken explanations")
                            else:  # Understanding
                                if result['interpretation'] == 'Sequential':
                                    st.info("You prefer step-by-step, linear learning progression")
                                else:
                                    st.info("You prefer seeing the big picture and making connections")
                    
                    # Recommendations
                    st.markdown("### Personalized Learning Recommendations")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Recommended Content Types")
                        if predictions['Input']['interpretation'] == 'Visual':
                            st.write("- Video tutorials and demonstrations")
                            st.write("- Interactive diagrams and infographics")
                            st.write("- Visual mind maps and flowcharts")
                        else:
                            st.write("- Detailed text explanations")
                            st.write("- Audio lectures and podcasts")
                            st.write("- Written exercises and essays")
                    
                    with col2:
                        st.markdown("#### Learning Strategies")
                        if predictions['Perception']['interpretation'] == 'Sensing':
                            st.write("- Practice with real-world examples")
                            st.write("- Follow step-by-step tutorials")
                            st.write("- Focus on practical applications")
                        else:
                            st.write("- Explore theoretical frameworks")
                            st.write("- Look for patterns and connections")
                            st.write("- Try innovative problem-solving")
        
        with tab3:
            st.markdown("### Study Dashboard")
            st.write("Debug: Tab 3 loaded")
            try:
                render_study_dashboard()
            except Exception as e:
                st.error(f"Error in render_study_dashboard: {str(e)}")
                st.exception(e)
        
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        st.exception(e)
        
        # Fallback content
        st.markdown("### Fallback: Manual Input Only")
        st.warning("The cognitive test module couldn't be loaded. Using manual input only.")

        # uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        # if uploaded_file is not None:
        #     df = pd.read_csv(uploaded_file)
        #     st.write("Data preview:")
        #     st.dataframe(df.head())
            
        #     if st.button("Analyze Uploaded Data"):
        #         st.info("This feature will be implemented to process batch data")


elif page == "Tutoring Session":
    st.title("Adaptive Tutoring Session")

    status, message = check_learning_style_status()
    if not status:
        st.error(f"❌ {message}")
        st.warning("⚠️ Please complete the Learning Style Analysis first!")
        
        # Debug display
        with st.expander("Debug Info"):
            st.write("Session State user_data:", st.session_state.get('user_data', 'Not found'))
        
        # Buttons to continue
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Learning Style Analysis", type="primary"):
                st.session_state.page = "Learning Style Analysis"
                st.rerun()
        
        with col2:
            if st.button("Use Demo Learning Style"):
                # Demo data for quick testing
                st.session_state.user_data['learning_style'] = {
                    'Perception': {
                        'interpretation': 'Sensing',
                        'confidence': 0.75,
                        'predicted_class': 1,
                        'probabilities': {'Sensing': 0.75, 'Intuitive': 0.25}
                    },
                    'Input': {
                        'interpretation': 'Visual',
                        'confidence': 0.82,
                        'predicted_class': 1,
                        'probabilities': {'Visual': 0.82, 'Verbal': 0.18}
                    },
                    'Understanding': {
                        'interpretation': 'Sequential',
                        'confidence': 0.68,
                        'predicted_class': 1,
                        'probabilities': {'Sequential': 0.68, 'Global': 0.32}
                    }
                }
                st.rerun()
    else:
        # Learning style available - show tutor
        render_adaptive_tutor_page()
    
    # if not st.session_state.user_data['learning_style']:
    #     st.warning("⚠️ Please complete the Learning Style Analysis first")
        
    #     col1, col2 = st.columns([1, 3])
    #     with col1:
    #         if st.button("Go to Analysis", type="primary"):
    #             st.session_state.navigation = "📊 Learning Style Analysis"
    #             st.rerun()
        
    #     with col2:
    #         if st.button("Use Demo Mode"):
    #             # Demo Learning Style for testing
    #             st.session_state.user_data['learning_style'] = {
    #                 'Perception': {
    #                     'interpretation': 'Sensing',
    #                     'confidence': 0.75,
    #                     'score': 0.7
    #                 },
    #                 'Input': {
    #                     'interpretation': 'Visual',
    #                     'confidence': 0.82,
    #                     'score': 0.8
    #                 },
    #                 'Understanding': {
    #                     'interpretation': 'Sequential',
    #                     'confidence': 0.68,
    #                     'score': 0.6
    #                 }
    #             }
    #             st.rerun()
    # else:
    #     # Render Adaptive Tutor Page
    #     render_adaptive_tutor_page()

elif page == "Progress & Analytics":
    st.title("Progress & Analytics")
    
    if st.session_state.user_data['learning_style']:
        
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Performance", "Learning Path", "Achievements"])
        
        st.warning("⚠️ No data available. Complete some learning sessions first.")
        with tab1:
            st.header("Learning Overview")
            
            # Summary Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_time = sum([s.get('duration', 0) for s in st.session_state.user_data.get('session_history', [])])
                st.metric("Total Learning Time", f"{total_time:.0f} min")
            
            with col2:
                exercises = sum([s.get('exercises_completed', 0) for s in st.session_state.user_data.get('session_history', [])])
                st.metric("Exercises Completed", exercises)
            
            with col3:
                accuracy = np.mean([s.get('accuracy', 0) for s in st.session_state.user_data.get('session_history', [])]) if st.session_state.user_data.get('session_history') else 0
                st.metric("Average Accuracy", f"{accuracy:.1%}")
            
            with col4:
                streak = calculate_learning_streak(st.session_state.user_data.get('session_history', []))
                st.metric("Learning Streak", f"{streak} days")
        
        with tab2:
            st.header("Performance Analysis")
            
            # Performance over time
            if st.session_state.user_data.get('performance_history'):
                df = pd.DataFrame(st.session_state.user_data['performance_history'])
                
                fig = px.line(df, x=df.index, y='score', 
                            title='Performance Trend',
                            labels={'score': 'Performance Score', 'index': 'Session'})
                fig.add_hline(y=0.7, line_dash="dash", 
                            annotation_text="Target Performance")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Complete some learning sessions to see your performance trend")
        
        with tab3:
            st.header("Your Learning Path")
            
            # Knowledge levels visualization
            if st.session_state.user_data.get('knowledge_levels'):
                knowledge_df = pd.DataFrame(
                    list(st.session_state.user_data['knowledge_levels'].items()),
                    columns=['Topic', 'Level']
                )
                
                fig = px.bar(knowledge_df, x='Level', y='Topic', 
                           orientation='h',
                           title='Knowledge Levels by Topic',
                           color='Level',
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Start learning to track your knowledge growth")
        
        with tab4:
            st.header("🏆 Achievements")
            
            # Achievement system
            achievements = check_achievements(st.session_state.user_data)
            
            cols = st.columns(3)
            for i, (achievement, unlocked) in enumerate(achievements.items()):
                with cols[i % 3]:
                    if unlocked:
                        st.success(f"🏆 {achievement}")
                    else:
                        st.info(f"🔒 {achievement}")
    
    else:
        st.info("Complete the Learning Style Analysis to see your progress")

elif page == "ℹ️ About":
    st.title("ℹ️ About This System")
    
    st.markdown("""
    ### Adaptive Learning System with AI-Powered Tutoring
    
    This system combines advanced machine learning techniques to provide personalized education:
    
    This adaptive learning system combines:
    
    1. **Learning Style Analysis**: Based on cognitive tasks and ML models
    2. **Deep Reinforcement Learning**: For personalized content recommendations
    3. **Real-time Adaptation**: Adjusts to your performance and engagement
    
    ### Research Background
    
    This system is part of ongoing research in personalized education technology.
    
    ### Privacy & Data
    
    - All data is stored locally
    - No personal information is shared
    - You can export or delete your data anytime
    
   
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
    Adaptive Learning System v1.0 | Made with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)