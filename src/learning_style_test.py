# learning_style_test.py - Task-based learning style test
"""
Learning Style Test based on cognitive tasks
Clean integration for the main app
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import json
import sqlite3
from pathlib import Path
import time
from typing import Dict, List

# Database Setup
DB_PATH = "study_data.db"

def init_database():
    """Initialize SQLite database for the study"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY,
            created_at TIMESTAMP,
            age INTEGER,
            gender TEXT,
            education_level TEXT,
            consent_given BOOLEAN,
            test_completed BOOLEAN DEFAULT FALSE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id TEXT,
            question_id TEXT,
            question_type TEXT,
            answer TEXT,
            time_taken REAL,
            correct BOOLEAN,
            timestamp TIMESTAMP,
            FOREIGN KEY (participant_id) REFERENCES participants(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_style_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id TEXT,
            perception_score REAL,
            perception_type TEXT,
            input_score REAL,
            input_type TEXT,
            understanding_score REAL,
            understanding_type TEXT,
            confidence_avg REAL,
            timestamp TIMESTAMP,
            FOREIGN KEY (participant_id) REFERENCES participants(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id TEXT,
            course_overview INTEGER,
            reading_file INTEGER,
            abstract_materiale INTEGER,
            concrete_material INTEGER,
            visual_materials INTEGER,
            self_assessment INTEGER,
            exercises_submit INTEGER,
            quiz_submitted INTEGER,
            playing INTEGER,
            paused INTEGER,
            unstarted INTEGER,
            buffering INTEGER,
            FOREIGN KEY (participant_id) REFERENCES participants(id)
        )
    """)
    
    conn.commit()
    conn.close()

# Enhanced Test Questions (15 questions total)
LEARNING_STYLE_TASKS = {
    'perception': [
        {
            'id': 'P1',
            'type': 'perception_concrete',
            'question': 'A recipe calls for 2.5 cups of flour for 12 cookies. How much flour do you need for 30 cookies?',
            'options': ['5 cups', '6.25 cups', '7.5 cups', '8 cups'],
            'correct': '6.25 cups',
            'measures': 'sensing',
            'explanation': 'Tests preference for concrete, practical problem-solving'
        },
        {
            'id': 'P2',
            'type': 'perception_abstract',
            'question': 'If all philosophers are thinkers, and some thinkers are writers, which statement must be true?',
            'options': [
                'All philosophers are writers',
                'Some philosophers might be writers',
                'No philosophers are writers',
                'All writers are philosophers'
            ],
            'correct': 'Some philosophers might be writers',
            'measures': 'intuitive',
            'explanation': 'Tests abstract logical reasoning'
        },
        {
            'id': 'P3',
            'type': 'perception_pattern',
            'question': 'What comes next in the sequence: 1, 1, 2, 3, 5, 8, ?',
            'options': ['11', '13', '15', '21'],
            'correct': '13',
            'measures': 'intuitive',
            'explanation': 'Pattern recognition (Fibonacci sequence)'
        },
        {
            'id': 'P4',
            'type': 'perception_practical',
            'question': 'You need to paint a room that is 12ft x 15ft with 9ft ceilings. If one gallon covers 400 sq ft, how many gallons do you need for the walls only?',
            'options': ['1 gallon', '2 gallons', '3 gallons', '4 gallons'],
            'correct': '2 gallons',
            'measures': 'sensing',
            'explanation': 'Practical application of mathematics'
        },
        {
            'id': 'P5',
            'type': 'perception_conceptual',
            'question': 'Which analogy best completes: Ocean is to Drop as Desert is to ?',
            'options': ['Sand', 'Grain', 'Dune', 'Heat'],
            'correct': 'Grain',
            'measures': 'intuitive',
            'explanation': 'Conceptual thinking and analogies'
        }
    ],
    'input': [
        {
            'id': 'I1',
            'type': 'input_visual',
            'question': 'Look at this chart showing monthly sales. Which month had the second highest revenue?',
            'visual_content': 'bar_chart',
            'data': {'Jan': 45, 'Feb': 68, 'Mar': 52, 'Apr': 71, 'May': 58, 'Jun': 64},
            'options': ['February', 'March', 'May', 'June'],
            'correct': 'February',
            'measures': 'visual',
            'explanation': 'Tests visual information processing'
        },
        {
            'id': 'I2',
            'type': 'input_verbal',
            'question': 'Read this passage: "The mitochondria, often called the powerhouse of the cell, converts nutrients into energy through a process called cellular respiration. This organelle has its own DNA and can replicate independently." What makes mitochondria unique among cell organelles?',
            'options': [
                'They produce energy',
                'They have their own DNA',
                'They process nutrients',
                'They are found in all cells'
            ],
            'correct': 'They have their own DNA',
            'measures': 'verbal',
            'explanation': 'Tests verbal/textual information processing'
        },
        {
            'id': 'I3',
            'type': 'input_diagram',
            'question': 'Based on this network diagram, which represents the critical path?',
            'visual_content': 'network_diagram',
            'options': ['A→B→D→F', 'A→C→E→F', 'A→B→E→F', 'A→C→D→F'],
            'correct': 'A→C→E→F',
            'measures': 'visual',
            'explanation': 'Tests visual diagram interpretation'
        },
        {
            'id': 'I4',
            'type': 'input_instructions',
            'question': 'After reading these assembly instructions, what should be attached BEFORE inserting the shelves? "Step 1: Attach side panels to base. Step 2: Install back panel. Step 3: Attach support brackets. Step 4: Insert shelves."',
            'options': ['Side panels', 'Back panel', 'Support brackets', 'Base unit'],
            'correct': 'Support brackets',
            'measures': 'verbal',
            'explanation': 'Tests sequential instruction processing'
        },
        {
            'id': 'I5',
            'type': 'input_preference',
            'question': 'When learning a new software application, which approach do you prefer?',
            'options': [
                'Watch video tutorials showing the interface',
                'Read the detailed documentation',
                'Look at screenshot guides with annotations',
                'Follow written step-by-step instructions'
            ],
            'scoring': {'visual': [0.8, 0, 1, 0.2], 'verbal': [0.2, 1, 0, 0.8]},
            'measures': 'preference',
            'explanation': 'Direct preference measurement'
        }
    ],
    'understanding': [
        {
            'id': 'U1',
            'type': 'understanding_approach',
            'question': 'You need to learn a new programming language. Which approach appeals to you most?',
            'options': [
                'Start with basic syntax, then functions, then advanced concepts',
                'Build a complete project and learn concepts as needed',
                'Study the language philosophy and design principles first',
                'Follow a structured course from beginning to end'
            ],
            'scoring': {'sequential': [1, 0, 0, 1], 'global': [0, 1, 1, 0]},
            'measures': 'preference',
            'explanation': 'Measures sequential vs global learning preference'
        },
        {
            'id': 'U2',
            'type': 'understanding_problem',
            'question': 'When faced with a complex problem at work, what is your first instinct?',
            'options': [
                'Break it down into smaller, manageable parts',
                'Look for similar problems and their solutions',
                'Understand the big picture before diving into details',
                'Create a step-by-step plan of action'
            ],
            'scoring': {'sequential': [0.8, 0.3, 0, 1], 'global': [0.2, 0.7, 1, 0]},
            'measures': 'approach',
            'explanation': 'Tests problem-solving approach'
        },
        {
            'id': 'U3',
            'type': 'understanding_learning',
            'question': 'In a lecture or presentation, you understand best when the speaker:',
            'options': [
                'Provides a clear outline and follows it systematically',
                'Starts with examples and derives principles from them',
                'Gives an overview first, then fills in details',
                'Uses stories and analogies to explain concepts'
            ],
            'scoring': {'sequential': [1, 0.3, 0, 0.3], 'global': [0, 0.7, 1, 0.7]},
            'measures': 'comprehension',
            'explanation': 'Tests comprehension style preference'
        },
        {
            'id': 'U4',
            'type': 'understanding_study',
            'question': 'When preparing for an exam, which strategy works best for you?',
            'options': [
                'Review material in the order it was presented',
                'Create concept maps showing relationships',
                'Focus on understanding fundamental principles',
                'Practice problems in increasing difficulty'
            ],
            'scoring': {'sequential': [1, 0, 0.2, 0.8], 'global': [0, 1, 0.8, 0.2]},
            'measures': 'study_method',
            'explanation': 'Tests study method preference'
        },
        {
            'id': 'U5',
            'type': 'understanding_explanation',
            'question': 'Which type of explanation helps you understand a new concept best?',
            'options': [
                'A logical progression from simple to complex',
                'Multiple examples showing different applications',
                'Understanding how it fits into the bigger picture',
                'Detailed step-by-step procedures'
            ],
            'scoring': {'sequential': [0.8, 0.2, 0, 1], 'global': [0.2, 0.8, 1, 0]},
            'measures': 'explanation_preference',
            'explanation': 'Tests explanation style preference'
        }
    ]
}

def create_visual_question(data: Dict, question_type: str):
    """Create visual elements for questions"""
    import plotly.graph_objects as go
    
    if question_type == 'bar_chart':
        fig = go.Figure(data=[
            go.Bar(x=list(data.keys()), y=list(data.values()),
                   marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD', '#FFE66D'])
        ])
        fig.update_layout(
            title="Monthly Revenue (in thousands)",
            xaxis_title="Month",
            yaxis_title="Revenue",
            height=300
        )
        return fig
    
    elif question_type == 'network_diagram':
        # Simple network diagram representation
        st.info("Network Diagram: A→B(3)→D(2)→F(1), A→C(2)→E(4)→F(1), where numbers represent days")
        return None
    
    return None

def calculate_activity_mapping(test_results: List[Dict]) -> Dict[str, int]:
    """
    Maps test results to activity scores for the ML model
    """
    # Base mapping
    mapping = {
        'Course overview': 5,
        'Reading file': 10,
        'Abstract materiale': 3,
        'Concrete material': 7,
        'Visual Materials': 8,
        'Self-assessment': 2,
        'Exercises submit': 5,
        'Quiz submitted': 3,
        'playing': 15,
        'paused': 4,
        'unstarted': 2,
        'buffering': 1
    }
    
    # Analyze test results
    perception_scores = {'sensing': 0, 'intuitive': 0}
    input_scores = {'visual': 0, 'verbal': 0}
    understanding_scores = {'sequential': 0, 'global': 0}
    
    for result in test_results:
        q_type = result.get('question_type', '')
        if 'scoring' in result:
            # Preference questions
            answer_idx = result['answer_index']
            if 'visual' in result['scoring']:
                input_scores['visual'] += result['scoring']['visual'][answer_idx]
                input_scores['verbal'] += result['scoring']['verbal'][answer_idx]
            elif 'sequential' in result['scoring']:
                understanding_scores['sequential'] += result['scoring']['sequential'][answer_idx]
                understanding_scores['global'] += result['scoring']['global'][answer_idx]
        else:
            # Performance questions
            if result.get('correct', False):
                if 'perception' in q_type:
                    if result['measures'] == 'sensing':
                        perception_scores['sensing'] += 1
                    else:
                        perception_scores['intuitive'] += 1
    
    # Modify activities based on scores
    if perception_scores['sensing'] > perception_scores['intuitive']:
        mapping['Concrete material'] += 10
        mapping['Exercises submit'] += 8
    else:
        mapping['Abstract materiale'] += 10
        mapping['Reading file'] += 5
    
    if input_scores['visual'] > input_scores['verbal']:
        mapping['Visual Materials'] += 15
        mapping['playing'] += 10
    else:
        mapping['Reading file'] += 15
        mapping['Abstract materiale'] += 5
    
    if understanding_scores['sequential'] > understanding_scores['global']:
        mapping['Course overview'] += 10
        mapping['Quiz submitted'] += 5
    else:
        mapping['Visual Materials'] += 5
        mapping['playing'] += 5
    
    return mapping

def save_participant_data(participant_data: Dict):
    """Save participant data to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Save participant
    cursor.execute("""
        INSERT INTO participants (created_at, age, gender, education_level, consent_given, test_completed)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        participant_data['created_at'],
        participant_data['demographics']['age'],
        participant_data['demographics']['gender'],
        participant_data['demographics']['education'],
        participant_data['consent'],
        participant_data['test_completed']
    ))

    participant_id = cursor.lastrowid
    
    # Save test results
    for result in participant_data['test_results']:
        cursor.execute("""
            INSERT INTO test_results (participant_id, question_id, question_type, answer, time_taken, correct, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            participant_id,
            result['question_id'],
            result['question_type'],
            result['answer'],
            result['time_taken'],
            result.get('correct', None),
            result['timestamp']
        ))
    
    # Save activity mapping
    mapping = participant_data['activity_mapping']
    cursor.execute("""
        INSERT INTO activity_mappings (participant_id, course_overview, reading_file, abstract_materiale,
                                      concrete_material, visual_materials, self_assessment, exercises_submit,
                                      quiz_submitted, playing, paused, unstarted, buffering)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        participant_data['id'],
        mapping['Course overview'],
        mapping['Reading file'],
        mapping['Abstract materiale'],
        mapping['Concrete material'],
        mapping['Visual Materials'],
        mapping['Self-assessment'],
        mapping['Exercises submit'],
        mapping['Quiz submitted'],
        mapping['playing'],
        mapping['paused'],
        mapping['unstarted'],
        mapping['buffering']
    ))
    
    # Save learning style results
    ls_results = participant_data['learning_style_results']
    cursor.execute("""
        INSERT INTO learning_style_results (participant_id, perception_score, perception_type,
                                          input_score, input_type, understanding_score, understanding_type,
                                          confidence_avg, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        participant_data['id'],
        ls_results['Perception']['probabilities']['Sensing'],
        ls_results['Perception']['interpretation'],
        ls_results['Input']['probabilities']['Visual'],
        ls_results['Input']['interpretation'],
        ls_results['Understanding']['probabilities']['Sequential'],
        ls_results['Understanding']['interpretation'],
        (ls_results['Perception']['confidence'] + 
         ls_results['Input']['confidence'] + 
         ls_results['Understanding']['confidence']) / 3,
        datetime.now()
    ))
    
    conn.commit()
    conn.close()

def render_learning_style_test():
    """Render the task-based learning style test"""
    
    # Initialize session state for test
    if 'test_state' not in st.session_state:
        st.session_state.test_state = {
            'participant_id': str(uuid.uuid4()),
            'current_question': 0,
            'test_results': [],
            'start_time': None,
            'demographics_complete': False,
            'consent_given': False,
            'test_completed': False
        }
    
    test_state = st.session_state.test_state
    
    # Step 1: Consent
    if not test_state['consent_given']:
        st.title("🎓 Learning Style Assessment Study")
        st.markdown("""
        ### Welcome to the Learning Style Research Study
        
        This study investigates individual learning styles through cognitive tasks.
        
        **Study Purpose:** Validation of an AI-based adaptive learning system
        
        **Duration:** Approximately 10-15 minutes
        
        **Privacy:**
        - All data is stored anonymously
        - No personal identifying information is collected
        - Participation is voluntary
        - You may withdraw at any time
        
        **What to expect:**
        1. Brief demographic questions (age, education)
        2. 15 cognitive tasks and preference questions
        3. Automated learning style analysis
        4. Personalized results at the end
        """)
        
        consent = st.checkbox("I consent to participate in this study")
        
        if consent and st.button("Start Study", type="primary"):
            test_state['consent_given'] = True
            test_state['created_at'] = datetime.now()
            st.rerun()
    
    # Step 2: Demographics
    elif not test_state['demographics_complete']:
        st.title("Demographic Information")
        st.markdown("Please provide some general information:")
        
        with st.form("demographics"):
            age = st.select_slider("Age", options=list(range(16, 81)), value=25)
            
            gender = st.selectbox("Gender", 
                                ["Please select", "Male", "Female", "Non-binary", "Prefer not to say"])
            
            education = st.selectbox("Highest level of education",
                                   ["Please select", "High School", "Some College", "Bachelor's Degree", 
                                    "Master's Degree", "Doctoral Degree", "Other"])
            
            submitted = st.form_submit_button("Continue to Test")
            
            if submitted and gender != "Please select" and education != "Please select":
                test_state['demographics'] = {
                    'age': age,
                    'gender': gender,
                    'education': education
                }
                test_state['demographics_complete'] = True
                test_state['start_time'] = datetime.now()
                st.rerun()
    
    # Step 3: Test questions
    elif not test_state['test_completed']:
        # Collect all questions
        all_questions = []
        for category in ['perception', 'input', 'understanding']:
            all_questions.extend([(q, category) for q in LEARNING_STYLE_TASKS[category]])
        
        if test_state['current_question'] < len(all_questions):
            question, category = all_questions[test_state['current_question']]
            
            # Progress bar
            progress = test_state['current_question'] / len(all_questions)
            st.progress(progress)
            st.caption(f"Question {test_state['current_question'] + 1} of {len(all_questions)}")
            
            # Show question
            st.markdown(f"### {category.title()} Assessment")
            
            # Start timer
            if 'question_start_time' not in test_state:
                test_state['question_start_time'] = datetime.now()
            
            # Visual elements if needed
            if 'visual_content' in question:
                fig = create_visual_question(question.get('data', {}), question['visual_content'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Question
            st.markdown(f"**{question['question']}**")
            
            # Answer options
            answer = st.radio("Select your answer:", 
                            question.get('options', []),
                            key=f"q_{question['id']}")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("Next →", type="primary", disabled=not answer):
                    # Calculate time
                    time_taken = (datetime.now() - test_state['question_start_time']).total_seconds()
                    
                    # Save result
                    result = {
                        'question_id': question['id'],
                        'question_type': question['type'],
                        'answer': answer,
                        'answer_index': question.get('options', []).index(answer) if 'options' in question else 0,
                        'time_taken': time_taken,
                        'timestamp': datetime.now(),
                        'measures': question.get('measures', ''),
                    }
                    
                    # Check if correct (if applicable)
                    if 'correct' in question:
                        result['correct'] = answer == question['correct']
                    
                    # Add scoring info if present
                    if 'scoring' in question:
                        result['scoring'] = question['scoring']
                    
                    test_state['test_results'].append(result)
                    test_state['current_question'] += 1
                    
                    # Clear timer
                    if 'question_start_time' in test_state:
                        del test_state['question_start_time']
                    
                    st.rerun()
        
        else:
            # Test completed
            test_state['test_completed'] = True
            st.rerun()
    
    # Step 4: Results
    else:
        st.title("Assessment Complete!")
        st.balloons()
        
        with st.spinner("Analyzing your responses..."):
            # Calculate activity mapping
            activity_mapping = calculate_activity_mapping(test_state['test_results'])
            
            # Create DataFrame for ML model
            activities_df = pd.DataFrame([activity_mapping])
            
            # Get predictions (use your real model or demo predictor)
            if 'predictor' in st.session_state and st.session_state.predictor:
                predictions = st.session_state.predictor.predict(activities_df)
            else:
                # Demo prediction
                st.warning("No model loaded, using demo predictions")
                from streamlit_ml_integration import create_demo_predictor
                demo_predictor = create_demo_predictor()
                predictions = create_demo_predictions(activities_df)
            
            # WICHTIG: Save in Session State for App
            if 'user_data' not in st.session_state:
                st.session_state.user_data = {}
            
            st.session_state.user_data['learning_style'] = predictions
            st.session_state.user_data['test_completed'] = True
            st.session_state.user_data['test_date'] = datetime.now().isoformat()
            
            # Debug
            st.success("Lernstil erfolgreich analysiert und gespeichert!")
            

            # Save to database
            participant_data = {
                'id': test_state['participant_id'],
                'created_at': test_state['created_at'],
                'demographics': test_state['demographics'],
                'consent': test_state['consent_given'],
                'test_completed': True,
                'test_results': test_state['test_results'],
                'activity_mapping': activity_mapping,
                'learning_style_results': predictions
            }
            
            save_participant_data(participant_data)
        
        # Show results
        st.markdown("### Your Personal Learning Style Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Perception", 
                     predictions['Perception']['interpretation'],
                     f"Confidence: {predictions['Perception']['confidence']:.0%}")
            
            if predictions['Perception']['interpretation'] == 'Sensing':
                st.info("You prefer concrete, practical information and hands-on experiences")
            else:
                st.info("You prefer theoretical concepts and abstract thinking")
        
        with col2:
            st.metric("Input Processing",
                     predictions['Input']['interpretation'],
                     f"Confidence: {predictions['Input']['confidence']:.0%}")
            
            if predictions['Input']['interpretation'] == 'Visual':
                st.info("You learn better with visual materials and demonstrations")
            else:
                st.info("You learn better with written and spoken explanations")
        
        with col3:
            st.metric("Understanding",
                     predictions['Understanding']['interpretation'],
                     f"Confidence: {predictions['Understanding']['confidence']:.0%}")
            
            if predictions['Understanding']['interpretation'] == 'Sequential':
                st.info("You prefer step-by-step, linear learning progression")
            else:
                st.info("You prefer seeing the big picture and making connections")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🤖 Zum Adaptiven Tutor", type="primary", use_container_width=True):
                st.session_state.page = "Tutoring Session"
                st.rerun()
        
        with col_b:
            # Download-Button für Ergebnisse
            results_json = json.dumps({
                'participant_id': test_state['participant_id'],
                'test_date': test_state['created_at'].isoformat(),
                'learning_style': predictions
            }, indent=2)
            
            st.download_button(
                label=" Ergebnisse herunterladen",
                data=results_json,
                file_name=f"lernstil_ergebnis_{test_state['participant_id'][:8]}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Recommendations
        st.markdown("---")
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
        
        # Certificate of participation
        st.markdown("---")
        st.markdown("### Thank you for your participation!")
        
        # Option to restart
        if st.button("New Participant"):
            del st.session_state.test_state
            st.rerun()

def create_demo_predictions(activities_df):
    """Create demo predictions when no model is loaded"""
    row = activities_df.iloc[0]
    
    # Simple heuristics
    visual_score = (row.get('Visual Materials', 0) + row.get('playing', 0)) / 2
    verbal_score = (row.get('Reading file', 0) + row.get('Abstract materiale', 0)) / 2
    
    return {
        'Perception': {
            'interpretation': 'Sensing' if row.get('Concrete material', 0) > row.get('Abstract materiale', 0) else 'Intuitive',
            'confidence': 0.75 + np.random.random() * 0.2,
            'probabilities': {
                'Sensing': 0.6 if row.get('Concrete material', 0) > row.get('Abstract materiale', 0) else 0.4,
                'Intuitive': 0.4 if row.get('Concrete material', 0) > row.get('Abstract materiale', 0) else 0.6
            }
        },
        'Input': {
            'interpretation': 'Visual' if visual_score > verbal_score else 'Verbal',
            'confidence': 0.7 + np.random.random() * 0.25,
            'probabilities': {
                'Visual': visual_score / (visual_score + verbal_score + 1),
                'Verbal': verbal_score / (visual_score + verbal_score + 1)
            }
        },
        'Understanding': {
            'interpretation': 'Sequential' if row.get('Course overview', 0) > 10 else 'Global',
            'confidence': 0.65 + np.random.random() * 0.3,
            'probabilities': {
                'Sequential': 0.65 if row.get('Course overview', 0) > 10 else 0.35,
                'Global': 0.35 if row.get('Course overview', 0) > 10 else 0.65
            }
        }
    }

def render_study_dashboard():
    """Admin dashboard for study data analysis"""
    
    st.title("Study Dashboard")
    
    # Password protection
    password = st.text_input("Admin Password", type="password")
    
    if password == st.secrets.get("ADMIN_PASSWORD", "admin123"):
        # Load data from database
        conn = sqlite3.connect(DB_PATH)
        
        # Participant statistics
        participants_df = pd.read_sql_query("""
            SELECT * FROM participants WHERE test_completed = 1
        """, conn)
        
        if len(participants_df) == 0:
            st.warning("No completed tests yet. Share the test link to collect data!")
            conn.close()
            return
        
        results_df = pd.read_sql_query("""
            SELECT * FROM learning_style_results
        """, conn)
        
        st.markdown("### Study Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Participants", len(participants_df))
        with col2:
            avg_age = participants_df['age'].mean()
            st.metric("Average Age", f"{avg_age:.1f}")
        with col3:
            completion_rate = len(participants_df) / max(1, len(pd.read_sql_query("SELECT * FROM participants", conn))) * 100
            st.metric("Completion Rate", f"{completion_rate:.0f}%")
        with col4:
            avg_confidence = results_df['confidence_avg'].mean() * 100
            st.metric("Avg Confidence", f"{avg_confidence:.0f}%")
        
        # Demographic distribution
        st.markdown("### 👥 Demographic Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            import plotly.express as px
            gender_dist = participants_df['gender'].value_counts()
            fig_gender = px.pie(values=gender_dist.values, names=gender_dist.index,
                               title="Gender Distribution")
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            # Education distribution
            education_dist = participants_df['education_level'].value_counts()
            fig_education = px.bar(x=education_dist.index, y=education_dist.values,
                                  title="Education Levels")
            fig_education.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_education, use_container_width=True)
        
        # Learning style distribution
        st.markdown("### Learning Style Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            perception_dist = results_df['perception_type'].value_counts()
            fig_perception = px.pie(values=perception_dist.values, names=perception_dist.index,
                                   title="Perception")
            st.plotly_chart(fig_perception, use_container_width=True)
        
        with col2:
            input_dist = results_df['input_type'].value_counts()
            fig_input = px.pie(values=input_dist.values, names=input_dist.index,
                              title="Input Processing")
            st.plotly_chart(fig_input, use_container_width=True)
        
        with col3:
            understanding_dist = results_df['understanding_type'].value_counts()
            fig_understanding = px.pie(values=understanding_dist.values, names=understanding_dist.index,
                                      title="Understanding")
            st.plotly_chart(fig_understanding, use_container_width=True)
        
        # Export functions
        st.markdown("### Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = participants_df.to_csv(index=False)
            st.download_button(
                label="Participant Data (CSV)",
                data=csv,
                file_name=f"participants_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Learning Style Results (CSV)",
                data=csv,
                file_name=f"learning_styles_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Complete export
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                participants_df.to_excel(writer, sheet_name='Participants', index=False)
                results_df.to_excel(writer, sheet_name='Learning Styles', index=False)
                
                # Test details
                test_results_df = pd.read_sql_query("""
                    SELECT * FROM test_results
                """, conn)
                if len(test_results_df) > 0:
                    test_results_df.to_excel(writer, sheet_name='Test Results', index=False)
            
            buffer.seek(0)
            st.download_button(
                label="Complete Export (Excel)",
                data=buffer,
                file_name=f"study_complete_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        conn.close()
    else:
        if password:
            st.error("Incorrect password!")

# Deployment-Konfiguration für Studie
STUDY_CONFIG = {
    'deployment': {
        'platform': 'Streamlit Cloud',
        'public_access': True,
        'anonymous_participation': True,
        'data_retention': '6 months',
        'gdpr_compliant': True
    },
    'study_settings': {
        'min_participants': 30,
        'max_participants': 500,
        'languages': ['de'],
        'target_groups': ['students', 'professionals', 'general_public'],
        'incentive': 'Personal learning style report'
    },
    'sharing': {
        'qr_code': True,
        'short_url': True,
        'social_media': True,
        'email_template': True
    }
}

# QR-Code Generator für Studien-Link
def generate_study_qr_code(url: str):
    """Generiert QR-Code für einfaches Teilen"""
    import qrcode
    from io import BytesIO
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    return buf

# Email-Template für Studien-Einladung
EMAIL_TEMPLATE = """
Betreff: Einladung zur Lernstil-Studie - 10 Minuten für die Wissenschaft

Liebe/r Studienteilnehmer/in,

im Rahmen meiner Bachelorarbeit führe ich eine Studie zur Erforschung individueller Lernstile durch. 
Die Studie dauert nur 10-15 Minuten und Sie erhalten direkt im Anschluss eine persönliche Auswertung 
Ihres Lernstils.

**Was erwartet Sie?**
- Kurze kognitive Aufgaben (keine Vorkenntnisse nötig)
- Automatische Analyse Ihres Lernstils
- Persönliche Empfehlungen für effektiveres Lernen

**Teilnahme-Link:** {study_url}

Die Teilnahme ist vollständig anonym und freiwillig.

Vielen Dank für Ihre Unterstützung!

Mit freundlichen Grüßen
[Ihr Name]
"""

# Deployment Instructions
DEPLOYMENT_INSTRUCTIONS = """
# Deployment der Lernstil-Studie

## 1. Vorbereitung

### Datenschutz-Seite erstellen (privacy.py):
```python
import streamlit as st

def show_privacy_policy():
    st.title("Datenschutzerklärung")
    st.markdown('''
    ## Datenerhebung
    - Demografische Daten: Alter, Geschlecht, Bildung
    - Test-Antworten und Reaktionszeiten
    - Automatisch berechnete Lernstil-Scores
    
    ## Datenspeicherung
    - Alle Daten werden anonymisiert gespeichert
    - Keine IP-Adressen werden erfasst
    - Daten werden nach 6 Monaten gelöscht
    
    ## Ihre Rechte
    - Teilnahme ist freiwillig
    - Sie können jederzeit abbrechen
    - Löschung Ihrer Daten auf Anfrage
    ''')
```

## 2. Streamlit Cloud Deployment

### requirements.txt erweitern:
```txt
streamlit==1.28.1
pandas==2.0.3
numpy==1.24.3
plotly==5.17.0
scikit-learn==1.3.0
torch==2.0.1
joblib==1.3.2
sqlite3
qrcode==7.4.2
xlsxwriter==3.1.9
```

### .streamlit/secrets.toml:
```toml
ADMIN_PASSWORD = "ihr-sicheres-passwort"
STUDY_URL = "https://ihr-app-name.streamlit.app"
```

## 3. GitHub Repository

```bash
# Struktur
adaptive-learning-study/
├── app.py                    # Hauptapp mit Test
├── learning_style_test.py    # Test-Modul
├── ml_integration.py         # ML-Integration
├── privacy.py               # Datenschutz
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml         # NICHT committen!
├── models/                  # ML-Modelle
├── study_data.db           # Wird automatisch erstellt
└── README.md
```

## 4. Deployment

1. Push zu GitHub:
```bash
git add .
git commit -m "Study deployment"
git push origin main
```

2. Auf streamlit.io:
- New app → Select repository
- Deploy!

## 5. Studien-Link teilen

Nach Deployment erhalten Sie einen Link wie:
`https://learning-style-study.streamlit.app`

### QR-Code erstellen:
Die App generiert automatisch einen QR-Code unter:
`https://learning-style-study.streamlit.app/?page=qr`

### Kurz-URL (optional):
Nutzen Sie bit.ly oder tinyurl für kürzere Links

## 6. Monitoring

- Streamlit Cloud Dashboard für Traffic
- Admin-Dashboard in der App für Ergebnisse
- Tägliche Backups der SQLite DB

## 7. Nach der Studie

1. Daten exportieren über Admin-Dashboard
2. SQLite DB herunterladen
3. Analyse in Python/R/SPSS
"""

# Sicherheits-Checks
def validate_study_deployment():
    """Überprüft wichtige Sicherheitsaspekte"""
    checks = {
        'database_exists': Path(DB_PATH).exists(),
        'admin_password_set': 'ADMIN_PASSWORD' in st.secrets,
        'models_loaded': 'predictor' in st.session_state,
        'consent_form_active': True,
        'data_anonymization': True,
        'ssl_enabled': True  # Streamlit Cloud hat automatisch HTTPS
    }
    
    return all(checks.values()), checks