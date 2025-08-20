# Adaptive Tutoring System with Deep Reinforcement Learning
**Bachelor's Thesis Implementation - Comprehensive Evaluation Package**

An intelligent educational platform that combines learning style classification with deep reinforcement learning for personalized content delivery, achieving 78.5% average accuracy in learning style prediction and sophisticated adaptive tutoring capabilities.

## Quick Start for Evaluation

### Prerequisites
- **Python 3.9+** (Tested on Python 3.10.11)
- **8GB RAM minimum** (recommended for RL model loading)
- **2GB free disk space**

### Installation and Setup
```bash
# Navigate to project directory
cd adaptive_tutoring_system

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r src/requirements.txt

# Launch the system
streamlit run src/app.py
```

The system will automatically open in your browser at `http://localhost:8501`

---

## System Overview

This adaptive tutoring system represents a complete implementation of personalized educational technology, featuring:

- **Learning Style Classification**: 78.5% average accuracy across 3 FSLSM dimensions
- **Deep Q-Network Agent**: Dueling DQN architecture for adaptive content delivery  
- **Comprehensive Web Interface**: Interactive assessment and tutoring platform
- **Research-Grade Implementation**: Complete with evaluation metrics and analysis tools

### Core Components
- **ML Pipeline**: Ensemble methods (Random Forest, XGBoost, Gradient Boosting)
- **RL Engine**: PyTorch-based DQN with 30D state space and 9D action space
- **Web Application**: Streamlit-based interface with real-time adaptation
- **Analytics**: Comprehensive evaluation and visualization suite

---

## Project Architecture

```
adaptive_learning_system/
├── src/                          # Main source code
│   ├── app.py                   # Main Streamlit application
│   ├── learning_style_test.py   # Interactive assessment interface
│   ├── adaptive_tutor.py        # Deep RL tutoring system
│   ├── main_analysis_ls.py      # Complete ML pipeline
│   ├── deep_rl_training.py      # DQN training implementation
│   ├── generate_thesis_plots.py # Research visualization suite
│   └── models/                  # Trained model files
│       ├── feature_pipeline.pkl
│       ├── final_model_Input.pkl
│       ├── final_model_Perception.pkl
│       ├── final_model_Understanding.pkl
│       └── best_model.pth
├── data/                        # Training datasets
│   ├── CSHS.xlsx               # High school learning data
│   └── CSMS.xlsx               # Middle school learning data  
├── results/                     # Evaluation outputs
│   ├── figures/                # Research visualizations
│   └── metrics/                # Performance evaluations
└── README.md                   # This comprehensive guide
```

---

## Academic Evaluation

### Core Evaluation Scenarios

#### 1. Learning Style Classification Demo
```bash
# Quick integration test showing ML pipeline
python src/quickstart_integration.py
```
**Expected Output**: Demonstrates 78.5% classification accuracy with confidence scores

#### 2. Complete System Evaluation  
```bash
# Launch full web application
streamlit run src/app.py
```
**Key Features to Evaluate**:
- Learning style assessment interface
- Real-time ML predictions with confidence scores  
- Adaptive tutoring with RL-based content selection
- Progress analytics and visualization

#### 3. Research Analysis Pipeline
```bash
# Generate comprehensive analysis
python src/main_analysis_ls.py
```
**Outputs**: Complete evaluation metrics, confusion matrices, feature importance

#### 4. Thesis Visualizations
```bash
# Create publication-quality plots
python src/generate_thesis_plots.py
```
**Generated**: All research figures used in thesis documentation

### Expected Performance Metrics
- **Perception Dimension**: 77.2% accuracy (best_single)
- **Input Dimension**: 78.7% accuracy (voting ensemble)  
- **Understanding Dimension**: 79.7% accuracy (best_single)
- **Overall Average**: 78.5% classification accuracy
- **RL Training**: 3 configurations (500, 1451, 3000 episodes) with comprehensive evaluation

---

## Technical Implementation

### Machine Learning Pipeline
- **Ensemble Methods**: Voting and Stacking classifiers
- **Base Models**: Random Forest, XGBoost, Gradient Boosting
- **Feature Engineering**: 12 to 40 advanced behavioral indicators  
- **Class Imbalance**: SMOTE and threshold optimization
- **Evaluation**: McNemar's test, F1-optimization, confidence quantification

### Deep Reinforcement Learning
- **Architecture**: Dueling DQN with value/advantage separation
- **State Space**: 30-dimensional comprehensive learning context
- **Action Space**: 9-dimensional continuous pedagogical control  
- **Reward Function**: Multi-component (learning 40%, personalization 25%, wellbeing 20%, pedagogy 15%)
- **Training**: Three configurations with comprehensive evaluation

### Web Application  
- **Framework**: Streamlit with session state management
- **Real-time ML**: Cached model loading and prediction
- **Database**: SQLite for student profiles and progress
- **Visualization**: Plotly interactive charts and analytics

---

## Data Requirements

### Training Data (Included)
- **CSHS.xlsx**: 983 high school student activity records
- **CSMS.xlsx**: Additional middle school behavioral data
- **Features**: 12 Moodle LMS activity types mapped to learning styles

### Model Files (Pre-trained, Included)
All required model files are included. The system works out-of-the-box.
- `feature_pipeline.pkl` - Feature preprocessing pipeline
- `final_model_*.pkl` - Three dimension-specific classifiers  
- `best_model.pth` - Trained DQN model (optional)

---

## Detailed Usage Instructions

### 1. Initial Setup
```bash
# Verify Python version (3.9+ required)  
python --version

# Install dependencies
pip install -r src/requirements.txt

# Verify installation
python -c "import streamlit, torch, sklearn, xgboost; print('All dependencies installed successfully')"
```

### 2. Launch Main Application
```bash
streamlit run src/app.py
```

**Navigation Guide**:
- **Home**: System overview and quick demo
- **Learning Style Test**: Interactive assessment (takes 5-10 minutes)
- **Adaptive Tutor**: Main tutoring interface with RL adaptation
- **Analytics**: Progress tracking and learning insights

### 3. Evaluation Workflows

#### Academic Reviewer Workflow
1. Launch `streamlit run src/app.py`
2. Complete learning style assessment (demonstrates ML pipeline)
3. Try adaptive tutoring (demonstrates RL integration)  
4. Review analytics and visualizations
5. Run `python src/generate_thesis_plots.py` for research figures

#### Technical Validation Workflow
1. Run `python src/quickstart_integration.py` (demonstrates core ML)
2. Run `python src/main_analysis_ls.py` (comprehensive evaluation)
3. Examine output files in `results/` directory
4. Review training logs and metrics

---

## Performance Validation

### Key Validation Commands
```bash
# 1. Verify ML Pipeline
python src/quickstart_integration.py
# Expected: Shows predictions for 10 students with confidence scores

# 2. Generate Research Metrics  
python src/main_analysis_ls.py
# Expected: Creates detailed evaluation in results/metrics/

# 3. Create Thesis Visualizations
python src/generate_thesis_plots.py
# Expected: Generates 6 publication-quality plots in results/figures/

# 4. RL System Analysis
python src/evaluate_deep_rl_results.py
# Expected: Comprehensive RL training analysis and metrics
```

### Expected Output Files
- `results/metrics/final_results_summary.json` - Complete ML evaluation
- `results/figures/*.png` - All research visualizations  
- `study_data.db` - Student interaction database
- Multiple `training_metrics.json` files - RL training logs

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Import Errors
```bash
# Missing dependencies
pip install --upgrade -r src/requirements.txt

# PyTorch CPU version (if GPU issues)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Model Loading Issues
```bash
# Verify model files exist
ls -la src/models/
# Should show: feature_pipeline.pkl, final_model_*.pkl files

# If missing, run training pipeline
python src/main_analysis_ls.py  # Recreates ML models
```

#### Streamlit Port Conflicts
```bash
# Use different port
streamlit run src/app.py --server.port 8502
```

#### Memory Issues (RL Models)
- **Minimum 8GB RAM recommended**
- Close other applications
- Use CPU-only PyTorch if needed

### Performance Expectations
- **Startup Time**: 30-60 seconds (model loading)
- **Prediction Time**: Less than 1 second per classification  
- **Web Interface**: Responsive on modern browsers
- **RL Training**: Requires 4-8GB RAM, 30+ minutes

---

## System Capabilities

### 1. Learning Style Classification
- **Input**: Student activity data (12 behavioral features)
- **Processing**: Feature engineering (40 features) + ensemble classification
- **Output**: Probabilistic predictions for 3 FSLSM dimensions with confidence scores

### 2. Adaptive Content Selection  
- **Input**: Student profile + current learning state (30D vector)
- **Processing**: Dueling DQN with learned educational policies
- **Output**: Personalized content recommendations (9D action vector)

### 3. Real-time Adaptation
- **Monitoring**: Engagement, performance, fatigue indicators
- **Adaptation**: Dynamic difficulty, content type, pacing adjustments
- **Feedback**: Continuous learning and profile updates

### 4. Educational Analytics
- **Progress Tracking**: Topic mastery, learning velocity, engagement trends
- **Style Evolution**: Changes in learning preferences over time  
- **Performance Insights**: Strength/weakness identification and recommendations

---

## Research Contributions

### Novel Technical Achievements
1. **Hybrid Assessment**: Combines questionnaire + behavioral inference
2. **Educational RL**: DQN specifically designed for learning contexts
3. **Multi-component Rewards**: Pedagogically grounded reward function
4. **Confidence Integration**: Uncertainty-aware personalization
5. **Real-world Deployment**: Complete web-based implementation

### Validated Results  
- **78.5% Classification Accuracy** across three learning style dimensions
- **Comprehensive RL Evaluation** with multiple training configurations
- **Statistical Validation** using McNemar's test and confidence intervals
- **Feature Engineering Impact**: 10.3% accuracy improvement
- **Processing Dimension Exclusion**: Scientifically justified (53.8% approximately random)

---

## User Types and Focus Areas

### Academic Evaluators
- **Focus**: Main web application (`streamlit run src/app.py`)
- **Key Demonstrations**: Learning style test + adaptive tutoring
- **Documentation**: Thesis chapters align with actual implementation

### Technical Reviewers
- **Focus**: Code quality, ML pipeline (`src/main_analysis_ls.py`)
- **Key Metrics**: Classification accuracy, RL training logs
- **Architecture**: Modular design, proper evaluation methodology

### Researchers  
- **Focus**: Reproducible results (`src/generate_thesis_plots.py`)
- **Key Outputs**: Research visualizations, statistical validation
- **Extensions**: Well-documented for future improvements

---

## Support Information

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.9+ (tested on 3.10.11)
- **RAM**: 8GB minimum, 16GB recommended  
- **Storage**: 2GB free space
- **Network**: Internet connection for initial package installation

### If Something Doesn't Work
1. **Check Dependencies**: Run `pip install -r src/requirements.txt`
2. **Verify Python Version**: Ensure Python 3.9+ is installed
3. **Check File Permissions**: Ensure all files are readable
4. **Memory Requirements**: Ensure minimum 8GB RAM available
5. **Port Availability**: Default Streamlit port 8501 should be free

---

## System Validation Checklist

Before evaluation, verify these key components work:

- [ ] **Web application launches** (`streamlit run src/app.py`)
- [ ] **Learning style test completes** (provides ML predictions)
- [ ] **Adaptive tutor responds** (shows RL-based recommendations)  
- [ ] **Analytics display correctly** (shows progress visualizations)
- [ ] **Quick integration runs** (`python src/quickstart_integration.py`)
- [ ] **Research plots generate** (`python src/generate_thesis_plots.py`)

All checkboxes should be completed for full system validation.

---

## Additional Resources

### Key Files for Understanding
- **`CLAUDE.md`** - Development documentation and system overview
- **`src/app.py`** - Main application entry point and architecture
- **`src/main_analysis_ls.py`** - Complete ML evaluation methodology
- **`results/metrics/final_results_summary.json`** - Detailed performance metrics

### Research Outputs
- **`results/figures/`** - All thesis visualizations  
- **`results/metrics/`** - Statistical evaluation results
- **Training logs in model directories** - RL training progression

### Repository
**GitHub**: https://github.com/edvur/adaptive_learning_systems/

---

## Advanced Usage (Optional)

### Custom RL Training Configurations
```bash
# Quick test (50 episodes)
python src/deep_rl_training.py --config debug

# Standard training (500 episodes)  
python src/deep_rl_training.py --config standard

# Intensive training (1451 episodes)
python src/deep_rl_training.py --config intensive

# Research configuration (3000 episodes)
python src/deep_rl_training.py --config research
```

### Model Retraining
```bash
# Retrain ML models from scratch
python src/main_analysis_ls.py

# Evaluate Deep RL results
python src/evaluate_deep_rl_results.py

# Extract specific results  
python src/extract_results.py
```

### Custom Visualizations
```bash
# Generate all thesis plots
python src/generate_thesis_plots.py

# Data quality analysis
python src/data_quality_check.py

# Quick partial analysis
python src/quick_analysis_partial_labels.py
```

---

This system represents a complete, functional implementation of adaptive educational technology with research-grade evaluation and real-world applicability. The combination of learning style classification and deep reinforcement learning creates a novel approach to personalized education that can be immediately deployed and evaluated.

For questions about specific implementation details or evaluation procedures, all code is thoroughly documented and follows academic research standards. The system is designed to work seamlessly for academic evaluation while providing comprehensive insights into the technical achievements and research contributions.