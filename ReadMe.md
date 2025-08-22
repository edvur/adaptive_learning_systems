# Adaptive Learning System with Deep Reinforcement Learning

An intelligent educational platform that combines learning style classification with deep reinforcement learning for personalized content delivery, achieving **78.5% average accuracy** in learning style prediction and sophisticated adaptive tutoring capabilities.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Project Structure](#project-structure)
4. [Evaluation Workflows](#evaluation-workflows)
5. [Performance Metrics](#performance-metrics)
6. [Technical Implementation Details](#technical-implementation-details)
7. [Troubleshooting](#troubleshooting)
8. [Research Contributions](#research-contributions)
9. [Advanced Usage (Optional)](#advanced-usage-optional)
10. [Support and Additional Resources](#support-and-additional-resources)

---

## Quick Start

### Prerequisites
- **Python 3.9+** (Tested on Python 3.10.11)
- **8GB RAM minimum** (recommended for RL model loading)
- **2GB free disk space**

### Installation
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

The system opens automatically at `http://localhost:8501`

## System Overview

### Core Capabilities
- **Learning Style Classification**: 78.5% average accuracy across 3 FSLSM dimensions (Perception, Input, Understanding)
- **Deep Q-Network Agent**: Dueling DQN architecture for adaptive content delivery
- **Web Interface**: Interactive assessment and tutoring platform
- **Real-time Adaptation**: Dynamic content selection based on student performance

### Technical Architecture
- **ML Pipeline**: Ensemble methods (Random Forest, XGBoost, Gradient Boosting) with 40 engineered features
- **RL Engine**: PyTorch-based DQN with 30D state space and 9D action space
- **Web Application**: Streamlit-based interface with SQLite database
- **Analytics**: Comprehensive evaluation and visualization suite

## Project Structure

```
adaptive_tutoring_system/
├── src/                                # Main source code
│   ├── app.py                          # Main Streamlit application
│   ├── main_analysis_ls.py             # Complete ML classification pipeline
│   ├── deep_rl_training.py             # DQN training implementation
│   ├── adaptive_tutor.py               # Deep RL tutoring system
│   ├── learning_style_test.py          # Interactive assessment interface
│   ├── integration_LS_AT.py            # Bridge ML models with tutoring system
│   ├── streamlit_ml_integration.py     # Streamlit-ML interface bridge
│   ├── generate_plots.py               # Research visualization generator
│   ├── data_loader_module.py           # Data loading and preprocessing
│   ├── feature_engineering_module.py   # Feature engineering pipeline
│   ├── model_definitions_module.py     # ML model definitions
│   ├── model_training_module.py        # Training and optimization
│   ├── integration_setup_RL.py         # RL environment setup
│   ├── requirements.txt
│   └── models/                         # Pre-trained model files
│       ├── feature_pipeline.pkl        # Feature preprocessing
│       ├── final_model_*.pkl           # Learning style classifiers
│       └── best_model.pth              # Trained DQN (optional)
├── data/                               # Training datasets
│   ├── CSHS.xlsx                       # Course data (1749 learners)
│   └── CSMS.xlsx                       # Course data (564 learners)
├── results/                            # Evaluation outputs
│   ├── figures/                        # Research visualizations
│   └── metrics/                        # Performance evaluations
└── README.md
```

## Evaluation Workflows

### 1. Complete System Demo (Recommended for Academic Review)
```bash
streamlit run src/app.py
```
**Features to evaluate**:
- Learning style assessment (5-10 minutes, demonstrates ML pipeline)
- Adaptive tutoring with RL-based content selection
- Real-time analytics and progress visualization
- Student profile management

### 2. ML Pipeline Validation
```bash
python src/main_analysis_ls.py
```
**Outputs**: Complete evaluation metrics, confusion matrices, feature importance analysis

### 3. Research Visualizations
```bash
python src/generate_plots.py
```
**Generated**: All publication-quality plots used in thesis documentation


## Performance Metrics

### Learning Style Classification Results
| Dimension | Accuracy | Method | F1-Score |
|-----------|----------|---------|----------|
| **Perception** | 77.2% | Best individual model | 0.871 |
| **Input** | 78.7% | Voting ensemble | 0.881 |
| **Understanding** | 79.7% | Best individual model | 0.823 |
| **Overall Average** | **78.5%** | - | **0.858** |

### Deep RL Training Configurations
- **Standard**: 500 episodes (quick validation)
- **Intensive**: 1451 episodes (balanced training)
- **Research**: 3000 episodes (comprehensive evaluation)

## Technical Implementation Details

### Machine Learning Pipeline
- **Feature Engineering**: 12 → 40 behavioral indicators using theory-driven transformations
- **Class Imbalance Handling**: BorderlineSMOTE and optimized decision thresholds
- **Model Selection**: Ensemble of Random Forest, XGBoost, and Gradient Boosting
- **Validation**: McNemar's statistical testing and confidence quantification
- **Processing Dimension Exclusion**: Scientifically justified due to 53.8% accuracy (no statistical significance)

### Deep Reinforcement Learning
- **Architecture**: Dueling DQN with separate value/advantage streams
- **State Representation**: 30-dimensional learning context (profile + performance + engagement)
- **Action Space**: 9-dimensional continuous pedagogical control
- **Reward Function**: Multi-component design (learning 40%, personalization 25%, wellbeing 20%, pedagogy 15%)
- **Training**: Experience replay with prioritized sampling

### Web Application Architecture
- **Framework**: Streamlit with session state management
- **Database**: SQLite for student profiles and interaction history
- **Model Integration**: Cached loading with real-time prediction
- **Visualization**: Plotly interactive charts for analytics

## Troubleshooting

### Common Issues and Solutions

#### Installation Problems
```bash
# Verify Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install --upgrade -r src/requirements.txt

# For PyTorch GPU issues
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Model Loading Issues
```bash
# Verify model files exist
ls -la src/models/
# Should show: feature_pipeline.pkl, final_model_*.pkl

# Regenerate models if missing
python src/main_analysis_ls.py
```

#### Performance Issues
- **Minimum Requirements**: 8GB RAM, Python 3.9+
- **Startup Time**: 30-60 seconds (model loading)
- **Alternative Port**: `streamlit run src/app.py --server.port 8502`

## Research Contributions

### Technical Innovations
1. **Hybrid Learning Style Assessment**: Combines behavioral inference with questionnaire validation
2. **Educational Deep RL**: Custom DQN architecture optimized for learning contexts
3. **Multi-component Reward Design**: Pedagogically grounded reward function balancing multiple objectives
4. **Uncertainty-Aware Personalization**: Confidence-based adaptation strategies
5. **Production-Ready Implementation**: Complete web-based system with database integration

### Validated Findings
- **Feature Engineering Impact**: 10.3% accuracy improvement through theory-driven feature creation
- **Processing Dimension Limitation**: Statistical analysis proving inadequacy of Moodle data for Active/Reflective classification
- **Ensemble Method Effectiveness**: Voting classifiers outperform individual models for Input dimension
- **RL Training Stability**: Consistent performance across multiple training configurations


## Advanced Usage (Optional)

### Custom RL Training
```bash
# Different training intensities
python src/deep_rl_training.py --config standard    # 500 episodes
python src/deep_rl_training.py --config intensive   # 1451 episodes
python src/deep_rl_training.py --config research    # 3000 episodes
```

### Model Retraining
```bash
# Complete ML pipeline from scratch
python src/main_analysis_ls.py
```

### Data Analysis
```bash
# Quality analysis and visualizations
python 4labels-experiment/data_quality_check.py  
```

## Support and Additional Resources

### Key Implementation Files
- **`src/app.py`** - Main application architecture and user interface
- **`src/main_analysis_ls.py`** - Complete ML evaluation methodology
- **`results/metrics/final_results_summary.json`** - Detailed performance metrics

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Memory**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for models and results
- **Network**: Required for initial dependency installation

### Expected Output Files
- `results/metrics/final_results_summary.json` - Complete ML evaluation
- `results/figures/*.png` - Research visualizations
- `study_data.db` - Student interaction database
- Training logs in model directories

---

This system provides a complete, research-grade implementation of adaptive educational technology. The combination of learning style classification and deep reinforcement learning creates a novel approach to personalized education that can be immediately deployed and comprehensively evaluated.

For technical validation: Run `python src/main_analysis_ls.py` for comprehensive ML evaluation and `python src/generate_plots.py` for research visualizations.