# Summary: Multi-Label Learning Style Classification with 4 Labels

## 1. Initial Situation and Objectives

### Dataset
- **Source**: IEEE DataPort - Learning Style Identification
- **Basis**: Felder-Silverman Learning Style Model (FSLSM)
- **Data**: 2 courses (CSMS: 564 learners, CSHS: 1749 learners)
- **Available Samples**: 983 out of 2622 (37.5%) with complete labels
- **Features**: 12 Moodle activity metrics + 21 engineered features = 33 features

### Target Variables (4 Labels)
1. **Processing**: Active (ACT) vs. Reflective (REF)
2. **Perception**: Sensing (SEN) vs. Intuitive (INT)
3. **Input**: Visual (VIS) vs. Verbal (VRB)
4. **Understanding**: Sequential (SEQ) vs. Global (GLO)

### Goal
Average accuracy of at least 80% across all 4 labels

## 2. Methodological Approach

### 2.1 Feature Engineering
```python
# New features based on learning style theory
- visual_text_ratio = Visual Materials / (Reading + Abstract)
- concrete_abstract_ratio = Concrete / Abstract
- active_passive_ratio = (Exercises + Quiz) / (Reading + Visual)
- total_engagement = Sum of all activities
- video_completion_rate = Playing / (Playing + Paused + Unstarted)
# ... total of 21 additional features
```

### 2.2 Tested Approaches

#### Baseline Models
- Random Forest (balanced)
- Gradient Boosting
- Support Vector Machines
- Neural Networks (MLP)
- K-Nearest Neighbors

#### Advanced Techniques
1. **Label-specific optimization**: Individual model per label
2. **Ensemble methods**: Voting and Stacking
3. **Resampling**: SMOTE, ADASYN for imbalanced labels
4. **Threshold optimization**: Label-specific decision thresholds
5. **Semi-Supervised Learning**: Pseudo-labeling
6. **Label Dependencies**: Utilization of label correlations

## 3. Results

### 3.1 Best Individual Results per Label

| Label | Best Accuracy | Best Model | Problem |
|-------|---------------|------------|----------|
| **Processing** | 53.8% | Random Forest | No significant feature differences |
| **Perception** | 77.3% | Random Forest (balanced) | Strong class imbalance (3.35:1) |
| **Input** | 78.7% | Random Forest (balanced) | Strong class imbalance (3.70:1) |
| **Understanding** | 71.6% | Gradient Boosting | Moderate performance |

### 3.2 Overall Results of Different Approaches

| Approach | Avg. Accuracy | Exact Match Ratio | Remark |
|----------|---------------|-------------------|---------|
| Baseline RF | 68.1% | 22.8% | Without optimization |
| With Feature Engineering | 68.9% | 23.4% | 33 features instead of 12 |
| Label-specific | 69.7% | 24.9% | Individual model/label |
| With Threshold Opt. | 68.5% | 26.4% | Worsens average |
| Ensemble (Voting) | 69.1% | 23.5% | Marginal improvement |

**Best Result: 69.7% average accuracy**

### 3.3 Detailed Analysis of Processing Label

```
Confusion Matrix for Processing:
                Predicted
Actual      Reflective  Active
Reflective      11      90     (10.9% correct)
Active           7      89     (92.7% correct)

Problem: Model classifies almost everything as "Active"
```

**Statistical Analysis**:
- Exercises Submit: Active=3.76, Reflective=3.57 (p=0.18, not significant!)
- Quiz Submitted: Active=11.29, Reflective=11.28 (practically identical)
- Reading File: Active=42.03, Reflective=44.86 (minimal difference)

## 4. Data Quality Problems

### 4.1 Identified Problems

1. **Missing Significance in Processing**
   - T-test shows no significant differences (p>0.05)
   - Features do not discriminate between Active/Reflective

2. **Duplicates**
   - 88 data points with identical features
   - All with label combination: ACT-SEN-VIS-GLO
   - Indicates systematic error

3. **Class Imbalance**
   - Perception: 757 SEN vs. 226 INT (3.35:1)
   - Input: 774 VIS vs. 209 VRB (3.70:1)

4. **Data Loss**
   - 62.5% of data have incomplete labels
   - Only 983 out of 2622 samples usable

### 4.2 Label Correlations

```
               Processing  Perception     Input  Understanding
Processing       1.0000     0.1126     0.0890      0.0698
Perception       0.1126     1.0000    -0.0003      0.2644
Input            0.0890    -0.0003     1.0000     -0.0626
Understanding    0.0698     0.2644    -0.0626      1.0000
```

Only Perception-Understanding shows moderate correlation (0.26)

## 5. Scientific Insights

### 5.1 Main Finding
**The available Moodle activity data is not sufficient to model the Processing dimension (Active vs. Reflective) of the FSLSM.**

### 5.2 Possible Causes
1. **Inappropriate Features**: Click counts do not capture cognitive processes
2. **Missing Temporal Aspects**: When and how long was studied?
3. **Missing Sequential Patterns**: In what order were activities performed?
4. **External Factors**: Offline learning activities are not captured

### 5.3 Theoretical Implications
- Active/Reflective is an internal cognitive preference
- May not manifest in coarse activity metrics
- Requires finer behavioral markers (e.g., dwell time, scrolling behavior)

## 6. Rationale for 3-Label Approach

### 6.1 Statistical Rationale
- Processing contributes only 53.8% (barely better than chance)
- Without Processing: Expected accuracy ~75-78%
- With Processing: Maximum accuracy ~70% (empirically confirmed)

### 6.2 Scientific Integrity
- Transparent presentation of limitations
- Focus on functioning aspects
- Avoidance of "overfitting" on non-discriminating features

### 6.3 Practical Value
- 3-label model is robust and reliable
- Can be practically deployed for Perception, Input, Understanding
- Processing requires other data sources

## 7. Recommendations for Future Research

1. **Extended Data Collection for Processing**
   - Timestamps for each activity
   - Duration of activities
   - Order and patterns
   - Pauses between activities

2. **Alternative Data Sources**
   - Eye-tracking for reading behavior
   - Mouse movements and scrolling behavior
   - Forum posts (reflection vs. action)

3. **Label Validation**
   - Verification of ILS questionnaire results
   - Test-retest reliability
   - External validation by instructors

## 8. Conclusion

The analysis shows that reliable 4-label classification is not possible with the available data. The Processing dimension cannot be adequately modeled with the given Moodle activity metrics. A 3-label approach (without Processing) promises significantly better and more scientifically sound results.

**Next Step**: Implementation of an optimized 3-label classifier with the goal of achieving average accuracy ≥75%.