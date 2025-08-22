"""
Data Quality Analysis: Verification of Label Correctness
Investigates:
1. Label distribution and anomalies
2. Consistency between features and labels
3. Duplicates and contradictions
4. Statistical outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

def load_raw_data():
    """Loads raw data without preprocessing"""
    print("Loading raw data...")
    # Use config for portable paths
    from config import get_config
    config = get_config()
    
    csms = pd.read_excel(config.get_data_path('csms'))
    cshs = pd.read_excel(config.get_data_path('cshs'))
    
    # Add source identifier
    csms['source'] = 'CSMS'
    cshs['source'] = 'CSHS'
    
    df = pd.concat([csms, cshs], ignore_index=True)
    df.columns = df.columns.str.strip()
    
    print(f"Total datasets: {len(df)}")
    return df

def check_label_distribution(df):
    """Überprüft die Verteilung der Labels"""
    print("\n" + "="*60)
    print("1. LABEL-VERTEILUNG ANALYSE")
    print("="*60)
    
    target_cols = ['Processing', 'Perception', 'Input', 'Understanding']
    
    # Overall distribution
    print("\nOverall label distribution:")
    for col in target_cols:
        value_counts = df[col].value_counts(dropna=False)
        print(f"\n{col}:")
        for value, count in value_counts.items():
            print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")
    
    # Distribution by source
    print("\nDistribution by data source:")
    for source in ['CSMS', 'CSHS']:
        print(f"\n{source}:")
        source_df = df[df['source'] == source]
        for col in target_cols:
            if col in source_df.columns:
                counts = source_df[col].value_counts()
                if len(counts) > 0:
                    print(f"  {col}: {counts.to_dict()}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(target_cols):
        data = df[col].dropna()
        if len(data) > 0:
            axes[i].hist(data.map({'ACT': 1, 'REF': 0, 'SEN': 1, 'INT': 0, 
                                   'VIS': 1, 'VRB': 0, 'SEQ': 1, 'GLO': 0}), 
                        bins=2, alpha=0.7)
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.close()

def check_label_combinations(df):
    """Überprüft Label-Kombinationen auf Plausibilität"""
    print("\n" + "="*60)
    print("2. LABEL-KOMBINATIONEN ANALYSE")
    print("="*60)
    
    target_cols = ['Processing', 'Perception', 'Input', 'Understanding']
    
    # Only complete datasets
    complete_df = df.dropna(subset=target_cols)
    print(f"\nComplete datasets: {len(complete_df)}")
    
    # Most frequent combinations
    if len(complete_df) > 0:
        combinations = complete_df[target_cols].value_counts().head(20)
        print("\nTop 20 label combinations:")
        for combo, count in combinations.items():
            print(f"  {combo}: {count} ({count/len(complete_df)*100:.1f}%)")
    
    # Correlation between labels
    print("\nLabel correlations:")
    
    # Mapping for numerical analysis
    mapping = {
        'ACT': 1, 'REF': 0,
        'SEN': 1, 'INT': 0,
        'VIS': 1, 'VRB': 0,
        'SEQ': 1, 'GLO': 0
    }
    
    if len(complete_df) > 0:
        numeric_labels = complete_df[target_cols].copy()
        for col in target_cols:
            numeric_labels[col] = numeric_labels[col].map(mapping)
        
        corr = numeric_labels.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Label Correlations')
        plt.tight_layout()
        plt.savefig('label_correlations.png')
        plt.close()
        
        print(corr)
        
        # Unexpected combinations
        print("\nChecking unexpected label combinations:")
        
        # Theoretically unlikely combinations
        # e.g. Very visual (VIS) but also very verbal (VRB)?
        suspicious = complete_df[
            (complete_df['Input'] == 'VIS') & 
            (complete_df['Processing'] == 'REF') &
            (complete_df['Perception'] == 'INT')
        ]
        
        if len(suspicious) > 0:
            print(f"  Suspicious combination (VIS+REF+INT): {len(suspicious)} cases")

def check_duplicates_and_inconsistencies(df):
    """Prüft auf Duplikate und inkonsistente Einträge"""
    print("\n" + "="*60)
    print("3. DUPLIKATE UND INKONSISTENZEN")
    print("="*60)
    
    feature_cols = [
        'Course overview', 'Reading file', 'Abstract materiale', 
        'Concrete material', 'Visual Materials', 'Self-assessment',
        'Exercises submit', 'Quiz submitted', 'playing', 
        'paused', 'unstarted', 'buffering'
    ]
    
    # Check for exact duplicates in features
    feature_duplicates = df[feature_cols].duplicated()
    n_duplicates = feature_duplicates.sum()
    
    print(f"\nExact feature duplicates: {n_duplicates}")
    
    if n_duplicates > 0:
        # Check if duplicates have different labels
        dup_indices = df[feature_duplicates].index
        
        # Compare labels for duplicates
        print("\nDuplicate analysis:")
        
        # Find groups of duplicates
        df_features = df[feature_cols].round(2)  # Round for comparison
        
        for idx in dup_indices[:5]:  # Analyze first 5 duplicates
            # Find all rows with same features
            same_features = (df_features == df_features.iloc[idx]).all(axis=1)
            similar_rows = df[same_features]
            
            if len(similar_rows) > 1:
                print(f"\nGroup with {len(similar_rows)} identical feature sets:")
                print("Labels:")
                print(similar_rows[['Processing', 'Perception', 'Input', 'Understanding']].value_counts())

def check_feature_label_consistency(df):
    """Prüft Konsistenz zwischen Features und Labels"""
    print("\n" + "="*60)
    print("4. FEATURE-LABEL KONSISTENZ")
    print("="*60)
    
    target_cols = ['Processing', 'Perception', 'Input', 'Understanding']
    feature_cols = [
        'Course overview', 'Reading file', 'Abstract materiale', 
        'Concrete material', 'Visual Materials', 'Self-assessment',
        'Exercises submit', 'Quiz submitted', 'playing', 
        'paused', 'unstarted', 'buffering'
    ]
    
    # Only complete data
    complete_df = df.dropna(subset=target_cols)
    
    if len(complete_df) == 0:
        print("No complete data for analysis")
        return
    
    # Mapping
    mapping = {
        'ACT': 1, 'REF': 0,
        'SEN': 1, 'INT': 0,
        'VIS': 1, 'VRB': 0,
        'SEQ': 1, 'GLO': 0
    }
    
    # Analyze Processing (Active vs Reflective)
    print("\nProcessing (Active vs Reflective) - Feature Analysis:")
    
    active_mask = complete_df['Processing'] == 'ACT'
    reflective_mask = complete_df['Processing'] == 'REF'
    
    # Expectation: Active Learners should have more Exercises/Quiz
    active_exercises = complete_df.loc[active_mask, 'Exercises submit'].mean()
    reflective_exercises = complete_df.loc[reflective_mask, 'Exercises submit'].mean()
    
    active_quiz = complete_df.loc[active_mask, 'Quiz submitted'].mean()
    reflective_quiz = complete_df.loc[reflective_mask, 'Quiz submitted'].mean()
    
    print(f"  Exercises - Active: {active_exercises:.2f}, Reflective: {reflective_exercises:.2f}")
    print(f"  Quiz - Active: {active_quiz:.2f}, Reflective: {reflective_quiz:.2f}")
    
    # Expectation: Reflective Learners should read more
    active_reading = complete_df.loc[active_mask, 'Reading file'].mean()
    reflective_reading = complete_df.loc[reflective_mask, 'Reading file'].mean()
    
    print(f"  Reading - Active: {active_reading:.2f}, Reflective: {reflective_reading:.2f}")
    
    # Statistical tests
    from scipy.stats import ttest_ind
    
    t_stat, p_value = ttest_ind(
        complete_df.loc[active_mask, 'Exercises submit'],
        complete_df.loc[reflective_mask, 'Exercises submit']
    )
    
    print(f"\n  T-Test for Exercises (Active vs Reflective): p={p_value:.4f}")
    
    if p_value > 0.05:
        print("No significant difference in Exercises!")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Processing vs Exercises
    axes[0, 0].boxplot([
        complete_df.loc[active_mask, 'Exercises submit'],
        complete_df.loc[reflective_mask, 'Exercises submit']
    ], labels=['Active', 'Reflective'])
    axes[0, 0].set_title('Processing vs Exercises Submit')
    axes[0, 0].set_ylabel('Exercises Submit')
    
    # Processing vs Reading
    axes[0, 1].boxplot([
        complete_df.loc[active_mask, 'Reading file'],
        complete_df.loc[reflective_mask, 'Reading file']
    ], labels=['Active', 'Reflective'])
    axes[0, 1].set_title('Processing vs Reading File')
    axes[0, 1].set_ylabel('Reading File')
    
    # Input (Visual vs Verbal)
    visual_mask = complete_df['Input'] == 'VIS'
    verbal_mask = complete_df['Input'] == 'VRB'
    
    # Visual vs Visual Materials
    axes[1, 0].boxplot([
        complete_df.loc[visual_mask, 'Visual Materials'],
        complete_df.loc[verbal_mask, 'Visual Materials']
    ], labels=['Visual', 'Verbal'])
    axes[1, 0].set_title('Input vs Visual Materials')
    axes[1, 0].set_ylabel('Visual Materials')
    
    # Perception vs Abstract/Concrete
    sensing_mask = complete_df['Perception'] == 'SEN'
    intuitive_mask = complete_df['Perception'] == 'INT'
    
    axes[1, 1].boxplot([
        complete_df.loc[sensing_mask, 'Concrete material'],
        complete_df.loc[intuitive_mask, 'Concrete material']
    ], labels=['Sensing', 'Intuitive'])
    axes[1, 1].set_title('Perception vs Concrete Material')
    axes[1, 1].set_ylabel('Concrete Material')
    
    plt.tight_layout()
    plt.savefig('feature_label_consistency.png')
    plt.close()

def detect_outliers(df):
    """Erkennt Ausreißer in den Daten"""
    print("\n" + "="*60)
    print("5. AUSREISSER-ERKENNUNG")
    print("="*60)
    
    feature_cols = [
        'Course overview', 'Reading file', 'Abstract materiale', 
        'Concrete material', 'Visual Materials', 'Self-assessment',
        'Exercises submit', 'Quiz submitted', 'playing', 
        'paused', 'unstarted', 'buffering'
    ]
    
    # Only features without NaN
    X = df[feature_cols].fillna(0)
    
    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso.fit_predict(X)
    
    n_outliers = (outliers == -1).sum()
    print(f"\nNumber of outliers: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
    
    # Analyze outliers
    outlier_indices = np.where(outliers == -1)[0]
    
    if len(outlier_indices) > 0:
        print("\nExamples of outliers:")
        for idx in outlier_indices[:5]:
            print(f"\nIndex {idx}:")
            print(f"  Total activities: {X.iloc[idx].sum():.0f}")
            print(f"  Max activity: {X.iloc[idx].max():.0f}")
            print(f"  Labels: {df.iloc[idx][['Processing', 'Perception', 'Input', 'Understanding']].values}")
    
    # PCA Visualisierung
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[outliers == 1, 0], X_pca[outliers == 1, 1], 
                c='blue', label='Normal', alpha=0.5)
    plt.scatter(X_pca[outliers == -1, 0], X_pca[outliers == -1, 1], 
                c='red', label='Ausreißer', s=100)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA: Ausreißer-Visualisierung')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outlier_detection.png')
    plt.close()

def generate_quality_report(df):
    """Erstellt einen Qualitätsbericht"""
    print("\n" + "="*60)
    print("QUALITÄTSBERICHT ZUSAMMENFASSUNG")
    print("="*60)
    
    issues = []
    
    # Check missing values
    missing_labels = df[['Processing', 'Perception', 'Input', 'Understanding']].isna().sum()
    total_missing = missing_labels.sum()
    
    if total_missing > len(df) * 0.5:
        issues.append(f"Very many missing labels: {total_missing} of {len(df)*4} possible")
    
    # Check class balance
    for col in ['Processing', 'Perception', 'Input', 'Understanding']:
        if col in df.columns:
            counts = df[col].value_counts()
            if len(counts) >= 2:
                ratio = counts.iloc[0] / counts.iloc[1]
                if ratio > 3 or ratio < 0.33:
                    issues.append(f"Strong class imbalance in {col}: Ratio {ratio:.2f}")
    
    # Summary
    print("\nIdentified problems:")
    for issue in issues:
        print(f"  {issue}")
    
    if len(issues) == 0:
        print(" No obvious data quality problems found")
    
    print("\nRecommendations:")
    print("1. Verify label assignment (ILS Questionnaire)")
    print("2. Check for data entry errors")
    print("3. Manually validate suspicious combinations")
    print("4. Consider external validation of labels")

def main():
    """Hauptfunktion für Datenqualitätsanalyse"""
    print("="*80)
    print("DATENQUALITÄTS-ANALYSE FÜR LERNSTIL-LABELS")
    print("="*80)
    
    # Load raw data
    df = load_raw_data()
    
    # Perform all checks
    check_label_distribution(df)
    check_label_combinations(df)
    check_duplicates_and_inconsistencies(df)
    check_feature_label_consistency(df)
    detect_outliers(df)
    generate_quality_report(df)
    
    print("\nAnalysis completed!")
    print("Check the generated plots:")
    print("  - label_distribution.png")
    print("  - label_correlations.png")
    print("  - feature_label_consistency.png")
    print("  - outlier_detection.png")

if __name__ == "__main__":
    main()
