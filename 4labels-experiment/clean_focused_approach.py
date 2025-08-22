"""
Strategie: Processing zuerst verbessern, dann andere Labels
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
# Direct data loading without external dependencies

# Konstanten
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Optimale Thresholds (werden dynamisch angepasst)
THRESHOLDS = {
    'Processing': 0.5,
    'Perception': 0.35,
    'Input': 0.35,
    'Understanding': 0.5
}

def create_processing_features(X):
    """
    Erstellt spezielle Features für Processing (Active vs Reflective)
    """
    X_new = X.copy()
    
    # Create total_engagement if it doesn't exist
    if 'total_engagement' not in X_new.columns:
        activity_cols = ['Course overview', 'Reading file', 'Abstract materiale', 
                        'Concrete material', 'Visual Materials', 'Self-assessment',
                        'Exercises submit', 'Quiz submitted', 'playing', 
                        'paused', 'unstarted', 'buffering']
        available_cols = [col for col in activity_cols if col in X_new.columns]
        X_new['total_engagement'] = X_new[available_cols].sum(axis=1)
    
    # Active Learners: Hohe Aktivität in Übungen und Tests
    X_new['active_score'] = (
        2 * X['Exercises submit'] + 
        2 * X['Quiz submitted'] + 
        X['Self-assessment']
    ) / (X_new['total_engagement'] + 1)
    
    # Reflective Learners: Mehr Lesen und weniger direkte Aktion
    X_new['reflective_score'] = (
        2 * X['Reading file'] + 
        X['Abstract materiale'] + 
        X['Course overview']
    ) / (X_new['total_engagement'] + 1)
    
    # Tempo der Aktivitäten (Active = schneller)
    X_new['activity_tempo'] = X_new['total_engagement'] / (X['Course overview'] + 1)
    
    # Verhältnis Praxis zu Theorie
    X_new['practice_theory_ratio'] = (
        X['Exercises submit'] + X['Quiz submitted']
    ) / (X['Reading file'] + X['Abstract materiale'] + 1)
    
    # Interaktivität mit Videos (Active = mehr Interaktion)
    X_new['video_interactivity'] = (X['playing'] + X['paused']) / (X['unstarted'] + 1)
    
    return X_new

def train_processing_specialist(X_train, y_train, X_val, y_val):
    """
    Trainiert optimierte Modelle für Processing
    """
    print("Training Processing Specialist...")
    
    # Erstelle Processing-Features
    X_train_proc = create_processing_features(X_train)
    X_val_proc = create_processing_features(X_val)
    
    # Skalierung
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_val_scaled = scaler.transform(X_val_proc)
    
    # Modelle für Processing
    models = {
        'RF_balanced': RandomForestClassifier(
            n_estimators=1000, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'GB_tuned': GradientBoostingClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE
        ),
        'ET_balanced': ExtraTreesClassifier(
            n_estimators=1000, max_depth=10,
            class_weight='balanced', random_state=RANDOM_STATE
        )
    }
    
    # Trainiere und evaluiere
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train['Processing'])
        score = model.score(X_val_scaled, y_val['Processing'])
        print(f"  {name}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = (name, model, scaler)
    
    # Threshold-Optimierung
    print("\nOptimizing threshold for Processing...")
    model = best_model[1]
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.7, 0.05):
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_val['Processing'], y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    THRESHOLDS['Processing'] = best_threshold
    print(f"  Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    return best_model

def train_other_labels(X_train, y_train, X_val, y_val, processing_model):
    """
    Trainiert Modelle für andere Labels mit Processing als Feature
    """
    print("\nTraining models for other labels...")
    
    # Verwende Processing-Vorhersage als Feature
    _, proc_model, proc_scaler = processing_model
    X_train_proc = create_processing_features(X_train)
    X_val_proc = create_processing_features(X_val)
    
    X_train_proc_scaled = proc_scaler.transform(X_train_proc)
    X_val_proc_scaled = proc_scaler.transform(X_val_proc)
    
    proc_pred_train = proc_model.predict_proba(X_train_proc_scaled)[:, 1]
    proc_pred_val = proc_model.predict_proba(X_val_proc_scaled)[:, 1]
    
    X_train['processing_pred'] = proc_pred_train
    X_val['processing_pred'] = proc_pred_val
    
    models = {}
    
    for label in ['Perception', 'Input', 'Understanding']:
        print(f"\n{label}:")
        
        # Skalierung
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Wähle bestes Modell basierend auf Label-Charakteristiken
        if label in ['Perception', 'Input']:
            # Unbalanced labels - verwende balanced Random Forest
            model = RandomForestClassifier(
                n_estimators=800, max_depth=12,
                class_weight='balanced', random_state=RANDOM_STATE
            )
        else:
            # Understanding - Gradient Boosting
            model = GradientBoostingClassifier(
                n_estimators=500, max_depth=5,
                learning_rate=0.05, random_state=RANDOM_STATE
            )
        
        model.fit(X_train_scaled, y_train[label])
        score = model.score(X_val_scaled, y_val[label])
        print(f"  Accuracy: {score:.4f}")
        
        # Threshold-Optimierung für unbalanced labels
        if label in ['Perception', 'Input']:
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in np.arange(0.2, 0.6, 0.05):
                y_pred = (y_pred_proba > threshold).astype(int)
                f1 = f1_score(y_val[label], y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            THRESHOLDS[label] = best_threshold
            print(f"  Optimal threshold: {best_threshold:.2f}")
        
        models[label] = (model, scaler)
    
    return models

def ensemble_predict(models, X_test, processing_model):
    """
    Ensemble-Vorhersage mit optimierten Thresholds
    """
    predictions = {}
    
    # Processing vorhersagen
    _, proc_model, proc_scaler = processing_model
    X_test_proc = create_processing_features(X_test)
    X_test_proc_scaled = proc_scaler.transform(X_test_proc)
    
    proc_proba = proc_model.predict_proba(X_test_proc_scaled)[:, 1]
    predictions['Processing'] = (proc_proba > THRESHOLDS['Processing']).astype(int)
    
    # Processing als Feature für andere Labels
    X_test['processing_pred'] = proc_proba
    
    # Andere Labels vorhersagen
    for label, (model, scaler) in models.items():
        X_test_scaled = scaler.transform(X_test)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        predictions[label] = (proba > THRESHOLDS[label]).astype(int)
    
    return predictions

def evaluate_results(y_test, predictions):
    """
    Detaillierte Evaluation der Ergebnisse
    """
    print("\n" + "="*60)
    print("FINALE ERGEBNISSE")
    print("="*60)
    
    accuracies = []
    
    for label in y_test.columns:
        y_true = y_test[label]
        y_pred = predictions[label]
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        accuracies.append(acc)
        
        print(f"\n{label}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Confusion Matrix für Processing
        if label == 'Processing':
            cm = confusion_matrix(y_true, y_pred)
            print(f"  Confusion Matrix:")
            print(f"    Reflective: {cm[0,0]} correct, {cm[0,1]} wrong")
            print(f"    Active: {cm[1,1]} correct, {cm[1,0]} wrong")
    
    avg_accuracy = np.mean(accuracies)
    print(f"\nDurchschnittliche Accuracy: {avg_accuracy:.4f}")
    
    return avg_accuracy, accuracies

def prepare_4label_data():
    """Load and prepare data with all 4 labels directly"""
    print("Loading raw data...")
    
    # Load data directly - adjust these paths as needed
    data_dir = "../data"  # Adjust relative to your file location
    try:
        csms = pd.read_excel(f"{data_dir}/CSMS.xlsx")
        cshs = pd.read_excel(f"{data_dir}/CSHS.xlsx")
    except FileNotFoundError:
        # Try alternative paths
        try:
            csms = pd.read_excel("../adaptive_tutoring_system/data/CSMS.xlsx")
            cshs = pd.read_excel("../adaptive_tutoring_system/data/CSHS.xlsx")
        except FileNotFoundError:
            print("Error: Data files not found. Please check the path to CSMS.xlsx and CSHS.xlsx")
            raise
    
    # Combine datasets
    csms['source'] = 'CSMS'
    cshs['source'] = 'CSHS'
    df = pd.concat([csms, cshs], ignore_index=True)
    df.columns = df.columns.str.strip()
    
    print(f"Total datasets: {len(df)}")
    
    # Features
    feature_cols = [
        'Course overview', 'Reading file', 'Abstract materiale', 
        'Concrete material', 'Visual Materials', 'Self-assessment',
        'Exercises submit', 'Quiz submitted', 'playing', 
        'paused', 'unstarted', 'buffering'
    ]
    
    # Labels (including Processing)
    label_cols = ['Processing', 'Perception', 'Input', 'Understanding']
    
    # Filter complete data
    complete_data = df.dropna(subset=feature_cols + label_cols)
    print(f"Complete data with all labels: {len(complete_data)}")
    
    X = complete_data[feature_cols]
    y = complete_data[label_cols]
    
    # Convert labels to binary
    label_mapping = {
        'ACT': 1, 'REF': 0,  # Processing
        'SEN': 1, 'INT': 0,  # Perception  
        'VIS': 1, 'VRB': 0,  # Input
        'SEQ': 1, 'GLO': 0   # Understanding
    }
    
    for col in y.columns:
        y[col] = y[col].map(label_mapping)
    
    return X, y

def main():
    """
    Hauptfunktion - Sauberer, fokussierter Ansatz
    """
    print("FOKUSSIERTER ANSATZ: PROCESSING FIRST")
    print("="*60)
    
    # 1. Lade Daten
    X, y = prepare_4label_data()
    print(f"Daten geladen: {X.shape}")
    print(f"Labels: {list(y.columns)}")
    
    # Check if Processing label exists
    if 'Processing' not in y.columns:
        print("Error: Processing label not found in data!")
        return
    
    # 2. Train-Val-Test Split (60-20-20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # 3. Trainiere Processing-Spezialist
    print("\nPhase 1: Processing Specialist")
    print("-"*40)
    processing_model = train_processing_specialist(X_train, y_train, X_val, y_val)
    
    # 4. Trainiere andere Labels
    print("\nPhase 2: Other Labels with Processing Feature")
    print("-"*40)
    other_models = train_other_labels(X_train, y_train, X_val, y_val, processing_model)
    
    # 5. Finale Evaluation auf Test Set
    print("\nPhase 3: Final Evaluation")
    print("-"*40)
    predictions = ensemble_predict(other_models, X_test.copy(), processing_model)
    
    avg_accuracy, label_accuracies = evaluate_results(y_test, predictions)
    
    # 6. Visualisierung
    plt.figure(figsize=(10, 6))
    labels = list(y_test.columns)
    
    colors = ['red' if acc < 0.7 else 'orange' if acc < 0.8 else 'green' 
              for acc in label_accuracies]
    
    bars = plt.bar(labels, label_accuracies, color=colors)
    
    # Werte auf Balken
    for bar, acc in zip(bars, label_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom')
    
    plt.axhline(y=0.8, color='black', linestyle='--', label='Ziel (80%)')
    plt.axhline(y=avg_accuracy, color='blue', linestyle='-', 
                label=f'Durchschnitt ({avg_accuracy:.1%})')
    
    plt.ylabel('Accuracy')
    plt.title('Fokussierter Ansatz: Processing First')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('focused_approach_results.png')
    plt.close()
    

if __name__ == "__main__":
    main()
