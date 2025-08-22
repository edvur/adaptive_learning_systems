import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
# Direct data loading without external dependencies

# Für bessere Reproduzierbarkeit
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_clean(csms_file='CSMS.xlsx', cshs_file='CSHS.xlsx'):
    """Load and clean data with all 4 labels directly"""
    print("Loading raw data...")
    
    # Load data directly - adjust these paths as needed
    data_dir = "../data"  # Adjust relative to your file location
    try:
        csms = pd.read_excel(f"{data_dir}/{csms_file}")
        cshs = pd.read_excel(f"{data_dir}/{cshs_file}")
    except FileNotFoundError:
        # Try alternative paths
        try:
            csms = pd.read_excel(f"../adaptive_tutoring_system/data/{csms_file}")
            cshs = pd.read_excel(f"../adaptive_tutoring_system/data/{cshs_file}")
        except FileNotFoundError:
            print(f"Error: Data files not found. Please check the path to {csms_file} and {cshs_file}")
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

class MultiLabelMLPClassifier:
    """
    Spezialisierter MLP Classifier für Multi-Label-Probleme
    mit Label-Korrelationen
    """
    def __init__(self, hidden_layers=(300, 200, 100), learning_rate=0.001, 
                 epochs=1000, batch_size=32, random_state=42):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.models = {}
        self.label_correlations = None
        
    def _compute_label_correlations(self, y):
        """Berechnet Korrelationen zwischen Labels"""
        if isinstance(y, pd.DataFrame):
            self.label_correlations = y.corr()
        else:
            self.label_correlations = pd.DataFrame(y).corr()
    
    def fit(self, X, y):
        """Trainiert spezialisierte Modelle für jedes Label"""
        # Berechne Label-Korrelationen
        self._compute_label_correlations(y)
        
        # Trainiere ein Modell für jedes Label
        for i, label in enumerate(y.columns):
            print(f"Trainiere Modell für {label}...")
            
            # Erstelle erweiterte Features basierend auf Label-Korrelationen
            X_extended = self._create_extended_features(X, y, i)
            
            # MLP für dieses Label
            mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=self.batch_size,
                learning_rate='adaptive',
                learning_rate_init=self.learning_rate,
                max_iter=self.epochs,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.random_state
            )
            
            mlp.fit(X_extended, y.iloc[:, i])
            self.models[label] = mlp
    
    def _create_extended_features(self, X, y, label_idx):
        """Erstellt erweiterte Features mit Label-Informationen"""
        X_extended = X.copy()
        
        # Füge Vorhersagen anderer Labels als Features hinzu (wenn verfügbar)
        # Dies hilft, Label-Korrelationen zu nutzen
        for j, other_label in enumerate(y.columns):
            if j != label_idx:
                # Verwende das tatsächliche Label während des Trainings
                # (beim Predict müssen wir dies anders handhaben)
                correlation = abs(self.label_correlations.iloc[label_idx, j])
                if correlation > 0.3:  # Nur stark korrelierte Labels
                    X_extended[f'corr_label_{other_label}'] = y.iloc[:, j]
        
        return X_extended
    
    def predict(self, X):
        """Vorhersage mit Berücksichtigung von Label-Abhängigkeiten"""
        predictions = np.zeros((len(X), len(self.models)))
        
        # Iterative Vorhersage mit Label-Propagation
        for iteration in range(2):  # 2 Iterationen für Label-Propagation
            for i, (label, model) in enumerate(self.models.items()):
                X_extended = X.copy()
                
                # Füge vorherige Vorhersagen als Features hinzu
                for j, other_label in enumerate(self.models.keys()):
                    if j != i and iteration > 0:
                        correlation = abs(self.label_correlations.iloc[i, j])
                        if correlation > 0.3:
                            X_extended[f'corr_label_{other_label}'] = predictions[:, j]
                
                # Stelle sicher, dass die Feature-Anzahl übereinstimmt
                for col in model.feature_names_in_:
                    if col not in X_extended.columns and col.startswith('corr_label_'):
                        X_extended[col] = 0
                
                # Wähle nur die Features, die das Modell erwartet
                X_extended = X_extended[model.feature_names_in_]
                
                predictions[:, i] = model.predict(X_extended)
        
        return predictions.astype(int)

def create_polynomial_features(X, degree=2):
    """Erstellt polynomiale Features manuell"""
    X_poly = X.copy()
    
    # Erstelle Interaktionen zwischen wichtigen Features
    important_features = ['visual_text_ratio', 'concrete_abstract_ratio', 
                         'active_passive_ratio', 'total_engagement']
    
    for i, feat1 in enumerate(important_features):
        if feat1 in X.columns:
            for feat2 in important_features[i+1:]:
                if feat2 in X.columns:
                    X_poly[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                    
    # Quadratische Features für wichtige Variablen
    for feat in important_features:
        if feat in X.columns:
            X_poly[f'{feat}_squared'] = X[feat] ** 2
    
    return X_poly

def create_meta_features(X):
    """Erstellt Meta-Features basierend auf Datenaggregation"""
    X_meta = X.copy()
    
    # Perzentil-Features
    for percentile in [25, 50, 75]:
        X_meta[f'percentile_{percentile}'] = X.quantile(percentile/100, axis=1)
    
    # Verhältnisse zu Durchschnittswerten
    for col in ['Course overview', 'Reading file', 'Visual Materials', 'Exercises submit']:
        if col in X.columns:
            col_mean = X[col].mean()
            if col_mean > 0:
                X_meta[f'{col}_to_mean_ratio'] = X[col] / col_mean
    
    return X_meta

def train_optimized_model(X_train, y_train, X_test, y_test):
    """
    Trainiert ein optimiertes Modell mit allen Tricks
    """
    # 1. Erstelle erweiterte Features
    X_train_poly = create_polynomial_features(X_train)
    X_test_poly = create_polynomial_features(X_test)
    
    X_train_meta = create_meta_features(X_train_poly)
    X_test_meta = create_meta_features(X_test_poly)
    
    # 2. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_meta)
    X_test_scaled = scaler.transform(X_test_meta)
    
    # Konvertiere zurück zu DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_meta.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_meta.columns)
    
    # 3. Trainiere spezialisiertes Multi-Label-Modell
    print("\nTrainiere spezialisiertes Multi-Label Deep Learning Modell...")
    ml_mlp = MultiLabelMLPClassifier(
        hidden_layers=(400, 300, 200, 100),
        learning_rate=0.001,
        epochs=1500,
        batch_size=16,
        random_state=RANDOM_STATE
    )
    
    ml_mlp.fit(X_train_scaled, y_train)
    
    # 4. Vorhersage
    y_pred = ml_mlp.predict(X_test_scaled)
    
    # 5. Evaluation
    exact_match = (y_test == y_pred).all(axis=1).mean()
    
    print("\nDetaillierte Ergebnisse:")
    print(f"Exact Match Ratio: {exact_match:.4f}")
    
    avg_accuracy = 0
    for i, col in enumerate(y_test.columns):
        acc = accuracy_score(y_test[col], y_pred[:, i])
        f1 = f1_score(y_test[col], y_pred[:, i])
        prec = precision_score(y_test[col], y_pred[:, i])
        rec = recall_score(y_test[col], y_pred[:, i])
        
        print(f"\n{col}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        
        avg_accuracy += acc
    
    avg_accuracy /= len(y_test.columns)
    print(f"\nDurchschnittliche Accuracy: {avg_accuracy:.4f}")
    
    return ml_mlp, avg_accuracy, exact_match

def create_super_ensemble(X_train, y_train, X_test, y_test):
    """
    Erstellt ein Super-Ensemble mit mehreren Ansätzen
    """
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.multioutput import MultiOutputClassifier
    
    print("\nErstelle Super-Ensemble...")
    
    # Basis-Modelle
    models = {
        'rf1': RandomForestClassifier(n_estimators=1000, max_depth=15, min_samples_split=2, 
                                     class_weight='balanced', random_state=RANDOM_STATE),
        'rf2': RandomForestClassifier(n_estimators=800, max_depth=20, min_samples_leaf=1,
                                     class_weight='balanced_subsample', random_state=RANDOM_STATE+1),
        'et1': ExtraTreesClassifier(n_estimators=1000, max_depth=15, min_samples_split=2,
                                   class_weight='balanced', random_state=RANDOM_STATE),
        'et2': ExtraTreesClassifier(n_estimators=800, max_depth=None, min_samples_leaf=1,
                                   class_weight='balanced', random_state=RANDOM_STATE+1),
        'gb': GradientBoostingClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                                        subsample=0.8, random_state=RANDOM_STATE)
    }
    
    # Trainiere alle Modelle
    predictions = []
    
    for name, model in models.items():
        print(f"Trainiere {name}...")
        multi_model = MultiOutputClassifier(model, n_jobs=-1)
        multi_model.fit(X_train, y_train)
        pred = multi_model.predict(X_test)
        predictions.append(pred)
    
    # Majority Voting
    predictions = np.array(predictions)
    y_pred_ensemble = np.zeros_like(predictions[0])
    
    # Für jede Vorhersage, wähle die häufigste Klasse
    for i in range(y_pred_ensemble.shape[0]):
        for j in range(y_pred_ensemble.shape[1]):
            votes = predictions[:, i, j]
            y_pred_ensemble[i, j] = int(np.mean(votes) > 0.5)
    
    # Evaluation
    exact_match = (y_test == y_pred_ensemble).all(axis=1).mean()
    
    avg_accuracy = 0
    for i, col in enumerate(y_test.columns):
        acc = accuracy_score(y_test[col], y_pred_ensemble[:, i])
        avg_accuracy += acc
    
    avg_accuracy /= len(y_test.columns)
    
    print(f"\nSuper-Ensemble Ergebnisse:")
    print(f"Exact Match Ratio: {exact_match:.4f}")
    print(f"Durchschnittliche Accuracy: {avg_accuracy:.4f}")
    
    return y_pred_ensemble, avg_accuracy

def augment_data(X, y, augmentation_factor=2):
    """
    Data Augmentation für besseres Training
    """
    print(f"\nData Augmentation (Faktor: {augmentation_factor})...")
    
    X_aug = X.copy()
    y_aug = y.copy()
    
    for _ in range(augmentation_factor - 1):
        # Erstelle leicht modifizierte Kopien der Daten
        X_noisy = X.copy()
        
        # Füge kleines Rauschen hinzu
        noise_level = 0.05
        for col in X.columns:
            if X[col].std() > 0:
                noise = np.random.normal(0, X[col].std() * noise_level, len(X))
                X_noisy[col] += noise
        
        # Füge augmentierte Daten hinzu
        X_aug = pd.concat([X_aug, X_noisy])
        y_aug = pd.concat([y_aug, y])
    
    return X_aug.reset_index(drop=True), y_aug.reset_index(drop=True)

def main():
    print("="*80)
    print("DEEP LEARNING ANSATZ FÜR LERNSTILANALYSE")
    print("="*80)
    
    # Lade Daten
    X, y = load_and_clean('CSMS.xlsx', 'CSHS.xlsx')
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nOriginal Datensatz:")
    print(f"Training: {X_train.shape}")
    print(f"Test: {X_test.shape}")
    
    # Data Augmentation
    X_train_aug, y_train_aug = augment_data(X_train, y_train, augmentation_factor=3)
    print(f"\nNach Data Augmentation:")
    print(f"Training: {X_train_aug.shape}")
    
    # Ansatz 1: Spezialisiertes Multi-Label Deep Learning
    print("\n" + "="*60)
    print("ANSATZ 1: Multi-Label Deep Learning mit Label-Korrelationen")
    print("="*60)
    
    model1, acc1, em1 = train_optimized_model(X_train_aug, y_train_aug, X_test, y_test)
    
    # Ansatz 2: Super-Ensemble
    print("\n" + "="*60)
    print("ANSATZ 2: Super-Ensemble mit 5 optimierten Modellen")
    print("="*60)
    
    # Erstelle erweiterte Features für Ensemble
    X_train_poly = create_polynomial_features(X_train_aug)
    X_test_poly = create_polynomial_features(X_test)
    
    X_train_meta = create_meta_features(X_train_poly)
    X_test_meta = create_meta_features(X_test_poly)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_meta)
    X_test_scaled = scaler.transform(X_test_meta)
    
    pred2, acc2 = create_super_ensemble(X_train_scaled, y_train_aug, X_test_scaled, y_test)
    
    # Ansatz 3: Kombination beider Ansätze
    print("\n" + "="*60)
    print("ANSATZ 3: Hybrid - Kombination aus Deep Learning und Ensemble")
    print("="*60)
    
    # Trainiere nochmal für Hybrid
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train_meta.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test_meta.columns)
    
    # Deep Learning Vorhersage
    dl_pred = model1.predict(X_test_df)
    
    # Kombiniere Vorhersagen (gewichtetes Voting)
    hybrid_pred = np.zeros_like(dl_pred)
    for i in range(hybrid_pred.shape[0]):
        for j in range(hybrid_pred.shape[1]):
            # 60% Deep Learning, 40% Ensemble
            weighted_vote = 0.6 * dl_pred[i, j] + 0.4 * pred2[i, j]
            hybrid_pred[i, j] = int(weighted_vote > 0.5)
    
    # Evaluation Hybrid
    exact_match_hybrid = (y_test == hybrid_pred).all(axis=1).mean()
    
    avg_accuracy_hybrid = 0
    for i, col in enumerate(y_test.columns):
        acc = accuracy_score(y_test[col], hybrid_pred[:, i])
        avg_accuracy_hybrid += acc
    
    avg_accuracy_hybrid /= len(y_test.columns)
    
    print(f"\nHybrid-Modell Ergebnisse:")
    print(f"Exact Match Ratio: {exact_match_hybrid:.4f}")
    print(f"Durchschnittliche Accuracy: {avg_accuracy_hybrid:.4f}")
    
    # Finale Zusammenfassung
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG ALLER ANSÄTZE")
    print("="*80)
    
    results = [
        ("Multi-Label Deep Learning", acc1, em1),
        ("Super-Ensemble", acc2, 0),  # em nicht berechnet für Ensemble
        ("Hybrid-Modell", avg_accuracy_hybrid, exact_match_hybrid)
    ]
    
    best_acc = 0
    best_model = ""
    
    for name, acc, em in results:
        print(f"\n{name}:")
        print(f"  Durchschnittliche Accuracy: {acc:.2%}")
        if em > 0:
            print(f"  Exact Match Ratio: {em:.2%}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = name
    
    print("\n" + "="*80)
    print(f"BESTES MODELL: {best_model}")
    print(f"Beste Accuracy: {best_acc:.2%}")
    print("="*80)
    
    
    # Visualisierung
    plt.figure(figsize=(10, 6))
    model_names = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    
    plt.bar(model_names, accuracies)
    plt.axhline(y=0.8, color='r', linestyle='--', label='Ziel (80%)')
    plt.ylabel('Durchschnittliche Accuracy')
    plt.title('Deep Learning Ansätze - Vergleich')
    plt.legend()
    plt.tight_layout()
    plt.savefig('deep_learning_comparison.png')
    plt.close()
    
    print("\nAnalyse abgeschlossen.")

if __name__ == "__main__":
    main()