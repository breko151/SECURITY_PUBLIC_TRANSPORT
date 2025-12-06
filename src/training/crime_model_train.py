import argparse
import pandas as pd
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys
import os
from datetime import datetime

# Add project root to path to import src modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.labels import load_thresholds, apply_labeling

def train_model(system, group, data_dir, models_dir, reports_dir):
    print(f"Training model for System: {system}, Group: {group}")
    
    sys_name = "metro" if system == "metro" else "metrobus"
    
    # Try to find the data file
    filename = f"carpetas_afluencia_{sys_name}_grupo_{group}_wm_final_red.csv"
    data_path = data_dir / "test" / filename
    
    if not data_path.exists():
        filename_alt = f"carpetas_afluencia_{sys_name}_grupo_{group}_wm.csv"
        data_path_alt = data_dir / filename_alt
        if data_path_alt.exists():
            data_path = data_path_alt
        else:
            raise FileNotFoundError(f"Data file not found: {data_path} or {data_path_alt}")
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Labeling
    print("Applying labeling...")
    thresholds = load_thresholds(system, group, data_dir)
    df = apply_labeling(df, thresholds)
    
    # Features
    features = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'semana_1']
    if group in [4, 7]:
        features.append('sexo_victima')
        
    X = df[features]
    y = df['label']
    
    # Split Train/Test (80/20)
    print("Splitting data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline
    cat_features = ['alcaldia', 'categoria_delito_adaptada']
    if 'sexo_victima' in features:
        cat_features.append('sexo_victima')
        
    num_features = ['semana_mes', 'semana_1']
    
    # Check if columns exist
    missing_cols = [c for c in features if c not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('num', MinMaxScaler(), num_features)
        ],
        remainder='passthrough' # Just in case, though we selected features
    )
    
    # Using DecisionTree but limiting depth to reduce overfitting
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_leaf=5))
    ])
    
    # Train
    print("Fitting model on TRAIN set...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating on TEST set...")
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report (Test Set):")
    print(report)
    
    # Save report
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"report_{sys_name}_{group}_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: DecisionTree (max_depth=10)\n")
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
        
    # Save Model with timestamp to avoid overwriting immediately
    models_dir_final = models_dir / "final"
    models_dir_final.mkdir(parents=True, exist_ok=True)
    
    model_filename = f"clf_crime_{sys_name}_dataset_{group}_wm_2_mas_perc_{timestamp}.pkl"
    save_path = models_dir_final / model_filename
    
    print(f"Saving new model candidate to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, required=True, choices=["metro", "metrobus"])
    parser.add_argument("--group", type=int, required=True, choices=[3, 4, 6, 7])
    parser.add_argument("--data_dir", type=str, default="data/datasets_aux")
    parser.add_argument("--models_dir", type=str, default="models/models_trained")
    parser.add_argument("--reports_dir", type=str, default="reports/training")
    
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / args.data_dir
    models_dir = root_dir / args.models_dir
    reports_dir = root_dir / args.reports_dir
    
    train_model(args.system, args.group, data_dir, models_dir, reports_dir)
