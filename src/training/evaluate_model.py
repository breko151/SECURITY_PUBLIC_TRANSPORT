import argparse
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.labels import load_thresholds, apply_labeling

def evaluate_model(model_path, system, group, data_dir):
    print(f"Evaluating model: {model_path}")
    print(f"System: {system}, Group: {group}")
    
    # Load Model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    sys_name = "metro" if system == "metro" else "metrobus"
    
    # Load Data
    filename = f"carpetas_afluencia_{sys_name}_grupo_{group}_wm_final_red.csv"
    data_path = data_dir / "test" / filename
    
    if not data_path.exists():
        filename_alt = f"carpetas_afluencia_{sys_name}_grupo_{group}_wm.csv"
        data_path_alt = data_dir / filename_alt
        if data_path_alt.exists():
            data_path = data_path_alt
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
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
    
    # Predict
    print("Predicting on full dataset...")
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--system", type=str, required=True, choices=["metro", "metrobus"])
    parser.add_argument("--group", type=int, required=True, choices=[3, 4, 6, 7])
    parser.add_argument("--data_dir", type=str, default="data/datasets_aux")
    
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / args.data_dir
    
    evaluate_model(args.model_path, args.system, args.group, data_dir)
