import pandas as pd
from pathlib import Path

def load_thresholds(system: str, group: int, base_path: Path) -> pd.DataFrame:
    """
    Load threshold dataframe for a given system and group.
    """
    suffix = "_mb" if system == "metrobus" else ""
    filename = f"rangos_dataset_grupo_{group}_2_mas_perc{suffix}.csv"
    file_path = base_path / "test" / filename
    
    if not file_path.exists():
        # Try looking in the parent directory if not in test
        file_path_alt = base_path / filename
        if file_path_alt.exists():
            return pd.read_csv(file_path_alt)
        raise FileNotFoundError(f"Threshold file not found: {file_path}")
        
    return pd.read_csv(file_path)

def apply_labeling(df: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """
    Apply High/Low labeling based on thresholds.
    """
    # Create a dictionary for faster lookup: {categ_delito: threshold}
    # Assuming 'percentil' column holds the threshold value
    thresh_dict = dict(zip(thresholds['categ_delito'], thresholds['percentil']))
    
    def get_label(row):
        cat = row['categoria_delito_adaptada']
        count = row['conteo']
        thresh = thresh_dict.get(cat)
        
        if thresh is None:
            return "Low" # Default
            
        if count > thresh:
            return "High"
        else:
            return "Low"

    df = df.copy()
    df['label'] = df.apply(get_label, axis=1)
    return df
