import pandas as pd
from pathlib import Path
import sys

def validate_data(data_dir: Path):
    print("Validating data files...")
    
    # Check for required files
    required_files = [
        "test/carpetas_afluencia_metro_grupo_3_wm_final_red.csv",
        "test/rangos_dataset_grupo_3_2_mas_perc.csv"
        # Add others as needed
    ]
    
    for f in required_files:
        p = data_dir / f
        if not p.exists():
            print(f"WARNING: Missing file {f}")
        else:
            print(f"OK: {f}")
            
            # Basic content check
            try:
                df = pd.read_csv(p)
                if df.empty:
                    print(f"  ERROR: File is empty {f}")
                else:
                    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
                    if df.isnull().any().any():
                        print("  WARNING: Contains null values")
                        print(df.isnull().sum())
            except Exception as e:
                print(f"  ERROR: Could not read {f}: {e}")

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data/datasets_aux"
    validate_data(data_dir)
