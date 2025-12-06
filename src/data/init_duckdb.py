import duckdb
from pathlib import Path
import sys

def init_db():
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data/fact_constellation_schema"
    db_path = root_dir / "data/transport.duckdb"
    
    print(f"Initializing DuckDB at {db_path}...")
    con = duckdb.connect(str(db_path))
    
    # List of CSV files and their target table names
    files = {
        "dim_estaciones.csv": "dim_estaciones",
        "dim_espacio.csv": "dim_espacio",
        "dim_tiempo.csv": "dim_tiempo",
        "ftb_afluencia_estaciones.csv": "ftb_afluencia_estaciones",
        "ftb_carpetas_investigacion_fgj.csv": "ftb_carpetas_investigacion_fgj",
        "dim_delitos.csv": "dim_delitos",
        "dim_edad_victima.csv": "dim_edad_victima",
        "dim_sexo_victima.csv": "dim_sexo_victima"
    }
    
    for filename, table_name in files.items():
        file_path = data_dir / filename
        if file_path.exists():
            print(f"Loading {table_name} from {filename}...")
            # Create table directly from CSV with auto-detection
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
        else:
            print(f"WARNING: {filename} not found!")
            
    # Verify
    tables = con.execute("SHOW TABLES").fetchall()
    print("Tables created:", [t[0] for t in tables])
    
    con.close()
    print("Database initialization complete.")

if __name__ == "__main__":
    init_db()
