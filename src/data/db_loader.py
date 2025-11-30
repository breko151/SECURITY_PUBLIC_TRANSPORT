import os
import pandas as pd
import pyodbc
from pathlib import Path
from dotenv import load_dotenv
import sys

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

def get_connection(autocommit=False):
    load_dotenv()
    server = os.getenv('SERVER')
    database = os.getenv('DATABASE')
    username = os.getenv('USERNAME')
    password = os.getenv('PASSWORD')
    
    # Connect to master to create DB if needed
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER={server};'
        f'UID={username};'
        f'PWD={password};'
        'TrustServerCertificate=yes;'
    )
    if database and not autocommit:
        conn_str += f'DATABASE={database};'
        
    conn = pyodbc.connect(conn_str, autocommit=autocommit)
    return conn

def create_database_if_not_exists(db_name):
    print(f"Checking database {db_name}...")
    conn = get_connection(autocommit=True)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"SELECT name FROM master.sys.databases WHERE name = '{db_name}'")
        if not cursor.fetchone():
            print(f"Creating database {db_name}...")
            cursor.execute(f"CREATE DATABASE [{db_name}]")
        else:
            print(f"Database {db_name} already exists.")
    finally:
        cursor.close()
        conn.close()

def load_csv_to_sql(csv_path, table_name, conn):
    print(f"Loading {csv_path.name} into {table_name}...")
    df = pd.read_csv(csv_path)
    
    # Basic type mapping for T-SQL
    # This is a simplified loader. For production, use SQLAlchemy or bulk insert tools (bcp).
    cursor = conn.cursor()
    
    # Drop table if exists
    cursor.execute(f"IF OBJECT_ID('{table_name}', 'U') IS NOT NULL DROP TABLE {table_name}")
    
    # Create table
    cols = []
    for col, dtype in df.dtypes.items():
        sql_type = "NVARCHAR(MAX)"
        if "int" in str(dtype):
            sql_type = "INT"
        elif "float" in str(dtype):
            sql_type = "FLOAT"
        elif "datetime" in str(dtype):
            sql_type = "DATETIME"
        cols.append(f"[{col}] {sql_type}")
    
    create_stmt = f"CREATE TABLE {table_name} ({', '.join(cols)})"
    cursor.execute(create_stmt)
    
    # Insert data
    # Using executemany is faster than loop, but slower than bulk insert
    placeholders = ",".join(["?"] * len(df.columns))
    insert_stmt = f"INSERT INTO {table_name} VALUES ({placeholders})"
    
    data = df.where(pd.notnull(df), None).values.tolist()
    
    # Batch insert to avoid memory issues
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        cursor.executemany(insert_stmt, batch)
        print(f"  Inserted {min(i+batch_size, len(data))}/{len(data)} rows", end='\r')
    
    print()
    conn.commit()
    cursor.close()

def main():
    data_dir = Path(__file__).resolve().parents[2] / "data/fact_constellation_schema"
    db_name = "crimen_equip_urbano_afluencia_metro_metrobus_cdmx"
    
    # 1. Create DB
    create_database_if_not_exists(db_name)
    
    # 2. Load Tables
    conn = get_connection()
    
    files_to_load = [
        ("dim_estaciones.csv", "dim_estaciones"),
        ("dim_espacio.csv", "dim_espacio"),
        ("dim_tiempo.csv", "dim_tiempo"),
        ("ftb_afluencia_estaciones.csv", "ftb_afluencia_estaciones"),
        ("ftb_carpetas_investigacion_fgj.csv", "ftb_carpetas_investigacion_fgj")
    ]
    
    for filename, table_name in files_to_load:
        file_path = data_dir / filename
        if file_path.exists():
            load_csv_to_sql(file_path, table_name, conn)
        else:
            print(f"Warning: File {filename} not found.")
            
    conn.close()
    print("Database regeneration complete.")

if __name__ == "__main__":
    main()
