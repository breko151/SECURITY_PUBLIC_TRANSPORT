import pytest
import pandas as pd
import duckdb
from pathlib import Path
from src.querys import (
    query_top_stations_affluence_trends,
    query_top_stations_crime_trends,
    query_top_crimes_historical,
    query_crimes_exploration_gender,
    query_crimes_exploration_age_group,
    query_crimes_exploration_distances,
    query_crimes_part_of_day
)

# Constants for testing
TRANSPORT = 'STC Metro'
WEEKDAY = 'Lunes'
WEEK_YEAR = '10'
YEAR = 2023
RADIO = 500.0
N = 5
CRIME_VAR = 'Robo a transeúnte'

@pytest.fixture(scope="module")
def db_connection():
    root_dir = Path(__file__).resolve().parents[1]
    db_path = root_dir / "data/transport.duckdb"
    conn = duckdb.connect(str(db_path), read_only=True)
    yield conn
    conn.close()

@pytest.fixture(scope="module")
def valid_station_id(db_connection):
    # Fetch a valid station ID from the database
    query = f"SELECT cve_est FROM dim_estaciones WHERE sistema = '{TRANSPORT}' LIMIT 1"
    result = db_connection.execute(query).fetchone()
    if result:
        return result[0]
    else:
        pytest.fail("No valid station ID found in the database.")

def test_query_top_stations_affluence_trends():
    # Test case 1: No filter
    df = query_top_stations_affluence_trends(
        transport=TRANSPORT,
        level_div='Línea',
        filter_div=[],
        weekday=WEEKDAY,
        week_year=WEEK_YEAR,
        n=N
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = ['nombre', 'linea', 'afluencia_promedio']
    assert all(col in df.columns for col in expected_cols)
    
    # Data validation
    assert df['afluencia_promedio'].dtype in [float, int]
    assert (df['afluencia_promedio'] >= 0).all()
    assert len(df) <= N
    
    # Sort order validation
    assert df['afluencia_promedio'].is_monotonic_decreasing

    # Test case 2: With filter (Línea)
    df_filter = query_top_stations_affluence_trends(
        transport=TRANSPORT,
        level_div='Línea',
        filter_div=['Línea 1'],
        weekday=WEEKDAY,
        week_year=WEEK_YEAR,
        n=N
    )
    assert isinstance(df_filter, pd.DataFrame)

def test_query_top_stations_crime_trends():
    # Test case 1: Sex 'Ambos', No filter
    df = query_top_stations_crime_trends(
        transport=TRANSPORT,
        level_div='Línea',
        filter_div=[],
        sex='Ambos',
        weekday=WEEKDAY,
        week_year=WEEK_YEAR,
        radio=RADIO,
        n=N
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['linea', 'nombre', 'promedio_delitos']
    assert all(col in df.columns for col in expected_cols)
    
    # Data validation
    assert df['promedio_delitos'].dtype in [float, int, object] # Decimal might be object or float depending on driver
    # Convert to numeric if it's object/decimal for checking
    promedio_delitos = pd.to_numeric(df['promedio_delitos'])
    assert (promedio_delitos >= 0).all()
    assert len(df) <= N
    
    # Sort order validation
    assert promedio_delitos.is_monotonic_decreasing

    # Test case 2: Specific sex
    df_sex = query_top_stations_crime_trends(
        transport=TRANSPORT,
        level_div='Línea',
        filter_div=[],
        sex='Hombre',
        weekday=WEEKDAY,
        week_year=WEEK_YEAR,
        radio=RADIO,
        n=N
    )
    assert isinstance(df_sex, pd.DataFrame)

def test_query_top_crimes_historical(valid_station_id):
    df = query_top_crimes_historical(
        transport=TRANSPORT,
        cve_est=valid_station_id,
        radio=RADIO,
        n=N
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['cve_est', 'linea', 'nombre', 'clase_delito', 'conteo_delitos']
    assert all(col in df.columns for col in expected_cols)
    
    # Data validation
    assert df['conteo_delitos'].dtype in [int, 'int64']
    assert (df['conteo_delitos'] >= 0).all()
    assert len(df) <= N
    
    # Sort order validation
    assert df['conteo_delitos'].is_monotonic_decreasing

def test_query_crimes_exploration_gender(valid_station_id):
    df = query_crimes_exploration_gender(
        transport=TRANSPORT,
        cve_est=valid_station_id,
        radio=RADIO,
        weekday=WEEKDAY,
        crime_var=CRIME_VAR
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['sexo_victima', 'conteo_delitos']
    assert all(col in df.columns for col in expected_cols)
    
    # Data validation
    assert df['conteo_delitos'].dtype in [int, 'int64']
    assert (df['conteo_delitos'] >= 0).all()
    
    # Sort order validation
    assert df['conteo_delitos'].is_monotonic_decreasing

def test_query_crimes_exploration_age_group(valid_station_id):
    df = query_crimes_exploration_age_group(
        transport=TRANSPORT,
        cve_est=valid_station_id,
        radio=RADIO,
        weekday=WEEKDAY,
        crime_var=CRIME_VAR
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['grupo_quinquenal_inegi', 'conteo_delitos']
    assert all(col in df.columns for col in expected_cols)
    
    # Data validation
    assert df['conteo_delitos'].dtype in [int, 'int64']
    assert (df['conteo_delitos'] >= 0).all()

def test_query_crimes_exploration_distances(valid_station_id):
    df = query_crimes_exploration_distances(
        transport=TRANSPORT,
        cve_est=valid_station_id,
        radio=RADIO,
        weekday=WEEKDAY,
        crime_var=CRIME_VAR
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['distancia']
    assert all(col in df.columns for col in expected_cols)
    
    # Data validation
    assert df['distancia'].dtype in [float, int]
    assert (df['distancia'] >= 0).all()
    
    # Sort order validation (ASC)
    assert df['distancia'].is_monotonic_increasing

def test_query_crimes_part_of_day(valid_station_id):
    df = query_crimes_part_of_day(
        transport=TRANSPORT,
        cve_est=valid_station_id,
        radio=RADIO,
        weekday=WEEKDAY,
        crime_var=CRIME_VAR
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['parte_dia', 'conteo_delitos']
    assert all(col in df.columns for col in expected_cols)
    
    # Data validation
    assert df['conteo_delitos'].dtype in [int, 'int64']
    assert (df['conteo_delitos'] >= 0).all()
    
    # Sort order validation
    assert df['conteo_delitos'].is_monotonic_decreasing
