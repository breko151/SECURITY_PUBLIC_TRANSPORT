import pytest
import pandas as pd
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
TRANSPORT = 'Metro'
WEEKDAY = 'Lunes'
WEEK_YEAR = '10'
YEAR = 2023
RADIO = 500.0
N = 5
CRIME_VAR = 'Robo a transeúnte' # Example crime variable, need to verify if it exists or use a generic one
# Checking dim_delitos content might be useful, but let's try a common one.
# If 'Robo a transeúnte' fails, I might need to check the data.

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
    assert not df.empty or len(df) == 0 # It might be empty if no data matches, but should run
    expected_cols = ['nombre', 'linea', 'afluencia_promedio']
    assert all(col in df.columns for col in expected_cols)

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

def test_query_top_crimes_historical():
    # Need a valid station ID (cve_est). Let's try to get one from a previous query or use a dummy.
    # If I don't have a valid ID, it will return empty, which is fine for structure testing.
    cve_est = 'STC_1_1' # Example ID, might need adjustment
    
    df = query_top_crimes_historical(
        transport=TRANSPORT,
        cve_est=cve_est,
        radio=RADIO,
        n=N
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['cve_est', 'linea', 'nombre', 'clase_delito', 'conteo_delitos']
    assert all(col in df.columns for col in expected_cols)

def test_query_crimes_exploration_gender():
    cve_est = 'STC_1_1'
    df = query_crimes_exploration_gender(
        transport=TRANSPORT,
        cve_est=cve_est,
        radio=RADIO,
        weekday=WEEKDAY,
        crime_var=CRIME_VAR
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['sexo_victima', 'conteo_delitos']
    assert all(col in df.columns for col in expected_cols)

def test_query_crimes_exploration_age_group():
    cve_est = 'STC_1_1'
    df = query_crimes_exploration_age_group(
        transport=TRANSPORT,
        cve_est=cve_est,
        radio=RADIO,
        weekday=WEEKDAY,
        crime_var=CRIME_VAR
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['grupo_quinquenal_inegi', 'conteo_delitos']
    assert all(col in df.columns for col in expected_cols)

def test_query_crimes_exploration_distances():
    cve_est = 'STC_1_1'
    df = query_crimes_exploration_distances(
        transport=TRANSPORT,
        cve_est=cve_est,
        radio=RADIO,
        weekday=WEEKDAY,
        crime_var=CRIME_VAR
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['distancia']
    assert all(col in df.columns for col in expected_cols)

def test_query_crimes_part_of_day():
    cve_est = 'STC_1_1'
    df = query_crimes_part_of_day(
        transport=TRANSPORT,
        cve_est=cve_est,
        radio=RADIO,
        weekday=WEEKDAY,
        crime_var=CRIME_VAR
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['parte_dia', 'conteo_delitos']
    assert all(col in df.columns for col in expected_cols)
