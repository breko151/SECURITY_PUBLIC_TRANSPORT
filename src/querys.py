# Bibliotecas
import duckdb
import pandas as pd
from pathlib import Path

# Configuración de la base de datos
ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data/transport.duckdb"

# At module level
PARTS_OF_DAY = {
    'Mañana': (6, 11),
    'Tarde': (12, 18),
    'Noche': (19, 23)
}

def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)

def query_top_stations_affluence_trends(transport: str, level_div: str, filter_div: list, weekday: str, week_year: str, n: int):
    conn = get_connection()
    
    params = [transport, weekday, week_year]
    
    if not filter_div:
        Query = """
                SELECT view_aflu.nombre, view_aflu.linea, AVG(view_aflu.afluencia) as afluencia_promedio
                FROM (
                    SELECT est.nombre, est.linea, tiem.anio, tiem.semana_anio, afl.afluencia
                    FROM ftb_afluencia_estaciones AS afl
                    JOIN dim_estaciones AS est ON afl.cve_est = est.cve_est
                    JOIN dim_espacio AS esp ON esp.id_espacio = est.id_espacio
                    JOIN dim_tiempo AS tiem ON tiem.id_tiempo = afl.id_tiempo
                    WHERE est.sistema = ? AND tiem.dia_semana = ? AND tiem.semana_anio = ?
                ) as view_aflu
                GROUP BY view_aflu.nombre, view_aflu.linea
                ORDER BY afluencia_promedio DESC
                LIMIT ?;
                """
        params.append(n)
    else:
        if level_div == 'Línea':
            filter_div = ['L' + elem.split()[-1] for elem in filter_div]
        
        # Create placeholders for IN clause
        placeholders = ', '.join(['?'] * len(filter_div))
        
        filter_clause = ""
        if level_div == 'Alcaldía':
            filter_clause = f'esp.alcaldia IN ({placeholders})'
        elif level_div == 'Línea':
            filter_clause = f'est.linea IN ({placeholders})'
        elif level_div == 'Zona':
            filter_clause = f'esp.zona IN ({placeholders})'
        
        # Insert filter parameters before weekday/week_year
        # Current params: [transport, weekday, week_year]
        # We need: [transport, *filter_div, weekday, week_year, n]
        
        params = [transport] + filter_div + [weekday, week_year, n]

        Query = f"""
                SELECT view_aflu.nombre, view_aflu.linea, AVG(view_aflu.afluencia) as afluencia_promedio
                FROM (
                    SELECT est.nombre, est.linea, tiem.anio, tiem.semana_anio, afl.afluencia
                    FROM ftb_afluencia_estaciones AS afl
                    JOIN dim_estaciones AS est ON afl.cve_est = est.cve_est
                    JOIN dim_espacio AS esp ON esp.id_espacio = est.id_espacio
                    JOIN dim_tiempo AS tiem ON tiem.id_tiempo = afl.id_tiempo
                    WHERE est.sistema = ? AND {filter_clause} AND tiem.dia_semana = ? AND tiem.semana_anio = ?
                ) as view_aflu
                GROUP BY view_aflu.nombre, view_aflu.linea
                ORDER BY afluencia_promedio DESC
                LIMIT ?;
                """
    
    df = conn.execute(Query, params).df()
    conn.close()
    return df


def query_top_stations_crime_trends(transport: str, level_div: str, filter_div: list, sex: str, weekday: str, week_year: str, radio: float, n: int):
    conn = get_connection()

    # Base parameters
    params = [transport]
    
    # Filter logic
    filter_clause = ""
    if filter_div:
        if level_div == 'Línea':
            filter_div = ['L' + elem.split()[-1] for elem in filter_div]
        
        placeholders = ', '.join(['?'] * len(filter_div))
        
        if level_div == 'Alcaldía':
            filter_clause = f'AND esp.alcaldia IN ({placeholders})'
        elif level_div == 'Línea':
            filter_clause = f'AND est.linea IN ({placeholders})'
        elif level_div == 'Zona':
            filter_clause = f'AND esp.zona IN ({placeholders})'
            
        params.extend(filter_div)

    # Add remaining parameters for the main query part
    # Note: The query structure is complex with subqueries.
    # We need to be careful about parameter order.
    # Let's look at the query structure.
    # It uses `transport` and `filter_div` in the `unique_stations` subquery.
    # Then it uses `transport`, `filter_div`, `weekday`, `week_year`, `radio` in the `delitos` subquery.
    # And `sex` if applicable.
    
    # Wait, the original query repeated the filter logic in two places: `unique_stations` and `delitos` subquery.
    # So we need to pass the parameters twice if we use placeholders?
    # Yes, unless we use CTEs to simplify.
    # Let's rewrite with CTEs to make it cleaner and safer.
    
    # Actually, let's stick to the structure but parameterize correctly.
    # Params for unique_stations: [transport, *filter_div]
    # Params for delitos: [transport, *filter_div, weekday, week_year, radio] (+ sex if needed)
    
    # Constructing the full parameter list
    full_params = []
    
    # 1. unique_stations params
    full_params.append(transport)
    if filter_div:
        full_params.extend(filter_div)
        
    # 2. delitos params
    full_params.append(transport)
    if filter_div:
        full_params.extend(filter_div)
    
    full_params.append(weekday)
    full_params.append(week_year)
    full_params.append(radio)
    
    sex_clause = ""
    if sex != 'Ambos':
        sex_clause = "AND sex.sexo_victima = ?"
        full_params.append(sex)
        
    full_params.append(n) # LIMIT

    # Constructing the query string
    # We need to inject the filter_clause string (which contains placeholders)
    
    # unique_stations WHERE clause
    # WHERE est.sistema = ? {filter_clause}
    # Note: filter_clause starts with AND if it exists
    
    Query = f"""
            SELECT 
                final.linea, 
                final.nombre, 
                CAST(AVG(CAST(final.delitos AS FLOAT)) AS DECIMAL(10,2)) AS promedio_delitos
            FROM
                (
                SELECT 
                    main.linea, 
                    main.nombre, 
                    main.anio, 
                    COALESCE(delitos.delitos, 0) AS delitos
                FROM
                    (SELECT DISTINCT unique_stations.linea, unique_stations.nombre, tiem.anio
                    FROM
                        (SELECT linea, nombre
                        FROM dim_estaciones AS est
                        JOIN dim_espacio AS esp ON esp.id_espacio = est.id_espacio
                        WHERE est.sistema = ? {filter_clause}) AS unique_stations
                    CROSS JOIN 
                        (SELECT anio
                        FROM dim_tiempo
                        WHERE anio BETWEEN 2019 AND 2023) AS tiem) AS main
                LEFT JOIN 
                    (SELECT 
                        tiem.anio,
                        est.nombre,
                        est.linea,
                        COUNT(carpetas.id_delito) AS delitos
                    FROM ftb_carpetas_investigacion_fgj as carpetas
                    JOIN dim_delitos AS del ON del.id_delito = carpetas.id_delito
                    JOIN dim_estaciones AS est ON est.cve_est = carpetas.cve_est_mas_cercana
                    JOIN dim_espacio AS esp ON esp.id_espacio = est.id_espacio
                    JOIN dim_tiempo AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
                    JOIN dim_sexo_victima AS sex ON sex.id_sexo = carpetas.id_sexo
                    WHERE est.sistema = ? {filter_clause} AND tiem.dia_semana = ? AND tiem.semana_anio = ? AND carpetas.dist_delito_estacion <= ? {sex_clause}
                    GROUP BY tiem.anio, est.nombre, est.linea) AS delitos
                ON main.anio = delitos.anio AND main.nombre = delitos.nombre AND main.linea = delitos.linea
            ) AS final
            GROUP BY 
                final.linea, 
                final.nombre
            ORDER BY 
                promedio_delitos DESC
            LIMIT ?
            """
            
    df = conn.execute(Query, full_params).df()
    conn.close()
    return df

def query_top_crimes_historical(transport: str, cve_est: str, radio: float, n: int):
    conn = get_connection()
    Query = """
            SELECT 
                est.cve_est,
                est.linea,
                est.nombre,
                del.clase_cndfe_snieg_2018 AS clase_delito,
                COUNT(carpetas.id_delito) AS conteo_delitos
            FROM ftb_carpetas_investigacion_fgj as carpetas
            JOIN dim_delitos AS del ON del.id_delito = carpetas.id_delito
            JOIN dim_estaciones AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN dim_tiempo AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            WHERE est.sistema = ? AND est.cve_est = ? AND carpetas.dist_delito_estacion <= ?
            GROUP BY est.cve_est, est.linea, est.nombre, del.clase_cndfe_snieg_2018
            ORDER BY conteo_delitos DESC
            LIMIT ?
            """
    df = conn.execute(Query, [transport, cve_est, radio, n]).df()
    conn.close()
    return df

def query_crimes_exploration_gender(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = get_connection()
    Query = """
            SELECT
                sex.sexo_victima,
                COUNT(carpetas.id_delito) AS conteo_delitos
            FROM ftb_carpetas_investigacion_fgj as carpetas
            JOIN dim_delitos AS del ON del.id_delito = carpetas.id_delito
            JOIN dim_estaciones AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN dim_tiempo AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            JOIN dim_sexo_victima AS sex ON sex.id_sexo = carpetas.id_sexo
            WHERE est.sistema = ? AND est.cve_est = ? AND carpetas.dist_delito_estacion <= ?
            AND tiem.dia_semana = ?
            AND del.variable_cndfe_snieg_2018 = ?
            GROUP BY sex.sexo_victima
            ORDER BY conteo_delitos DESC
            """
    df = conn.execute(Query, [transport, cve_est, radio, weekday, crime_var]).df()
    conn.close()
    return df

def query_crimes_exploration_age_group(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = get_connection()
    Query = """
            WITH unique_grupos AS (
                SELECT DISTINCT grupo_quinquenal_inegi 
                FROM dim_edad_victima
            ),
            conteos AS (
                SELECT
                    age.grupo_quinquenal_inegi,
                    COUNT(carpetas.id_delito) AS conteo_delitos
                FROM ftb_carpetas_investigacion_fgj as carpetas
                JOIN dim_delitos AS del ON del.id_delito = carpetas.id_delito
                JOIN dim_estaciones AS est ON est.cve_est = carpetas.cve_est_mas_cercana
                JOIN dim_tiempo AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
                JOIN dim_edad_victima AS age ON age.id_edad = carpetas.id_edad
                WHERE est.sistema = ? 
                AND est.cve_est = ? 
                AND carpetas.dist_delito_estacion <= ?
                AND tiem.dia_semana = ?
                AND del.variable_cndfe_snieg_2018 = ?
                GROUP BY age.grupo_quinquenal_inegi
            )
            SELECT 
                u.grupo_quinquenal_inegi,
                COALESCE(c.conteo_delitos, 0) AS conteo_delitos
            FROM unique_grupos u
            LEFT JOIN conteos c ON u.grupo_quinquenal_inegi = c.grupo_quinquenal_inegi
            ORDER BY u.grupo_quinquenal_inegi;
            """
    df = conn.execute(Query, [transport, cve_est, radio, weekday, crime_var]).df()
    conn.close()
    return df

def query_crimes_exploration_distances(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = get_connection()
    Query = """
            SELECT
                carpetas.dist_delito_estacion AS distancia
            FROM ftb_carpetas_investigacion_fgj as carpetas
            JOIN dim_delitos AS del ON del.id_delito = carpetas.id_delito
            JOIN dim_estaciones AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN dim_tiempo AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            WHERE est.sistema = ? AND est.cve_est = ? AND carpetas.dist_delito_estacion <= ?
            AND tiem.dia_semana = ?
            AND del.variable_cndfe_snieg_2018 = ?
            ORDER BY distancia
            """
    df = conn.execute(Query, [transport, cve_est, radio, weekday, crime_var]).df()
    conn.close()
    return df

def query_crimes_part_of_day(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = get_connection()
    
    # Use constants for time ranges
    morning_start, morning_end = PARTS_OF_DAY['Mañana']
    afternoon_start, afternoon_end = PARTS_OF_DAY['Tarde']
    night_start, night_end = PARTS_OF_DAY['Noche']
    
    # Simulating parts of day based on hour since dim_fases_dia is missing
    Query = """
            SELECT
                CASE 
                    WHEN tiem.hora BETWEEN ? AND ? THEN 'Mañana'
                    WHEN tiem.hora BETWEEN ? AND ? THEN 'Tarde'
                    WHEN tiem.hora BETWEEN ? AND ? THEN 'Noche'
                    ELSE 'Madrugada'
                END AS parte_dia,
                COUNT(carpetas.id_delito) AS conteo_delitos
            FROM ftb_carpetas_investigacion_fgj as carpetas
            JOIN dim_delitos AS del ON del.id_delito = carpetas.id_delito
            JOIN dim_estaciones AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN dim_tiempo AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            WHERE est.sistema = ? AND est.cve_est = ? AND carpetas.dist_delito_estacion <= ?
            AND tiem.dia_semana = ?
            AND del.variable_cndfe_snieg_2018 = ?
            GROUP BY parte_dia
            ORDER BY conteo_delitos DESC
            """
            
    params = [
        morning_start, morning_end,
        afternoon_start, afternoon_end,
        night_start, night_end,
        transport, cve_est, radio, weekday, crime_var
    ]
    
    df = conn.execute(Query, params).df()
    conn.close()
    return df
