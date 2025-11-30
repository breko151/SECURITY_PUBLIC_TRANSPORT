# Bibliotecas
import duckdb
import pandas as pd
from pathlib import Path

# Configuración de la base de datos
ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data/transport.duckdb"

def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)

def query_top_stations_affluence_trends(transport: str, level_div: str, filter_div: list, weekday: str, week_year: str, n: int):
    conn = get_connection()
    
    if not filter_div:
        Query = f"""
                SELECT view_aflu.nombre, view_aflu.linea, AVG(view_aflu.afluencia) as afluencia_promedio
                FROM (
                    SELECT est.nombre, est.linea, tiem.anio, tiem.semana_anio, afl.afluencia
                    FROM ftb_afluencia_estaciones AS afl
                    JOIN dim_estaciones AS est ON afl.cve_est = est.cve_est
                    JOIN dim_espacio AS esp ON esp.id_espacio = est.id_espacio
                    JOIN dim_tiempo AS tiem ON tiem.id_tiempo = afl.id_tiempo
                    WHERE est.sistema = '{transport}' AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}'
                ) as view_aflu
                GROUP BY view_aflu.nombre, view_aflu.linea
                ORDER BY afluencia_promedio DESC
                LIMIT {n};
                """
    else:
        if level_div == 'Línea':
            filter_div = ['L' + elem.split()[-1] for elem in filter_div]
        filter_div = [f"'{elem}'" for elem in filter_div]
        filter_values_in = ", ".join(filter_div)
        
        if level_div == 'Alcaldía':
            filter_div_in_str = f'esp.alcaldia IN ({filter_values_in})'
        elif level_div == 'Línea':
            filter_div_in_str = f'est.linea IN ({filter_values_in})'
        elif level_div == 'Zona':
            filter_div_in_str = f'esp.zona IN ({filter_values_in})'
        
        print('\n\n\nFILTRO')
        print(filter_div_in_str)

        Query = f"""
                SELECT view_aflu.nombre, view_aflu.linea, AVG(view_aflu.afluencia) as afluencia_promedio
                FROM (
                    SELECT est.nombre, est.linea, tiem.anio, tiem.semana_anio, afl.afluencia
                    FROM ftb_afluencia_estaciones AS afl
                    JOIN dim_estaciones AS est ON afl.cve_est = est.cve_est
                    JOIN dim_espacio AS esp ON esp.id_espacio = est.id_espacio
                    JOIN dim_tiempo AS tiem ON tiem.id_tiempo = afl.id_tiempo
                    WHERE est.sistema = '{transport}' AND {filter_div_in_str} AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}'
                ) as view_aflu
                GROUP BY view_aflu.nombre, view_aflu.linea
                ORDER BY afluencia_promedio DESC
                LIMIT {n};
                """
    
    df = conn.execute(Query).df()
    conn.close()
    return df


def query_top_stations_crime_trends(transport: str, level_div: str, filter_div: list, sex: str, weekday: str, week_year: str, radio: float, n: int):
    conn = get_connection()

    if not filter_div:
        if sex == 'Ambos':
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
                                FROM dim_estaciones
                                WHERE sistema = '{transport}') AS unique_stations
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
                            WHERE est.sistema = '{transport}' AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND carpetas.dist_delito_estacion <= {radio}
                            GROUP BY tiem.anio, est.nombre, est.linea) AS delitos
                        ON main.anio = delitos.anio AND main.nombre = delitos.nombre AND main.linea = delitos.linea
                    ) AS final
                    GROUP BY 
                        final.linea, 
                        final.nombre
                    ORDER BY 
                        promedio_delitos DESC
                    LIMIT {n}
                    """
        else:
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
                                FROM dim_estaciones
                                WHERE sistema = '{transport}') AS unique_stations
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
                            WHERE est.sistema = '{transport}' AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND carpetas.dist_delito_estacion <= {radio} AND sex.sexo_victima = '{sex}'
                            GROUP BY tiem.anio, est.nombre, est.linea) AS delitos
                        ON main.anio = delitos.anio AND main.nombre = delitos.nombre AND main.linea = delitos.linea
                    ) AS final
                    GROUP BY 
                        final.linea, 
                        final.nombre
                    ORDER BY 
                        promedio_delitos DESC
                    LIMIT {n}
                    """
    else:
        if level_div == 'Línea':
            filter_div = ['L' + elem.split()[-1] for elem in filter_div]
        filter_div = [f"'{elem}'" for elem in filter_div]
        filter_values_in = ", ".join(filter_div)
        if level_div == 'Alcaldía':
            filter_div_in_str = f'esp.alcaldia IN ({filter_values_in})'
        elif level_div == 'Línea':
            filter_div_in_str = f'est.linea IN ({filter_values_in})'
        elif level_div == 'Zona':
            filter_div_in_str = f"esp.zona IN ({filter_values_in})"
        print('\n\n\nFILTRO')
        print(filter_div_in_str)
        
        if sex == 'Ambos':
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
                                WHERE est.sistema = '{transport}' AND {filter_div_in_str}) AS unique_stations
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
                            WHERE est.sistema = '{transport}' AND {filter_div_in_str} AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND carpetas.dist_delito_estacion <= {radio}
                            GROUP BY tiem.anio, est.nombre, est.linea) AS delitos
                        ON main.anio = delitos.anio AND main.nombre = delitos.nombre AND main.linea = delitos.linea
                    ) AS final
                    GROUP BY 
                        final.linea, 
                        final.nombre
                    ORDER BY 
                        promedio_delitos DESC
                    LIMIT {n}
                    """
        else:
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
                                WHERE est.sistema = '{transport}' AND {filter_div_in_str}) AS unique_stations
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
                            WHERE est.sistema = '{transport}' AND {filter_div_in_str} AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND carpetas.dist_delito_estacion <= {radio} AND sex.sexo_victima = '{sex}'
                            GROUP BY tiem.anio, est.nombre, est.linea) AS delitos
                        ON main.anio = delitos.anio AND main.nombre = delitos.nombre AND main.linea = delitos.linea
                    ) AS final
                    GROUP BY 
                        final.linea, 
                        final.nombre
                    ORDER BY 
                        promedio_delitos DESC
                    LIMIT {n}
                    """
            
    df = conn.execute(Query).df()
    conn.close()
    return df

def query_top_crimes_historical(transport: str, cve_est: str, radio: float, n: int):
    conn = get_connection()
    Query = f"""
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
            WHERE est.sistema = '{transport}' AND est.cve_est = '{cve_est}' AND carpetas.dist_delito_estacion <= {radio}
            GROUP BY est.cve_est, est.linea, est.nombre, del.clase_cndfe_snieg_2018
            ORDER BY conteo_delitos DESC
            LIMIT {n}
            """
    df = conn.execute(Query).df()
    conn.close()
    return df

def query_crimes_exploration_gender(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = get_connection()
    Query = f"""
            SELECT
                sex.sexo_victima,
                COUNT(carpetas.id_delito) AS conteo_delitos
            FROM ftb_carpetas_investigacion_fgj as carpetas
            JOIN dim_delitos AS del ON del.id_delito = carpetas.id_delito
            JOIN dim_estaciones AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN dim_tiempo AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            JOIN dim_sexo_victima AS sex ON sex.id_sexo = carpetas.id_sexo
            WHERE est.sistema = '{transport}' AND est.cve_est = '{cve_est}' AND carpetas.dist_delito_estacion <= {radio}
            AND tiem.dia_semana = '{weekday}'
            AND del.variable_cndfe_snieg_2018 = '{crime_var}'
            GROUP BY sex.sexo_victima
            ORDER BY conteo_delitos DESC
            """
    df = conn.execute(Query).df()
    conn.close()
    return df

def query_crimes_exploration_age_group(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = get_connection()
    Query = f"""
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
                WHERE est.sistema = '{transport}' 
                AND est.cve_est = '{cve_est}' 
                AND carpetas.dist_delito_estacion <= {radio}
                AND tiem.dia_semana = '{weekday}'
                AND del.variable_cndfe_snieg_2018 = '{crime_var}'
                GROUP BY age.grupo_quinquenal_inegi
            )
            SELECT 
                u.grupo_quinquenal_inegi,
                COALESCE(c.conteo_delitos, 0) AS conteo_delitos
            FROM unique_grupos u
            LEFT JOIN conteos c ON u.grupo_quinquenal_inegi = c.grupo_quinquenal_inegi
            ORDER BY u.grupo_quinquenal_inegi;
            """
    df = conn.execute(Query).df()
    conn.close()
    return df

def query_crimes_exploration_distances(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = get_connection()
    Query = f"""
            SELECT
                carpetas.dist_delito_estacion AS distancia
            FROM ftb_carpetas_investigacion_fgj as carpetas
            JOIN dim_delitos AS del ON del.id_delito = carpetas.id_delito
            JOIN dim_estaciones AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN dim_tiempo AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            WHERE est.sistema = '{transport}' AND est.cve_est = '{cve_est}' AND carpetas.dist_delito_estacion <= {radio}
            AND tiem.dia_semana = '{weekday}'
            AND del.variable_cndfe_snieg_2018 = '{crime_var}'
            ORDER BY distancia
            """
    df = conn.execute(Query).df()
    conn.close()
    return df

def query_crimes_part_of_day(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = get_connection()
    # Simulating parts of day based on hour since dim_fases_dia is missing
    Query = f"""
            SELECT
                CASE 
                    WHEN tiem.hora BETWEEN 6 AND 11 THEN 'Mañana'
                    WHEN tiem.hora BETWEEN 12 AND 18 THEN 'Tarde'
                    WHEN tiem.hora BETWEEN 19 AND 23 THEN 'Noche'
                    ELSE 'Madrugada'
                END AS parte_dia,
                COUNT(carpetas.id_delito) AS conteo_delitos
            FROM ftb_carpetas_investigacion_fgj as carpetas
            JOIN dim_delitos AS del ON del.id_delito = carpetas.id_delito
            JOIN dim_estaciones AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN dim_tiempo AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            JOIN dim_sexo_victima AS sex ON sex.id_sexo = carpetas.id_sexo
            WHERE est.sistema = '{transport}' AND est.cve_est = '{cve_est}' AND carpetas.dist_delito_estacion <= {radio}
            AND tiem.dia_semana = '{weekday}'
            AND del.variable_cndfe_snieg_2018 = '{crime_var}'
            GROUP BY parte_dia
            ORDER BY conteo_delitos DESC
            """
    df = conn.execute(Query).df()
    conn.close()
    return df
