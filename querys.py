# Bibliotecas
import pyodbc
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

# Credenciales
load_dotenv()

SERVER = os.getenv('SERVER')
DATABASE = os.getenv('DATABASE')
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')

connectionString = (
    'DRIVER={ODBC Driver 17 for SQL Server};'
    f'SERVER={SERVER};'
    f'DATABASE={DATABASE};'
    f'UID={USERNAME};'
    f'PWD={PASSWORD};'
    'TrustServerCertificate=yes;'
)
conn = pyodbc.connect(connectionString)

def query_top_stations_affluence_trends(transport: str, level_div: str, sexo: str, filter_div: list, weekday: str, week_year: str, n: int):
    conn = pyodbc.connect(connectionString)
    if not filter_div:
        Query = f"""
                SELECT TOP {n} view_aflu.nombre, view_aflu.linea, AVG(view_aflu.afluencia) as afluencia_promedio
                FROM (
                    SELECT est.nombre, est.linea, tiem.anio, tiem.semana_anio, afl.afluencia
                    FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_afluencia_estaciones] AS afl
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON afl.cve_est = est.cve_est
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = afl.id_tiempo
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_sexo_victima] AS sex ON sex.id_sexo = afl.id_sexo
                    WHERE est.sistema = '{transport}' AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND sex.sexo_victima = '{sexo}'
                ) as view_aflu
                GROUP BY view_aflu.nombre, view_aflu.linea
                ORDER BY afluencia_promedio DESC;
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
                SELECT TOP {n} view_aflu.nombre, view_aflu.linea, AVG(view_aflu.afluencia) as afluencia_promedio
                FROM (
                    SELECT est.nombre, est.linea, tiem.anio, tiem.semana_anio, afl.afluencia
                    FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_afluencia_estaciones] AS afl
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON afl.cve_est = est.cve_est
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = afl.id_tiempo
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_sexo_victima] AS sex ON sex.id_sexo = afl.id_sexo
                    WHERE est.sistema = '{transport}' AND {filter_div_in_str} AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND sex.sexo_victima = '{sexo}'
                ) as view_aflu
                GROUP BY view_aflu.nombre, view_aflu.linea
                ORDER BY afluencia_promedio DESC;
                """
    cursor = conn.cursor()
    cursor.execute(Query)
    data = cursor.fetchall()

    columns = [column[0] for column in cursor.description]
    data_ = [tuple(row) for row in data]

    cursor.close()
    conn.close()

    return pd.DataFrame(data_, columns=columns)

def query_top_stations_crime_trends(transport: str, level_div: str, sexo: str, filter_div: list, weekday: str, week_year: str, radio: float, n: int):
    conn = pyodbc.connect(connectionString)

    if not filter_div:
        Query = f"""
                SELECT 
                    TOP {n} final.linea, 
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
                            FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones]
                            WHERE sistema = '{transport}') AS unique_stations
                        CROSS JOIN 
                            (SELECT anio
                            FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo]
                            WHERE anio BETWEEN 2019 AND 2023) AS tiem) AS main
                    LEFT JOIN 
                        (SELECT 
                            tiem.anio,
                            est.nombre,
                            est.linea,
                            COUNT(carpetas.id_delito) AS delitos
                        FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_carpetas_investigacion_fgj] as carpetas
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_delitos] AS del ON del.id_delito = carpetas.id_delito
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON est.cve_est = carpetas.cve_est_mas_cercana
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_sexo_victima] AS sex ON sex.id_sexo = carpetas.id_sexo
                        WHERE est.sistema = '{transport}' AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND carpetas.dist_delito_estacion <= {radio} AND sex.sexo_victima = '{sexo}'
                        GROUP BY tiem.anio, est.nombre, est.linea) AS delitos
                    ON main.anio = delitos.anio AND main.nombre = delitos.nombre AND main.linea = delitos.linea
                ) AS final
                GROUP BY 
                    final.linea, 
                    final.nombre
                ORDER BY 
                    promedio_delitos DESC
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

        Query = f"""
                SELECT 
                    TOP {n} final.linea, 
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
                            FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est
                            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                            WHERE est.sistema = '{transport}' AND {filter_div_in_str}) AS unique_stations
                        CROSS JOIN 
                            (SELECT anio
                            FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo]
                            WHERE anio BETWEEN 2019 AND 2023) AS tiem) AS main
                    LEFT JOIN 
                        (SELECT 
                            tiem.anio,
                            est.nombre,
                            est.linea,
                            COUNT(carpetas.id_delito) AS delitos
                        FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_carpetas_investigacion_fgj] as carpetas
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_delitos] AS del ON del.id_delito = carpetas.id_delito
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON est.cve_est = carpetas.cve_est_mas_cercana
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
                        JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_sexo_victima] AS sex ON sex.id_sexo = carpetas.id_sexo
                        WHERE est.sistema = '{transport}' AND {filter_div_in_str} AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND carpetas.dist_delito_estacion <= {radio} AND sex.sexo_victima = '{sexo}'
                        GROUP BY tiem.anio, est.nombre, est.linea) AS delitos
                    ON main.anio = delitos.anio AND main.nombre = delitos.nombre AND main.linea = delitos.linea
                ) AS final
                GROUP BY 
                    final.linea, 
                    final.nombre
                ORDER BY 
                    promedio_delitos DESC
                """
    cursor = conn.cursor()
    cursor.execute(Query)
    data = cursor.fetchall()

    columns = [column[0] for row in cursor.description]
    data_ = [tuple(row) for row in data]

    cursor.close()
    conn.close()

    return pd.DataFrame(data_, columns=columns)

def query_top_crimes_historical(transport: str, cve_est: str, radio: float, n: int):
    conn = pyodbc.connect(connectionString)
    Query = f"""
            SELECT TOP {n}
                est.cve_est,
                est.linea,
                est.nombre,
                del.clase_cndfe_snieg_2018 AS clase_delito,
                COUNT(carpetas.id_delito) AS conteo_delitos
            FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_carpetas_investigacion_fgj] as carpetas
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_delitos] AS del ON del.id_delito = carpetas.id_delito
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            WHERE est.sistema = '{transport}' AND est.cve_est = '{cve_est}' AND carpetas.dist_delito_estacion <= {radio}
            GROUP BY est.cve_est, est.linea, est.nombre, del.clase_cndfe_snieg_2018
            ORDER BY conteo_delitos DESC
            """
    cursor = conn.cursor()
    cursor.execute(Query)
    data = cursor.fetchall()

    columns = [column[0] for column in cursor.description]
    data_ = [tuple(row) for row in data]

    cursor.close()
    conn.close()

    return pd.DataFrame(data_, columns=columns)

def query_crimes_exploration_gender(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = pyodbc.connect(connectionString)
    Query = f"""
            SELECT
                sex.sexo_victima,
                COUNT(carpetas.id_delito) AS conteo_delitos
            FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_carpetas_investigacion_fgj] as carpetas
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_delitos] AS del ON del.id_delito = carpetas.id_delito
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_sexo_victima] AS sex ON sex.id_sexo = carpetas.id_sexo
            WHERE est.sistema = '{transport}' AND est.cve_est = '{cve_est}' AND carpetas.dist_delito_estacion <= {radio}
            AND tiem.dia_semana = '{weekday}'
            AND del.variable_cndfe_snieg_2018 = '{crime_var}'
            GROUP BY sex.sexo_victima
            ORDER BY conteo_delitos DESC
            """
    cursor = conn.cursor()
    cursor.execute(Query)
    data = cursor.fetchall()

    columns = [column[0] for column in cursor.description]
    data_ = [tuple(row) for row in data]

    cursor.close()
    conn.close()

    return pd.DataFrame(data_, columns=columns)

def query_crimes_exploration_age_group(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = pyodbc.connect(connectionString)
    Query = f"""
            WITH unique_grupos AS (
                SELECT DISTINCT grupo_quinquenal_inegi 
                FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_edad_victima]
            ),
            conteos AS (
                SELECT
                    age.grupo_quinquenal_inegi,
                    COUNT(carpetas.id_delito) AS conteo_delitos
                FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_carpetas_investigacion_fgj] as carpetas
                JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_delitos] AS del ON del.id_delito = carpetas.id_delito
                JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON est.cve_est = carpetas.cve_est_mas_cercana
                JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
                JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_edad_victima] AS age ON age.id_edad = carpetas.id_edad
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
    cursor = conn.cursor()
    cursor.execute(Query)
    data = cursor.fetchall()

    columns = [column[0] for column in cursor.description]
    data_ = [tuple(row) for row in data]

    cursor.close()
    conn.close()

    return pd.DataFrame(data_, columns=columns)

def query_crimes_exploration_distances(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = pyodbc.connect(connectionString)
    Query = f"""
            SELECT
                carpetas.dist_delito_estacion AS distancia
            FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_carpetas_investigacion_fgj] as carpetas
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_delitos] AS del ON del.id_delito = carpetas.id_delito
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            WHERE est.sistema = '{transport}' AND est.cve_est = '{cve_est}' AND carpetas.dist_delito_estacion <= {radio}
            AND tiem.dia_semana = '{weekday}'
            AND del.variable_cndfe_snieg_2018 = '{crime_var}'
            ORDER BY distancia
            """
    cursor = conn.cursor()
    cursor.execute(Query)
    data = cursor.fetchall()

    columns = [column[0] for column in cursor.description]
    data_ = [tuple(row) for row in data]

    cursor.close()
    conn.close()

    return pd.DataFrame(data_, columns=columns)

def query_crimes_part_of_day(transport: str, cve_est: str, radio: float, weekday: str, crime_var: str):
    conn = pyodbc.connect(connectionString)
    Query = f"""
            SELECT
                fase.parte_dia,
                COUNT(carpetas.id_delito) AS conteo_delitos
            FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_carpetas_investigacion_fgj] as carpetas
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_delitos] AS del ON del.id_delito = carpetas.id_delito
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON est.cve_est = carpetas.cve_est_mas_cercana
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_fases_dia] AS fase ON fase.id_fase_dia = tiem.id_fase_dia
            JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_sexo_victima] AS sex ON sex.id_sexo = carpetas.id_sexo
            WHERE est.sistema = '{transport}' AND est.cve_est = '{cve_est}' AND carpetas.dist_delito_estacion <= {radio}
            AND tiem.dia_semana = '{weekday}'
            AND del.variable_cndfe_snieg_2018 = '{crime_var}'
            GROUP BY fase.parte_dia
            ORDER BY conteo_delitos DESC
            """
    cursor = conn.cursor()
    cursor.execute(Query)
    data = cursor.fetchall()

    columns = [column[0] for column in cursor.description]
    data_ = [tuple(row) for row in data]

    cursor.close()
    conn.close()

    return pd.DataFrame(data_, columns=columns)