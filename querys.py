# Bibliotecas
import pyodbc
import pandas as pd
from datetime import datetime

# Credenciales
#SERVER = ' 2806:2f0:9260:a1d8:be8c:9f51:81bb:a15, 1433'
# Antiguas compu vaps
# SERVER = '2806:2f0:93a0:f3e1:dab5:85bb:2b96:66d5, 1433'
# DATABASE = 'crimen_equip_urbano_afluencia_metro_metrobus_cdmx'
# USERNAME = 'vaps2'
# PASSWORD = 'hola3311'

SERVER = '217.21.78.91, 1433'
DATABASE = 'crimen_equip_urbano_afluencia_metro_metrobus_cdmx'
USERNAME = 'braulio'
PASSWORD = 'Holas3312#'

connectionString = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + SERVER + ';DATABASE=' + DATABASE + ';UID=' + USERNAME + ';PWD=' + PASSWORD
conn = pyodbc.connect(connectionString)

def query_top_stations_affluence_trends(transport: str, level_div: str, filter_div: list, weekday: str, week_year: str, n: int):
    conn = pyodbc.connect(connectionString)
    
    if filter_div == []:
        Query = f"""
                SELECT TOP {n} view_aflu.nombre, view_aflu.linea, AVG(view_aflu.afluencia) as afluencia_promedio
                FROM (
                    SELECT est.nombre, est.linea, tiem.anio, tiem.semana_anio, afl.afluencia
                    FROM [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_afluencia_estaciones] AS afl
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON afl.cve_est = est.cve_est
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                    JOIN [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = afl.id_tiempo
                    WHERE est.sistema = '{transport}' AND tiem.dia_semana = '{weekday}' and tiem.semana_anio = '{week_year}'
                ) as view_aflu
                GROUP BY view_aflu.nombre, view_aflu.linea
                ORDER BY afluencia_promedio DESC;
                """
    else:
        if level_div == 'Línea':
            filter_div = ['L' + elem.split()[-1] for elem in filter_div]
        filter_div = [f"\'{elem}\'" for elem in filter_div]
        filter_values_in = ", ".join(filter_div)
        if level_div == 'Zona':
            filter_div_in_str = f"esp.zona IN ({filter_values_in})"
        elif level_div == 'Alcaldía':
            filter_div_in_str = f'esp.alcaldia IN ({filter_values_in})'
        elif level_div == 'Línea':
            filter_div_in_str = f'est.linea IN ({filter_values_in})'
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
                    WHERE est.sistema = '{transport}' AND {filter_div_in_str} AND tiem.dia_semana = '{weekday}' and tiem.semana_anio = '{week_year}'
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
    
    df = pd.DataFrame(data_, columns=columns)
    
    return df


def query_top_stations_crime_trends(transport: str, level_div: str, filter_div: list, weekday: str, week_year: str, radio: float, n: int):
    conn = pyodbc.connect(connectionString)
    
    if filter_div == []:
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
                        WHERE est.sistema = '{transport}' AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND carpetas.dist_delito_estacion <= {radio}
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
        filter_div = [f"\'{elem}\'" for elem in filter_div]
        filter_values_in = ", ".join(filter_div)
        if level_div == 'Zona':
            filter_div_in_str = f"esp.zona IN ({filter_values_in})"
        elif level_div == 'Alcaldía':
            filter_div_in_str = f'esp.alcaldia IN ({filter_values_in})'
        elif level_div == 'Línea':
            filter_div_in_str = f'est.linea IN ({filter_values_in})'
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
                        WHERE est.sistema = '{transport}' AND {filter_div_in_str} AND tiem.dia_semana = '{weekday}' AND tiem.semana_anio = '{week_year}' AND carpetas.dist_delito_estacion <= {radio}
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
    
    columns = [column[0] for column in cursor.description]
    data_ = [tuple(row) for row in data]
    
    cursor.close()
    conn.close()
    
    df = pd.DataFrame(data_, columns=columns)
    
    return df
