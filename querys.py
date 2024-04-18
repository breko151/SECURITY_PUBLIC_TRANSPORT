# Bibliotecas
import pyodbc
import pandas as pd
from datetime import datetime

# Credenciales
#SERVER = ' 2806:2f0:9260:a1d8:be8c:9f51:81bb:a15, 1433'
SERVER = '2806:2f0:93a0:f3e1:dab5:85bb:2b96:66d5, 1433'
DATABASE = 'crimen_equip_urbano_afluencia_metro_metrobus_cdmx'
USERNAME = 'vaps2'
PASSWORD = 'hola3311'

connectionString = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + SERVER + ';DATABASE=' + DATABASE + ';UID=' + USERNAME + ';PWD=' + PASSWORD
conn = pyodbc.connect(connectionString)

def query_top_stations_affluence_trends(transport: str, zone_city: str, weekday: str, week_year: str, n: int):
    conn = pyodbc.connect(connectionString)
    
    if zone_city == 'Ciudad de México':
        Query = f"""
            WITH AfluenciaFiltrada AS (
                SELECT TOP {n} est.nombre, est.linea , AVG(afl.id_afluencia) AS afluencia
                From [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est
                JOIN
                [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                JOIN
                [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_afluencia_estaciones] AS afl ON afl.cve_est = est.cve_est
                JOIN
                [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = afl.id_tiempo
                WHERE est.sistema = '{transport}' and tiem.dia_semana = '{weekday}' and tiem.semana_anio = '{week_year}'
                GROUP BY est.nombre, est.linea
                ORDER BY afluencia DESC
            )
            SELECT
                nombre,
                linea,
                AVG(afluencia) AS afluencia_promedio
            FROM
                AfluenciaFiltrada
            GROUP BY
                nombre,
                linea
            ORDER BY
                afluencia_promedio DESC
                """
    else:
        Query = f"""
            WITH AfluenciaFiltrada AS (
                SELECT TOP {n} est.nombre, est.linea, AVG(afl.id_afluencia) AS afluencia
                From [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est
                JOIN
                [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                JOIN
                [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_afluencia_estaciones] AS afl ON afl.cve_est = est.cve_est
                JOIN
                [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = afl.id_tiempo
                WHERE est.sistema = '{transport}' and esp.zona = '{zone_city}' and tiem.dia_semana = '{weekday}' and tiem.semana_anio = '{week_year}'
                GROUP BY est.nombre, est.linea
                ORDER BY afluencia DESC
            )
            SELECT
                nombre,
                linea,
                AVG(afluencia) AS afluencia_promedio
            FROM
                AfluenciaFiltrada
            GROUP BY
                nombre,
                linea
            ORDER BY
                afluencia_promedio DESC
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


def query_top_stations_crime_trends(transport: str, zone_city: str, weekday: str, week_year: str, radio: float, n: int):
    conn = pyodbc.connect(connectionString)
    
    if zone_city == 'Ciudad de México':
        Query = f"""
                WITH DelitosPorEstacion AS (
                    SELECT 
                    tiem.anio,
                    tiem.semana_anio,
                    est.nombre,
                    est.linea,
                    count(carpetas.id_delito) AS delitos
                    from [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_carpetas_investigacion_fgj] as carpetas
                    JOIN
                        [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_delitos] AS del ON del.id_delito = carpetas.id_delito
                    JOIN
                        [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON est.cve_est = carpetas.cve_est_mas_cercana
                    JOIN
                        [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                    JOIN
                        [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
                    WHERE
                        est.sistema = '{transport}' and tiem.dia_semana = '{weekday}' and tiem.semana_anio = '{week_year}' and carpetas.dist_delito_estacion <= {radio}
                    Group by
                        tiem.anio,
                        tiem.semana_anio,
                        est.nombre,
                        est.linea
                )
                SELECT
                    TOP {n} nombre, linea,
                    AVG(delitos) AS promedio_delitos
                FROM
                    DelitosPorEstacion
                GROUP BY
                    nombre, linea
                ORDER BY
                    promedio_delitos DESC
                """
    else:
        Query = f"""
                WITH DelitosPorEstacion AS (
                    SELECT 
                    tiem.anio,
                    tiem.semana_anio,
                    est.nombre,
                    est.linea,
                    count(carpetas.id_delito) AS delitos
                    from [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[ftb_carpetas_investigacion_fgj] as carpetas
                    JOIN
                        [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_delitos] AS del ON del.id_delito = carpetas.id_delito
                    JOIN
                        [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_estaciones] AS est ON est.cve_est = carpetas.cve_est_mas_cercana
                    JOIN
                        [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_espacio] AS esp ON esp.id_espacio = est.id_espacio
                    JOIN
                        [crimen_equip_urbano_afluencia_metro_metrobus_cdmx].[dbo].[dim_tiempo] AS tiem ON tiem.id_tiempo = carpetas.id_tiempo
                    WHERE
                        est.sistema = '{transport}' and esp.zona = '{zone_city}' and tiem.dia_semana = '{weekday}' and tiem.semana_anio = '{week_year}' and carpetas.dist_delito_estacion <= {radio}
                    Group by
                        tiem.anio,
                        tiem.semana_anio,
                        est.nombre,
                        est.linea
                )
                SELECT TOP {n} nombre, linea,
                    AVG(delitos) AS promedio_delitos
                FROM
                    DelitosPorEstacion
                GROUP BY
                    nombre, linea
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