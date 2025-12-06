import streamlit as st
import pandas as pd
import geopandas as gpd
import pickle
import base64
import json
from src.config import dict_munics

@st.cache_data
def load_geodata():
    # Geodata
    df_stations = pd.read_csv("./data/fact_constellation_schema/dim_estaciones_espacio_ok.csv")
    df_stations['cve_mun_inegi'] = df_stations['cve_mun_inegi'].astype(str).str.zfill(3)
    metro_lines = gpd.read_file('./data/shapefiles/metro/STC_Metro_lineas_utm14n_repr.shp')
    mb_lines = gpd.read_file('./data/shapefiles/mb/Metrobus_lineas_utm14n_repr.shp')
    zones_gdf = gpd.read_file('./data/shapefiles/zonas/zonas_geo.shp')
    munics_gdf = gpd.read_file('./data/shapefiles/alcaldias/alcaldias_geo.shp')
    police_sectors_gdf = gpd.read_file('./data/shapefiles/cuadrantes/sectores_agrupados_ssc.shp')
    lineas_cdmx = gpd.read_file('./assets/images/cdmx.json', encoding='utf-8')
    lineas_cdmx_json = lineas_cdmx.to_json()

    # Adjusts to geodata
    mb_lines = mb_lines[mb_lines['LINEA'] != '01 y 02']
    mb_lines_fix = mb_lines.copy()
    mb_lines_fix['LINEA'] = mb_lines_fix['LINEA'].apply(lambda x: x.replace('0', 'L'))
    df_stations_metro = df_stations[df_stations['sistema'] == 'STC Metro']
    df_stations_metrobus = df_stations[df_stations['sistema'] == 'Metrobús']
    munics_gdf['NOMGEO'] = munics_gdf['NOMGEO'].map(dict_munics)

    return {
        "df_stations": df_stations,
        "metro_lines": metro_lines,
        "mb_lines": mb_lines,
        "zones_gdf": zones_gdf,
        "munics_gdf": munics_gdf,
        "police_sectors_gdf": police_sectors_gdf,
        "lineas_cdmx": lineas_cdmx,
        "lineas_cdmx_json": lineas_cdmx_json,
        "mb_lines_fix": mb_lines_fix,
        "df_stations_metro": df_stations_metro,
        "df_stations_metrobus": df_stations_metrobus
    }

@st.cache_data
def load_images():
    images = {}
    
    image_home_logo_url = "./assets/images/MapaCDMX.png"
    with open(image_home_logo_url, "rb") as file_image_home:
        contents = file_image_home.read()
        images["home_logo"] = base64.b64encode(contents).decode("utf-8")
        
    metro_logo_url = "./assets/images/logo_metro.png"
    with open(metro_logo_url, "rb") as file_metro_logo:
        contents = file_metro_logo.read()
        images["metro_logo"] = base64.b64encode(contents).decode("utf-8")
        
    metrobus_logo_url = "./assets/images/logo_metrobus.png"
    with open(metrobus_logo_url, "rb") as file_metrobus_logo:
        contents = file_metrobus_logo.read()
        images["metrobus_logo"] = base64.b64encode(contents).decode("utf-8")

    metro_map_url = "./assets/images/MAPA_METRO.png"
    with open(metro_map_url, "rb") as file_metro_map:
        contents = file_metro_map.read()
        images["metro_map"] = base64.b64encode(contents).decode("utf-8")

    metrobus_map_url = "./assets/images/MAPA_METROBUS.png"
    with open(metrobus_map_url, "rb") as file_metrobus_map:
        contents = file_metrobus_map.read()
        images["metrobus_map"] = base64.b64encode(contents).decode("utf-8")
        
    return images

@st.cache_resource
def load_crime_model(transport: str, grouped_dataset_id: int):
    if transport == 'STC Metro':
        with open('./models/models_trained/final/clf_crime_metro_dataset_{}_wm_2_mas_perc.pkl'.format(grouped_dataset_id), 'rb') as file:
            loaded_pipeline = pickle.load(file)
    else:
        with open('./models/models_trained/final/clf_crime_metrobus_dataset_{}_wm_2_mas_perc.pkl'.format(grouped_dataset_id), 'rb') as file:
            loaded_pipeline = pickle.load(file)
    
    return loaded_pipeline

@st.cache_data
def load_afflu_forecast_values(transport: str, grouped_dataset_id: int):
    if transport == 'STC Metro':
        if grouped_dataset_id in {3, 4}:
            df = pd.read_csv('./data/predictions_sarima/predicciones_afluencia_alcaldia_semana_metro.csv')
        elif grouped_dataset_id in {6, 7}:
            df = pd.read_csv('./data/predictions_sarima/predicciones_afluencia_sector_policial_semana_metro.csv')
    elif grouped_dataset_id in {3, 4}:
        df = pd.read_csv('./data/predictions_sarima/predicciones_afluencia_alcaldia_semana_metrobus_final.csv')
    elif grouped_dataset_id in {6, 7}:
        df = pd.read_csv('./data/predictions_sarima/predicciones_afluencia_sector_policial_semana_metrobus.csv')

    return df

@st.cache_data
def load_weekly_crime_counts(transport: str, grouped_dataset_id: int):
    if transport == 'STC Metro':
        df = pd.read_csv(
            f'./data/datasets_aux/test/carpetas_afluencia_metro_grupo_{grouped_dataset_id}_wm_final_red.csv'
        )
    else:
        df = pd.read_csv(
            f'./data/datasets_aux/test/carpetas_afluencia_metrobus_grupo_{grouped_dataset_id}_wm_final_red.csv'
        )

    df['semana_anio_completa'] = df['anio'].astype(str) + ' - s' + df['semana_anio'].astype(str)

    return df

@st.cache_data
def load_thresholds_crime_model(transport: str, grouped_dataset_id: int):
    if transport == 'STC Metro':
        df = pd.read_csv('./data/datasets_aux/test/rangos_dataset_grupo_{}_2_mas_perc.csv'.format(grouped_dataset_id))
    else:
        df = pd.read_csv('./data/datasets_aux/test/rangos_dataset_grupo_{}_2_mas_perc_mb.csv'.format(grouped_dataset_id))
    
    return df
