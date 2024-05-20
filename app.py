# LIBRARIES NEEDED

# Web.
import streamlit as st
from streamlit_folium import st_folium
from streamlit_plotly_events import plotly_events
import base64

# Data.
import numpy as np
import pandas as pd
import json
from math import ceil
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import LineString
import folium
import plotly.graph_objects as go
from shapely.geometry import MultiPolygon, MultiPoint, Polygon, shape

# Model
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Data visualization.
# import seaborn as sns
# import matplotlib.pyplot as plt

# Datetime
from datetime import datetime, timedelta

# Constants
from colors import LINESM, LINESMB, LINESM_aux, LINESMB_aux

# Auxiliar proper modules
from plots import plot_top_stations_affluence_trends, plot_top_stations_crime_trends, plot_top_crime_station, plot_crime_exploration_gender, plot_crime_exploration_age_group, plot_crime_exploration_distances, plot_crime_exploration_day_parts
from querys import query_top_stations_affluence_trends, query_top_stations_crime_trends, query_top_crimes_historical, query_crimes_exploration_gender, query_crimes_exploration_age_group, query_crimes_exploration_distances, query_crimes_part_of_day

# Extras
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# GENERAL SETTINGS OF DASHBOARD

# Page config.
st.set_page_config(page_title="Metro y Metrobús Seguro",
    initial_sidebar_state="expanded",
    layout="wide",
    page_icon="🚈")

# Hide the legend of "Made with streamlit" and hamburger menu.
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#Successful trial to remove top blankspace at dashboard
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=0, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )


# AUXILIAR FUNCTIONS

# Get the week of month given a datetime object
def week_of_month(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))

# Get the monday for a certain week of the year (auxiliary to show the range of a certain week)
def get_monday_week_year(week, year):
    first_day_year = datetime(year, 1, 1)
    monday_first_week = first_day_year - timedelta(days=first_day_year.weekday())
    return monday_first_week + timedelta(weeks=week - 1)
def get_week_date_range(week_number, year):
    start_date = datetime(year, 1, 1)
    days_offset = (7 - start_date.weekday()) % 7
    start_date += timedelta(days=days_offset)
    start_week_date = start_date + timedelta(weeks=week_number - 1)
    end_week_date = start_week_date + timedelta(days=6)
    start_date_str = f"{start_week_date.day} de {month_names[start_week_date.month]} de {start_week_date.year}"
    end_date_str = f"{end_week_date.day} de {month_names[end_week_date.month]} de {end_week_date.year}"
    
    return f"{start_date_str} al {end_date_str}"

# Get columns from cve_est.
def get_station(df, cve_est, column=None):
    if column is None:
        return df[df['cve_est'] == cve_est]
    filter = df[df['cve_est'] == cve_est]
    print(filter)
    return filter[column].to_list()[0]

# List of default regions and lines
zones_ls = ["Centro", "Norte", "Sur", "Oriente", "Poniente"]
munics_ls = {
    'STC Metro': ['Álvaro Obregón', 'Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
       'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
       'Tláhuac', 'Venustiano Carranza'],
    'Metrobús': ['Álvaro Obregón', 'Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
       'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
       'Tlalpan', 'Venustiano Carranza', 'Xochimilco'],
}
lines_ls = {
    'STC Metro': ['Línea 1', 'Línea 2', 'Línea 3', 'Línea 4', 'Línea 5', 'Línea 6', 'Línea 7', 'Línea 8',
                  'Línea 9', 'Línea 12', 'Línea A', 'Línea B',],
    'Metrobús': ['Línea 1', 'Línea 2', 'Línea 3', 'Línea 4', 'Línea 5', 'Línea 6', 'Línea 7',],
}

years_queries_ls = [2019, 2020, 2021, 2022, 2023]
months_queries_ls = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
weekdays_queries_ls = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
crime_classes_queries_ls = [
    'Robo a transeúnte en vía pública',
    'Robo a transeúnte en espacio abierto al público',
    'Robo en transporte público individual',
    'Robo en transporte público colectivo',
    'Robo en transporte individual',
    'Robo a persona en un lugar privado',
    'Robo simple',
    'Robo de vehículo',
    'Robo de autopartes',
    'Robo a institución bancaria',
    'Robo a negocio', 
    'Amenazas',
    'Fraude',
    'Extorsión',
    'Abuso sexual',
    'Acoso sexual',
    'Violación simple',
    'Violación equiparada',
    'Otro tipo de violación',
    'Estupro',
    'Otros delitos que atentan contra la libertad y la seguridad sexual',
    'Homicidio',
    'Feminicidio',
    'Lesiones',

    # 'Secuestro',
    # 'Secuestro exprés',
    # 'Posesión simple de narcóticos',
    # 'Posesión con fines de comercio o suministro de narcóticos',
    # 'Delitos en materia de armas y objetos prohibidos',
    # 'Delitos en materia de armas, explosivos y otros materiales destructivos', 
]

crime_vars_queries_ls = [
    'Robo',
    'Amenazas',
    'Fraude',
    'Extorsión',
    'Abuso sexual',
    'Acoso sexual',
    'Violación',
    'Homicidio',
    'Lesiones',
    #    'Secuestro',
    #    'Delitos contra la salud relacionados con narcóticos en su modalidad de narcomenudeo',
    #    'Delitos en materia de armas y objetos prohibidos',
    #    'Delitos de delincuencia organizada',
    #    'Delitos en materia de armas, explosivos y otros materiales destructivos',
]

# Dictionaries to fix values
dict_weekday = {
    0: 'Lunes',
    1: 'Martes',
    2: 'Miércoles',
    3: 'Jueves',
    4: 'Viernes',
    5: 'Sábado',
    6: 'Domingo',
}

dict_munics = {
    'AZCAPOTZALCO': 'Azcapotzalco',
    'COYOACAN': 'Coyoacán',
    'CUAJIMALPA DE MORELOS': 'Cuajimalpa de Morelos',
    'GUSTAVO A. MADERO': 'Gustavo A. Madero',
    'IZTACALCO': 'Iztacalco',
    'IZTAPALAPA': 'Iztapalapa',
    'MAGDALENA CONTRERAS': 'Magdalena Contreras',
    'MILPA ALTA': 'Milpa Alta',
    'ALVARO OBREGON': 'Álvaro Obregón',
    'TLAHUAC': 'Tláhuac',
    'TLALPAN': 'Tlalpan',
    'XOCHIMILCO': 'Xochimilco',
    'BENITO JUAREZ': 'Benito Juárez',
    'CUAUHTEMOC': 'Cuauhtémoc',
    'MIGUEL HIDALGO': 'Miguel Hidalgo',
    'VENUSTIANO CARRANZA': 'Venustiano Carranza',
}

month_names = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
}


# Dictionaries to load point images of stations
POINTSM = {
    'L1': './images/circulos/STCMetro_L1.png',
    'L2': './images/circulos/STCMetro_L2.png',
    'L3': './images/circulos/STCMetro_L3.png',
    'L4': './images/circulos/STCMetro_L4.png',
    'L5': './images/circulos/STCMetro_L5.png',
    'L6': './images/circulos/STCMetro_L6.png',
    'L7': './images/circulos/STCMetro_L7.png',
    'L8': './images/circulos/STCMetro_L8.png',
    'L9': './images/circulos/STCMetro_L9.png',
    'LA': './images/circulos/STCMetro_LA.png',
    'LB': './images/circulos/STCMetro_LB.png',
    'L12': './images/circulos/STCMetro_L12.png',
}

POINTSMB = {
    'L1': './images/circulos/MB_L1.png',
    'L2': './images/circulos/MB_L2.png',
    'L3': './images/circulos/MB_L3.png',
    'L4': './images/circulos/MB_L4.png',
    'L5': './images/circulos/MB_L5.png',
    'L6': './images/circulos/MB_L6.png',
    'L7': './images/circulos/MB_L7.png',
}

# Geodata
df_stations = pd.read_csv("./fact_constellation_schema/dim_estaciones_espacio_ok.csv")
df_stations['cve_mun_inegi'] = df_stations['cve_mun_inegi'].astype(str).str.zfill(3)
metro_lines = gpd.read_file('./shapefiles/metro/STC_Metro_lineas_utm14n_repr.shp', index=False)
mb_lines = gpd.read_file('./shapefiles/mb/Metrobus_lineas_utm14n_repr.shp', index=False)
zones_gdf = gpd.read_file('./shapefiles/zonas/zonas_geo.shp', index=False)
munics_gdf = gpd.read_file('./shapefiles/alcaldias/alcaldias_geo.shp', index=False)
police_sectors_gdf = gpd.read_file('./shapefiles/cuadrantes/sectores_agrupados_ssc.shp', index=False)
lineas_cdmx = gpd.read_file('./images/cdmx.json', encoding='utf-8')
lineas_cdmx_json = lineas_cdmx.to_json()

# Adjusts to geodata
mb_lines = mb_lines[mb_lines['LINEA'] != '01 y 02']
mb_lines_fix = mb_lines.copy()
mb_lines_fix['LINEA'] = mb_lines_fix['LINEA'].apply(lambda x: x.replace('0', 'L'))
df_stations_metro = df_stations[df_stations['sistema'] == 'STC Metro']
df_stations_metrobus = df_stations[df_stations['sistema'] == 'Metrobús']
munics_gdf['NOMGEO'] = munics_gdf['NOMGEO'].map(dict_munics)

# Datetime values
today_ = datetime.now()
weekday = dict_weekday[today_.weekday()]
week_year = today_.strftime("%W")
month = today_.month
year = today_.year
last_day_of_year = datetime(year, 12, 31)
last_week_of_year = int(last_day_of_year.strftime("%W"))
week_month = week_of_month(today_)

# Load of images to display at dashboard without st.image()
image_home_logo_url = "./images/MapaCDMX.png"
with open(image_home_logo_url, "rb") as file_image_home:
    contents = file_image_home.read()
    data_url_image_home = base64.b64encode(contents).decode("utf-8")
metro_logo_url = "./images/logo_metro.png"
with open(metro_logo_url, "rb") as file_metro_logo:
    contents = file_metro_logo.read()
    data_url_metro_logo = base64.b64encode(contents).decode("utf-8")
metrobus_logo_url = "./images/logo_metrobus.png"
file_metrobus_logo = open(metrobus_logo_url, "rb")
contents = file_metrobus_logo.read()
data_url_metrobus_logo = base64.b64encode(contents).decode("utf-8")
file_metrobus_logo.close()

metro_map_url = "./images/MAPA_METRO.png"
file_metro_map = open(metro_map_url, "rb")
contents = file_metro_map.read()
data_url_metro_map = base64.b64encode(contents).decode("utf-8")
file_metro_map.close()

metrobus_map_url = "./images/MAPA_METROBUS.png"
file_metrobus_map = open(metrobus_map_url, "rb")
contents = file_metrobus_map.read()
data_url_metrobus_map = base64.b64encode(contents).decode("utf-8")
file_metrobus_map.close()

# Clicks at prediction map
if 'selected_click_pred_map' not in st.session_state:
    st.session_state.selected_click_pred_map = []

# Load of crime models
@st.cache_resource
def load_crime_model(transport: str, grouped_dataset_id: int):
    if transport == 'STC Metro':
        with open('./models_trained/final/clf_crime_metro_dataset_{}_wm_2_mas_perc.pkl'.format(grouped_dataset_id), 'rb') as file:
            loaded_pipeline = pickle.load(file)
    else:
        with open('./models_trained/final/clf_crime_metrobus_dataset_{}_wm_2_mas_perc.pkl'.format(grouped_dataset_id), 'rb') as file:
            loaded_pipeline = pickle.load(file)
    
    return loaded_pipeline

# Load of affluence forecasting values
@st.cache_resource
def load_afflu_forecast_values(transport: str, grouped_dataset_id: int):
    if transport == 'STC Metro':
        if grouped_dataset_id in {3, 4}:
            df = pd.read_csv('./predictions_sarima/predicciones_afluencia_alcaldia_semana_metro.csv')
        elif grouped_dataset_id in {6, 7}:
            df = pd.read_csv('./predictions_sarima/predicciones_afluencia_sector_policial_semana_metro.csv')
    elif grouped_dataset_id in {3, 4}:
        df = pd.read_csv('./predictions_sarima/predicciones_afluencia_alcaldia_semana_metrobus_final.csv')
    elif grouped_dataset_id in {6, 7}:
        df = pd.read_csv('./predictions_sarima/predicciones_afluencia_sector_policial_semana_metrobus.csv')

    return df

# Load of weekly crime counts values
@st.cache_resource
def load_weekly_crime_counts(transport: str, grouped_dataset_id: int):
    if transport == 'STC Metro':
        df = pd.read_csv(
            f'./datasets_aux/test/carpetas_afluencia_metro_grupo_{grouped_dataset_id}_wm_final_red.csv'
        )
    else:
        df = pd.read_csv(
            f'./datasets_aux/test/carpetas_afluencia_metrobus_grupo_{grouped_dataset_id}_wm_final_red.csv'
        )

    df['semana_anio_completa'] = df['anio'].astype(str) + ' - s' + df['semana_anio'].astype(str)

    return df

# Load criteria thresholds of predictions
@st.cache_resource
def load_thresholds_crime_model(transport: str, grouped_dataset_id: int):
    if transport == 'STC Metro':
        df = pd.read_csv('./datasets_aux/test/rangos_dataset_grupo_{}_2_mas_perc.csv'.format(grouped_dataset_id))
    else:
        df = pd.read_csv('./datasets_aux/test/rangos_dataset_grupo_{}_2_mas_perc_mb.csv'.format(grouped_dataset_id))
    
    return df


# PLOT FUNCTIONS

# EXPLORATORY SECTIONS

# TRENDS SECTION

# Normalize crime counts to 4-8 scatter point size
def normalize_size(value, min_value, max_value, min_size=4, max_size=8):
    return ((value - min_value) / (max_value - min_value)) * (max_size - min_size) + min_size

def generate_geom_grouped(geom_df, level):
    # Group regions by a certain level

    groups = geom_df.groupby(level)
    grouped_geoms = []
    sectors = []

    for sector, group in groups:
        grouped_geom = unary_union(group['geometry'])
        grouped_geoms.append(grouped_geom)
        sectors.append(sector)

    data_dict_grouped_geoms = {
        level: sectors,
        'geometry': grouped_geoms
    }

    #print(data_dict_grouped_geoms)

    return gpd.GeoDataFrame(data_dict_grouped_geoms, geometry='geometry')

def label_percentiles_5_parts(row, percentiles):
    if row < percentiles.iloc[0]:
        return 1
    elif row < percentiles.iloc[1]:
        return 2
    elif row < percentiles.iloc[2]:
        return 3
    elif row < percentiles.iloc[3]:
        return 4
    else:
        return 5
    
def label_percentiles_3_parts(row, percentiles):
    if row <= percentiles.iloc[0]:
        return 1
    elif row <= percentiles.iloc[1]:
        return 2
    else:
        return 3

def generate_ranges_labels_percentiles_3_parts(percentiles, min_value, max_value):
    sorted_values = sorted(percentiles)
    ranges = []
    ranges.append(f'{min_value} - {round(sorted_values[0], 2)}')
    
    for i in range(1, len(sorted_values)):
        ranges.append(f'{round(round(sorted_values[i-1], 2) + 0.01, 2)} - {round(sorted_values[i], 2)}')

    ranges.append(f'{round(round(sorted_values[-1], 2) + 0.1, 2)} - {max_value}')
    
    return ranges

# Plot the stations showing criminal trends
def plot_crime_trend_stations(datageom, df: pd.DataFrame, transport: str, level_div: str, filter_div: list):
    region_gdf_cp = datageom.copy(deep=True)
    center_geom = region_gdf_cp['geometry'].to_list()[0].centroid
    df['promedio_delitos'] = df['promedio_delitos'].astype(float)
    # if filter_div == []:
    #     filter_div = list(region_gdf_cp[level_div].unique())
    
    # dict_colors_percentile = {
    #     1: '#5bc7da',
    #     2: '#4fc261',
    #     3: '#d3a900',
    #     4: '#d35600',
    #     5: 'red',
    # }
    
    dict_colors_percentile = {
        1: '#5bc7da',
        2: '#ffa200',
        3: 'red',
    }
    
    fig = go.Figure()
    
    fig.update_layout(
        mapbox=dict(center=dict(lat=center_geom.y, lon=center_geom.x),zoom=9.8),
    )

    # Show stations and lines
    if transport == 'STC Metro':
        df_stations_metro_complete = df_stations_metro.merge(df[['linea', 'nombre', 'promedio_delitos']], on=['linea', 'nombre'])
        min_value = df_stations_metro_complete['promedio_delitos'].min()
        max_value = df_stations_metro_complete['promedio_delitos'].max()
        # percentile_values = df_stations_metro_complete['promedio_delitos'].quantile([0.15, 0.30, 0.50, 0.70])
        percentile_values = df_stations_metro_complete['promedio_delitos'].quantile([0.33, 0.66])
        df_stations_metro_complete['clase'] = df_stations_metro_complete['promedio_delitos'].apply(label_percentiles_3_parts, percentiles=percentile_values)
        colors_ls = [dict_colors_percentile[class_] for class_ in df_stations_metro_complete['clase'].to_list()]
        
        lines_unique = df_stations_metro_complete['linea'].unique()
        colors_classes_unique = list(dict_colors_percentile.keys())
        ranges_classes_unique = generate_ranges_labels_percentiles_3_parts(percentile_values, min_value, max_value)
        
        # Map boundaries of regions selected
        if level_div != 'linea':
            # Filter geometry of choroplet
            if filter_div != []:
                region_gdf_cp_aux = region_gdf_cp[region_gdf_cp[level_div].isin(filter_div)]
            else:
                region_gdf_cp_aux = region_gdf_cp
            
            for ind, row in region_gdf_cp_aux.iterrows():
                geometry_ = row['geometry']
                lon_geom, lat_geom = geometry_.exterior.xy
                lon_geom = np.array(lon_geom).tolist()
                lat_geom = np.array(lat_geom).tolist()
                trace_boundary = go.Scattermapbox(
                    lon=lon_geom,
                    lat=lat_geom,
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(196, 223, 225, 0.15)',
                    line=dict(color='#65999d', width=2),
                    hoverinfo='none',
                    showlegend=False,
                )
                fig.add_trace(trace_boundary)
                
            region_gdf_cp_aux[level_div] = 'union'
            region_gdf_grouped = generate_geom_grouped(region_gdf_cp_aux, level_div)
            polygon_ = region_gdf_grouped['geometry'].to_list()[0]
            
            # Filter just the fragment of line which intersects the area of the choroplet
            for line in lines_unique:
                metro_lines_aux = metro_lines[metro_lines['LINEA'] == line[1:]]
                metro_lines_aux['geometry_yx'] = metro_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
                lines_ = metro_lines_aux['geometry_yx'].iloc[0]
                
                intersection = lines_.intersection(polygon_)
                if intersection.type == 'MultiLineString':
                    # Oro molido
                    # https://gis.stackexchange.com/questions/456266/error-of-multilinestring-object-is-not-iterable
                    for ind_line in list(intersection.geoms):
                        if ind_line.is_empty:
                            #print("No hay intersección")
                            pass
                        else:
                            #print('Intersecta')
                            coords_pts = [[coord[0], coord[1]] for coord in ind_line.coords]
                            line_trace = go.Scattermapbox(
                                mode='lines',
                                lon = [coord[0] for coord in coords_pts],
                                lat = [coord[1] for coord in coords_pts],
                                line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                                hoverinfo='none',
                                showlegend=False,
                            )
                            fig.add_trace(line_trace)
                else:
                    if intersection.is_empty:
                        #print("No hay intersección")
                        pass
                    else:
                        #print('Intersecta')
                        coords_pts = [[coord[0], coord[1]] for coord in intersection.coords]
                        line_trace = go.Scattermapbox(
                            mode='lines',
                            lon = [coord[0] for coord in coords_pts],
                            lat = [coord[1] for coord in coords_pts],
                            line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                            hoverinfo='none',
                            showlegend=False,
                        )
                        fig.add_trace(line_trace)
        else:
            geometry_ = lineas_cdmx['geometry'].to_list()[0]
            lon_geom, lat_geom = geometry_.exterior.xy
            lon_geom = np.array(lon_geom).tolist()
            lat_geom = np.array(lat_geom).tolist()
            trace_boundary = go.Scattermapbox(
                lon=lon_geom,
                lat=lat_geom,
                mode='lines',
                fill='toself',
                fillcolor='rgba(196, 223, 225, 0.15)',
                line=dict(color='#65999d', width=2),
                hoverinfo='none',
                showlegend=False,
            )
            fig.add_trace(trace_boundary)
            
            # Map all the lines
            for line in lines_unique:
                metro_lines_aux = metro_lines[metro_lines['LINEA'] == line[1:]]
                metro_lines_aux['geometry_yx'] = metro_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
                lines_ = metro_lines_aux['geometry_yx'].iloc[0]
                coords_pts = [[coord[0], coord[1]] for coord in lines_.coords]
                line_trace = go.Scattermapbox(
                    mode='lines',
                    lon = [coord[0] for coord in coords_pts],
                    lat = [coord[1] for coord in coords_pts],
                    line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                    hoverinfo='none',
                    showlegend=False,
                )
                fig.add_trace(line_trace)
        
        # Plot stations with different sizes and colors
        for color_cl, range_cl in zip(colors_classes_unique, ranges_classes_unique):
            df_stations_metro_aux = df_stations_metro_complete[(df_stations_metro_complete['clase'] == color_cl)]
            lats = df_stations_metro_aux['latitud']
            lons = df_stations_metro_aux['longitud']
            ids = df_stations_metro_aux['cve_est']
            lines = df_stations_metro_aux['linea']
            names = df_stations_metro_aux['nombre']
            values_ = df_stations_metro_aux['promedio_delitos']
            marker_sizes_colors = [normalize_size(value, min_value, max_value, min_size=10, max_size=16) for value in values_]
            marker_sizes_white = [normalize_size(value, min_value, max_value, min_size=4, max_size=9) for value in values_]
            
            scatter_trace = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=marker_sizes_colors,
                        color=dict_colors_percentile[color_cl],
                        opacity = 1.0,
                    ),
                    hoverinfo='none',
                    name=range_cl,
                    showlegend=True,
            )
            scatter_trace_2 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=marker_sizes_white,
                        color='white',
                        opacity = 1.0,
                    ),
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
                    showlegend=False,
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)

    else:
        df_stations_metrobus_complete = df_stations_metrobus.merge(df[['linea', 'nombre', 'promedio_delitos']], on=['linea', 'nombre'])
        min_value = df_stations_metrobus_complete['promedio_delitos'].min()
        max_value = df_stations_metrobus_complete['promedio_delitos'].max()
        # percentile_values = df_stations_metrobus_complete['promedio_delitos'].quantile([0.15, 0.30, 0.50, 0.70])
        percentile_values = df_stations_metrobus_complete['promedio_delitos'].quantile([0.33, 0.66])
        df_stations_metrobus_complete['clase'] = df_stations_metrobus_complete['promedio_delitos'].apply(label_percentiles_3_parts, percentiles=percentile_values)
        colors_ls = [dict_colors_percentile[class_] for class_ in df_stations_metrobus_complete['clase'].to_list()]
        
        lines_unique = df_stations_metrobus_complete['linea'].unique()
        colors_classes_unique = list(dict_colors_percentile.keys())
        ranges_classes_unique = generate_ranges_labels_percentiles_3_parts(percentile_values, min_value, max_value)
        
        # Map boundaries of regions selected
        if level_div != 'linea':
            # Filter geometry of choroplet
            if filter_div != []:
                region_gdf_cp_aux = region_gdf_cp[region_gdf_cp[level_div].isin(filter_div)]
            else:
                region_gdf_cp_aux = region_gdf_cp
            
            for ind, row in region_gdf_cp_aux.iterrows():
                geometry_ = row['geometry']
                lon_geom, lat_geom = geometry_.exterior.xy
                lon_geom = np.array(lon_geom).tolist()
                lat_geom = np.array(lat_geom).tolist()
                trace_boundary = go.Scattermapbox(
                    lon=lon_geom,
                    lat=lat_geom,
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(196, 223, 225, 0.15)',
                    line=dict(color='#65999d', width=2),
                    hoverinfo='none',
                    showlegend=False,
                )
                fig.add_trace(trace_boundary)
                
            region_gdf_cp_aux[level_div] = 'union'
            region_gdf_grouped = generate_geom_grouped(region_gdf_cp_aux, level_div)
            polygon_ = region_gdf_grouped['geometry'].to_list()[0]
            
            # Filter just the fragment of line which intersects the area of the choroplet
            for line in lines_unique:
                metrobus_lines_aux = mb_lines[mb_lines['LINEA'].str[-1] == line[1:]]
                metrobus_lines_aux['geometry_yx'] = metrobus_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
                lines_ = metrobus_lines_aux['geometry_yx']
                
                for i in range(len(lines_)):
                    line_ = lines_.iloc[i]
                    intersection = line_.intersection(polygon_)
                    if intersection.type == 'MultiLineString':
                        # Oro molido
                        # https://gis.stackexchange.com/questions/456266/error-of-multilinestring-object-is-not-iterable
                        for ind_line in list(intersection.geoms):
                            if ind_line.is_empty:
                                #print("No hay intersección")
                                pass
                            else:
                                #print('Intersecta')
                                coords_pts = [[coord[0], coord[1]] for coord in ind_line.coords]
                                line_trace = go.Scattermapbox(
                                    mode='lines',
                                    lon = [coord[0] for coord in coords_pts],
                                    lat = [coord[1] for coord in coords_pts],
                                    line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                                    hoverinfo='none',
                                    showlegend=False,
                                )
                                fig.add_trace(line_trace)
                    else:
                        if intersection.is_empty:
                            #print("No hay intersección")
                            pass
                        else:
                            #print('Intersecta')
                            coords_pts = [[coord[0], coord[1]] for coord in intersection.coords]
                            line_trace = go.Scattermapbox(
                                mode='lines',
                                lon = [coord[0] for coord in coords_pts],
                                lat = [coord[1] for coord in coords_pts],
                                line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                                hoverinfo='none',
                                showlegend=False,
                            )
                            fig.add_trace(line_trace) 
        else:
            geometry_ = lineas_cdmx['geometry'].to_list()[0]
            lon_geom, lat_geom = geometry_.exterior.xy
            lon_geom = np.array(lon_geom).tolist()
            lat_geom = np.array(lat_geom).tolist()
            trace_boundary = go.Scattermapbox(
                lon=lon_geom,
                lat=lat_geom,
                mode='lines',
                fill='toself',
                fillcolor='rgba(196, 223, 225, 0.15)',
                line=dict(color='#65999d', width=2),
                hoverinfo='none',
                showlegend=False,
            )
            fig.add_trace(trace_boundary)
            
            # Map all the lines
            for line in lines_unique:
                metrobus_lines_aux = mb_lines[mb_lines['LINEA'].str[-1] == line[1:]]
                metrobus_lines_aux['geometry_yx'] = metrobus_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
                lines_ = metrobus_lines_aux['geometry_yx']
                
                for i in range(len(lines_)):
                    line_ = lines_.iloc[i]
                    coords_pts = [[coord[0], coord[1]] for coord in line_.coords]
                    line_trace = go.Scattermapbox(
                        mode='lines',
                        lon = [coord[0] for coord in coords_pts],
                        lat = [coord[1] for coord in coords_pts],
                        line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                        hoverinfo='none',
                        showlegend=False,
                    )
                    fig.add_trace(line_trace)

        # Plot stations with different sizes and colors
        for color_cl, range_cl in zip(colors_classes_unique, ranges_classes_unique):
            df_stations_metrobus_aux = df_stations_metrobus_complete[(df_stations_metrobus_complete['clase'] == color_cl)]
            lats = df_stations_metrobus_aux['latitud']
            lons = df_stations_metrobus_aux['longitud']
            ids = df_stations_metrobus_aux['cve_est']
            lines = df_stations_metrobus_aux['linea']
            names = df_stations_metrobus_aux['nombre']
            values_ = df_stations_metrobus_aux['promedio_delitos']
            marker_sizes_colors = [normalize_size(value, min_value, max_value, min_size=8, max_size=16) for value in values_]
            marker_sizes_white = [normalize_size(value, min_value, max_value, min_size=4, max_size=7) for value in values_]
            
            scatter_trace = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=marker_sizes_colors,
                        color=dict_colors_percentile[color_cl],
                        opacity = 1.0,
                    ),
                    hoverinfo='none',
                    name=range_cl,
                    showlegend=True,
            )
            scatter_trace_2 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=marker_sizes_white,
                        color='white',
                        opacity = 1.0,
                    ),
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
                    showlegend=False,
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)

    # colors_ls_aux_final = list(dict_colors_percentile.values())
    # labels_colors_ls_aux_final = list(dict_colors_percentile.keys())
    # for label, color in zip(labels_colors_ls_aux_final, colors_ls_aux_final):
    #     fig.add_trace(go.Scattermapbox(
    #         lat=[None],
    #         lon=[None],
    #         mode='markers',
    #         marker=dict(
    #             size=10,
    #             color=color,
    #             opacity=1.0,
    #         ),
    #         name=label,
    #         showlegend=True,
    #     ))
    
    fig.update_layout(
        title_text='',
        margin=dict(t=0, l=0, r=21, b=0),
        title=dict(
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
        ),
        legend=dict(
            title=dict(
                text='Puntos calientes',
                font=dict(
                    size=14,
                    color='black',
                )
            ),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        legend_title=dict(side='top center'),
        showlegend=True,

        mapbox_style="carto-positron",
        mapbox=dict(
            pitch=0,
        ),
        height = 480,
        autosize=True,
        dragmode=False,
    )
    
    return fig

# EXPLORATION SECTION

# Plot the stations showing criminal trends
def plot_transport_stations(datageom, transport: str):
    region_gdf_cp = datageom.copy(deep=True)
    center_geom = region_gdf_cp['geometry'].to_list()[0].centroid
    
    fig = go.Figure()
    
    fig.update_layout(
        mapbox=dict(center=dict(lat=center_geom.y, lon=center_geom.x),zoom=10.2),
    )

    # Show stations and lines
    if transport == 'STC Metro':
        df_stations_metro_complete = df_stations_metro
        lines_unique = df_stations_metro_complete['linea'].unique()
        
        # Map boundaries of regions selected
        geometry_ = lineas_cdmx['geometry'].to_list()[0]
        lon_geom, lat_geom = geometry_.exterior.xy
        lon_geom = np.array(lon_geom).tolist()
        lat_geom = np.array(lat_geom).tolist()
        trace_boundary = go.Scattermapbox(
            lon=lon_geom,
            lat=lat_geom,
            mode='lines',
            fill='toself',
            fillcolor='rgba(196, 223, 225, 0.15)',
            line=dict(color='#65999d', width=2),
            hoverinfo='none',
        )
        fig.add_trace(trace_boundary)
        
        # Map all the lines
        for line in lines_unique:
            metro_lines_aux = metro_lines[metro_lines['LINEA'] == line[1:]]
            metro_lines_aux['geometry_yx'] = metro_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
            lines_ = metro_lines_aux['geometry_yx'].iloc[0]
            coords_pts = [[coord[0], coord[1]] for coord in lines_.coords]
            line_trace = go.Scattermapbox(
                mode='lines',
                lon = [coord[0] for coord in coords_pts],
                lat = [coord[1] for coord in coords_pts],
                line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                hoverinfo='none',
            )
            fig.add_trace(line_trace)
        
        # Map all stations
        lats = df_stations_metro_complete['latitud']
        lons = df_stations_metro_complete['longitud']
        ids = df_stations_metro_complete['cve_est']
        lines = df_stations_metro_complete['linea']
        names = df_stations_metro_complete['nombre']
        scatter_trace = go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                textposition='top center',
                marker=dict(
                    size=8,
                    color='black',
                    opacity = 1.0,
                ),
                hoverinfo='none',
        )
        scatter_trace_2 = go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                textposition='top center',
                marker=dict(
                    size=4,
                    color='white',
                    opacity = 1.0,
                ),
                hovertext=ids,
                hoverlabel=dict(namelength=0),
                hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
        )
        fig.add_trace(scatter_trace)
        fig.add_trace(scatter_trace_2)

    else:
        df_stations_metrobus_complete = df_stations_metrobus
        lines_unique = df_stations_metrobus_complete['linea'].unique()
        
        # Map boundaries of regions selected
        geometry_ = lineas_cdmx['geometry'].to_list()[0]
        lon_geom, lat_geom = geometry_.exterior.xy
        lon_geom = np.array(lon_geom).tolist()
        lat_geom = np.array(lat_geom).tolist()
        trace_boundary = go.Scattermapbox(
            lon=lon_geom,
            lat=lat_geom,
            mode='lines',
            fill='toself',
            fillcolor='rgba(196, 223, 225, 0.15)',
            line=dict(color='#65999d', width=2),
            hoverinfo='none',
        )
        fig.add_trace(trace_boundary)
        
        # Map all lines
        for line in lines_unique:
            metrobus_lines_aux = mb_lines[mb_lines['LINEA'].str[-1] == line[1:]]
            metrobus_lines_aux['geometry_yx'] = metrobus_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
            lines_ = metrobus_lines_aux['geometry_yx']
            
            for i in range(len(lines_)):
                line_ = lines_.iloc[i]
                coords_pts = [[coord[0], coord[1]] for coord in line_.coords]
                line_trace = go.Scattermapbox(
                    mode='lines',
                    lon = [coord[0] for coord in coords_pts],
                    lat = [coord[1] for coord in coords_pts],
                    line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                    hoverinfo='none',
                )
                fig.add_trace(line_trace)

        # Map all the stations
        lats = df_stations_metrobus_complete['latitud']
        lons = df_stations_metrobus_complete['longitud']
        ids = df_stations_metrobus_complete['cve_est']
        lines = df_stations_metrobus_complete['linea']
        names = df_stations_metrobus_complete['nombre']
        
        scatter_trace = go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
            textposition='top center',
            marker=dict(
                size=8,
                color='black',
                opacity = 1.0,
            ),
            hoverinfo='none',
        )
        scatter_trace_2 = go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
            textposition='top center',
            marker=dict(
                size=4,
                color='white',
                opacity = 1.0,
            ),
            hovertext=ids,
            hoverlabel=dict(namelength=0),
            hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
        )
        fig.add_trace(scatter_trace)
        fig.add_trace(scatter_trace_2)
    
    fig.update_layout(
        title_text='',
        margin=dict(t=0, l=0, r=0, b=0),
        legend=dict(
            title='',
            traceorder='normal',
            orientation='h',
            y=0,
            x=0,
            xanchor='center',
            yanchor='bottom',
            itemsizing='constant',
            itemwidth=30,
            bgcolor='rgba(255, 255, 255, 0)'
        ),
        legend_title=dict(side='top right'),
        showlegend=False,

        mapbox_style="carto-positron",
        mapbox=dict(
            pitch=0,
        ),
        height = 444,
        autosize=True,
        dragmode=False,
    )
    
    return fig

# PREDICTION SECTION

# Plot crime level risk map
def plot_predictive_map(datageom, transport: str, unique_values_metric):
    metric = 'valor'
    
    centroids = []
    for index, row in datageom.iterrows():
        multi_polygon = row['geometry']
        if isinstance(multi_polygon, MultiPolygon):
            for polygon in multi_polygon.geoms:
                if isinstance(polygon, Polygon):
                    centroids.append(polygon.centroid)
        elif isinstance(multi_polygon, Polygon):
            centroids.append(multi_polygon.centroid)

    center_geom = MultiPoint(centroids).centroid
    
    fig = go.Figure()

    # Useful links to make the figures
    # https://community.plotly.com/t/annotations-on-plotly-choropleth-choropleth-mapbox-scattermapbox/74556/6
    # https://stackoverflow.com/questions/68709936/how-to-plot-a-shp-file-with-plotly-go-choroplethmapbox
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Choroplethmapbox.html
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.choroplethmapbox.html#plotly.graph_objects.choroplethmapbox.ColorBar
    
    colorscales = [
        ((0.0, '#ff5454'), (1.0, '#ff5454')),
        ((0.0, '#b0f2f4'), (1.0, '#b0f2f4')),
    ]
    colorborders = [
        '#b21800', '#5bc7da',
    ]
    markerlinewidths = [
        3, 1
    ]
    
    # Hide colorbar
    # https://community.plotly.com/t/hide-colorbar-from-px-choropleth/34970/13
    # Discrete map
    # https://community.plotly.com/t/discrete-colors-for-choroplethmapbox-with-plotly-graph-objects/37989/2
    
    # CHECK TO HIGHLIGHT CHOROPLET
    # https://towardsdatascience.com/highlighting-click-data-on-plotly-choropleth-map-377e721c5893
    
    # If the entire map is going to be red or blue (high and low risk), we should make a special plot to avoid a bug
    if datageom[metric].value_counts().iloc[0] == len(datageom):
        dfp = datageom
        label_ = datageom[metric].unique()[0]
        if label_ == 'Riesgo moderado':
            varaux = 1  
        else:
            varaux = 0
        fig.add_trace(
            go.Choroplethmapbox(
                    geojson=json.loads(dfp.to_json()), 
                    locations=dfp.index,
                    z=[varaux,] * len(dfp[metric]),
                    customdata=dfp['NOMGEO'],
                    colorscale=colorscales[varaux],
                    marker_opacity=0.8,
                    marker_line_width=markerlinewidths[varaux],
                    hoverlabel_bgcolor='white',
                    marker_line_color=colorborders[varaux],
                    hovertext=dfp[metric],
                    hoverlabel=dict(namelength=0),
                    hovertemplate = '<b>%{customdata}</b>: %{hovertext}',
                    showlegend=False,
                    showscale=False,
            )
        )
    # If the map has choroplets from both classes
    else:
        for i, label_ in enumerate(unique_values_metric):
            dfp = datageom[datageom[metric] == label_]
            fig.add_trace(
                go.Choroplethmapbox(
                        geojson=json.loads(dfp.to_json()), 
                        locations=dfp.index,
                        z=[i,] * len(dfp[metric]),
                        customdata=dfp['NOMGEO'],
                        colorscale=colorscales[i],
                        marker_opacity=0.8,
                        marker_line_width=markerlinewidths[i],
                        hoverlabel_bgcolor='white',
                        marker_line_color=colorborders[i],
                        hovertext=dfp[metric],
                        hoverlabel=dict(namelength=0),
                        hovertemplate = '<b>%{customdata}</b>: %{hovertext}',
                        showlegend=False,
                        showscale=False,
                )
            )
        
    fig.update_layout(
        title_text='',
        margin=dict(t=0, l=0, r=0, b=0),
        title=dict(
            y=0.5,
            x=0.5,
            xanchor='center',
            yanchor='top',
        ),
        legend=dict(
            title='',
            traceorder='normal',
            orientation='h',
            y=0.5,
            x=0.5,
            xanchor='center',
            yanchor='top',
            itemsizing='constant',
            itemwidth=30,
            bgcolor='rgba(255, 255, 255, 0)'
        ),
        mapbox_style="carto-positron",
        height = 450,
        autosize=True,
        mapbox=dict(center=dict(lat=center_geom.y, lon=center_geom.x),zoom=9.0),
        dragmode=False
    )

    return fig

# Plot the view of the region selected with its stations inside
def plot_complementary_predictive_map(datageom, transport: str, region_gdf_aux: pd.DataFrame, region_column: str):
    region_gdf_cp = datageom.copy(deep=True)
    metric = 'valor_'
    cve_region_selected = region_gdf_aux[region_column].to_list()[0]
    region_gdf_cp.loc[region_gdf_cp['CVE_MUN'] == cve_region_selected, metric] = 1
    region_gdf_cp = region_gdf_cp.dropna()
    
    center_geom = region_gdf_cp['geometry'].to_list()[0].centroid
    
    fig = go.Figure()
    
    fig.update_layout(
        mapbox=dict(center=dict(lat=center_geom.y, lon=center_geom.x),zoom=9.8),dragmode=False
    )

    # The figures were made thanks to this
    # https://community.plotly.com/t/annotations-on-plotly-choropleth-choropleth-mapbox-scattermapbox/74556/6
    # https://stackoverflow.com/questions/68709936/how-to-plot-a-shp-file-with-plotly-go-choroplethmapbox
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Choroplethmapbox.html
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.choroplethmapbox.html#plotly.graph_objects.choroplethmapbox.ColorBar
    
    # Show stations and lines
    if transport == 'STC Metro':
        lines_unique = df_stations_metro['linea'].unique()
        
        # Mapping all the lines
        for line in lines_unique:
            metro_lines_aux = metro_lines[metro_lines['LINEA'] == line[1:]]
            metro_lines_aux['geometry_yx'] = metro_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
            lines_ = metro_lines_aux['geometry_yx'].iloc[0]
            coords_pts = [[coord[0], coord[1]] for coord in lines_.coords]
            line_trace = go.Scattermapbox(
                mode='lines',
                lon = [coord[0] for coord in coords_pts],
                lat = [coord[1] for coord in coords_pts],
                line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                hoverinfo='none',
            )
            
            fig.add_trace(line_trace)
                    
        
        # Filter just the stations within the selected region to mark it differently
        for line in lines_unique:
            if region_column == 'CVE_MUN':
                region_column_aux = 'cve_mun_inegi'
            else:
                region_column_aux = 'sector'
            
            # Stations within the region
            df_stations_metro_aux = df_stations_metro[(df_stations_metro['linea'] == line) & (df_stations_metro[region_column_aux] == cve_region_selected)]
            lats = df_stations_metro_aux['latitud']
            lons = df_stations_metro_aux['longitud']
            ids = df_stations_metro_aux['cve_est']
            lines = df_stations_metro_aux['linea']
            names = df_stations_metro_aux['nombre']
            scatter_trace = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker_size=10,
                    marker_color='black',
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_2 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker_size=6,
                    marker_color='white',
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>'
            )
            scatter_trace_3 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker_size=3,
                    marker_color=LINESM_aux[line],
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)
            #fig.add_trace(scatter_trace_3)
            
            # Stations outside the region
            df_stations_metro_aux_out = df_stations_metro[(df_stations_metro['linea'] == line) & (df_stations_metro[region_column_aux] != cve_region_selected)]
            lats_out = df_stations_metro_aux_out['latitud']
            lons_out = df_stations_metro_aux_out['longitud']
            ids_out = df_stations_metro_aux_out['cve_est']
            lines_out = df_stations_metro_aux_out['linea']
            names_out = df_stations_metro_aux_out['nombre']
            scatter_trace_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker_size=6,
                    marker_color='gray',
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_2_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker_size=4,
                    marker_color='white',
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
            )
            scatter_trace_3_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker_size=3,
                    marker_color=LINESM_aux[line],
                    hovertemplate=''
            )
            fig.add_trace(scatter_trace_out)
            fig.add_trace(scatter_trace_2_out)
            #fig.add_trace(scatter_trace_3_out)

    else:
        lines_unique = df_stations_metrobus['linea'].unique()
        
        # Mapping all the lines
        for line in lines_unique:
            metrobus_lines_aux = mb_lines[mb_lines['LINEA'].str[-1] == line[1:]]
            metrobus_lines_aux['geometry_yx'] = metrobus_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
            lines_ = metrobus_lines_aux['geometry_yx']
            
            for i in range(len(lines_)):
                line_ = lines_.iloc[i]
                coords_pts = [[coord[0], coord[1]] for coord in line_.coords]
                line_trace = go.Scattermapbox(
                    mode='lines',
                    lon = [coord[0] for coord in coords_pts],
                    lat = [coord[1] for coord in coords_pts],
                    line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                    hoverinfo='none',
                )
                
                fig.add_trace(line_trace)

        # Filter just the stations within the selected region to mark it differently
        for line in lines_unique:
            if region_column == 'CVE_MUN':
                region_column_aux = 'cve_mun_inegi'
            else:
                region_column_aux = 'sector'
            
            # Stations within the region
            df_stations_metrobus_aux = df_stations_metrobus[(df_stations_metrobus['linea'] == line) & (df_stations_metrobus[region_column_aux] == cve_region_selected)]
            lats = df_stations_metrobus_aux['latitud']
            lons = df_stations_metrobus_aux['longitud']
            ids = df_stations_metrobus_aux['cve_est']
            lines = df_stations_metrobus_aux['linea']
            names = df_stations_metrobus_aux['nombre']
            scatter_trace = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker_size=10,
                    marker_color='black',
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_2 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker_size=6,
                    marker_color='white',
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
            )
            scatter_trace_3 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker_size=3,
                    marker_color=LINESM_aux[line],
                    hovertemplate=''
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)
            #fig.add_trace(scatter_trace_3)
            
            
            # Stations outside the region
            df_stations_metrobus_aux_out = df_stations_metrobus[(df_stations_metrobus['linea'] == line) & (df_stations_metrobus[region_column_aux] != cve_region_selected)]
            lats_out = df_stations_metrobus_aux_out['latitud']
            lons_out = df_stations_metrobus_aux_out['longitud']
            ids_out = df_stations_metrobus_aux_out['cve_est']
            lines_out = df_stations_metrobus_aux_out['linea']
            names_out = df_stations_metrobus_aux_out['nombre']
            scatter_trace_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker_size=6,
                    marker_color='gray',
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_2_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker_size=4,
                    marker_color='white',
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
            )
            scatter_trace_3_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker_size=3,
                    marker_color=LINESM_aux[line],
                    hovertemplate=''
            )
            fig.add_trace(scatter_trace_out)
            fig.add_trace(scatter_trace_2_out)
            #fig.add_trace(scatter_trace_3_out)
    
    fig.update_layout(
        title_text='',
        margin=dict(t=0, l=0, r=0, b=0),
        title=dict(
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
        ),
        legend=dict(
            title='',
            traceorder='normal',
            orientation='h',
            y=0.5,
            x=0.5,
            xanchor='center',
            yanchor='top',
            itemsizing='constant',
            itemwidth=30,
            bgcolor='rgba(255, 255, 255, 0)'
        ),
        legend_title=dict(side='top right'),
        showlegend=False,

        mapbox_style="carto-positron",
        mapbox=dict(
            pitch=20,
        ),
        height = 260,
        autosize=True,
        dragmode=False,
    )

    # Trace border of region selected
    geometry_ = region_gdf_cp['geometry'].iloc[0]
    lon_geom, lat_geom = geometry_.exterior.xy
    lon_geom = np.array(lon_geom).tolist()
    lat_geom = np.array(lat_geom).tolist()
    
    colorscales = [
        'rgba(255,84,84,0.3)',
        'rgba(40, 202, 207, 0.3)',
    ]
    colorborders = [
        '#b21800', '#5bc7da',
    ]
    markerlinewidths = [
        2.5, 2.5
    ]

    if region_gdf_aux['valor'].to_list()[0] == 'Riesgo elevado':
        cs = colorscales[0]
        cb = colorborders[0]
        mklw = markerlinewidths[0]
    else:
        cs = colorscales[1]
        cb = colorborders[1]
        mklw = markerlinewidths[1]

    trace_boundary = go.Scattermapbox(
        lon=lon_geom,
        lat=lat_geom,
        mode='lines',
        fill='toself',
        fillcolor=cs,
        line=dict(color=cb, width=mklw),
        hoverinfo='none',
    )
    
    fig.add_trace(trace_boundary)
    
    return fig


# DASHBOARD 

# Loose styles with CSS

#Aux center CSS for columns
#CSS style to center horizontally and vertically
center_css = """
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: row;
"""

# Home view.
def home():  # sourcery skip: extract-duplicate-method
    
    # Style config.
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #3c6f90;
        }
        [data-testid=stSidebar] h1 {
            color: white;
        }
    </style>
                
    """, unsafe_allow_html=True)
    st.title("¡Bienvenido a tu transporte seguro!\n")
    st.markdown('<br>', unsafe_allow_html=True,)
    col_1, col_mid, col_2 = st.columns([0.30, 0.05, 0.65])
    with col_1:
        st.image(r'./images/MapaCDMX.png',  use_column_width=True, output_format='PNG')
    with col_2:
        st.subheader("La delincuencia en el transporte público de la Ciudad de México")
        st.markdown(r'<div style="text-align: justify;">El transporte público es un elemento esencial en la vida cotidiana de las personas. En particular, para la Ciudad de México el STC Metro y Metrobús son los medios de transporte más utilizados, por lo que, es importante garantizar la seguridad y satisfacción de los usuarios. Sin embargo, debido al crecimiento en la red de transporte público, se ha generado una alta concentración de personas en las instalaciones de ambos medios de transporte, lo que ha propiciado un aumento en la incidencia delictiva.</div><br>', unsafe_allow_html=True,)
        st.markdown(r'<div style="text-align: justify;">Para conocer la dinámica de los delitos que ocurrieron dentro y en las cercanías de las estaciones de ambos medios de transporte, se requiere de un proceso de integración de datos públicos donde se considera:</div><br>', unsafe_allow_html=True,)
        st.markdown("""
                    - Carpetas de investigación de la Fiscalía General de Justicia de la Ciudad de México.
                    - Datos geoespaciales de las estaciones de Metro y Metrobús.
                    - Datos de afluencia de las estaciones de Metro y Metrobús.
                    """)
    st.markdown('<br>', unsafe_allow_html=True,)
    level_div = st.selectbox("Tipo de transporte", ["STC Metro", "Metrobús"])
    st.markdown('<br>', unsafe_allow_html=True,)
    if level_div == "STC Metro":
        col_1, col_mid, col_2 = st.columns([0.45, 0.05, 0.45])
        with col_1:
            st.markdown(r'<div style="text-align: justify;">El Sistema de Transporte Colectivo Metro (STC Metro) es una red de transporte público subterráneo que se encuentra en la Ciudad de México y parte de su área metropolitana. Según el INEGI, 90.2 millones de personas usaban mensualmente este transporte en 2022, por lo que lo vuelve en el transporte público más utilizado en la ciudad y su área metropolitana.</div><br>', unsafe_allow_html=True,)
            st.write(r'El metro cuenta con:')
            st.markdown("""
                        - 12 líneas.
                        - 195 estaciones.
                        - 269.52 km.""")
            st.write(r'Horarios de operación: ')
            st.markdown("""
                        - Lunes a viernes: 5:00-0:00 horas.
                        - Sábados: 6:00-0:00 horas.
                        - Domingos y días festivos: 7:00-0:00 horas.
                        """)
        with col_2:
            #st.markdown(
            #    f'<div style="{center_css}"><img src="data:image/gif;base64,{data_url_metro_map}" alt="Imagen home" width=600 ></div>',
            #    unsafe_allow_html=True,
            #)
            st.image(r'./images/MAPA_METRO.png', use_column_width=True, output_format='PNG')
    elif level_div == "Metrobús":
        col_1, col_mid, col_2 = st.columns([0.45, 0.05, 0.45])
        with col_1:
            st.write(r'<div style="text-align: justify;">El Metrobús es un sistema de autobuses con infraestructura dedicada, carriles exclusivos y sistemas de control, que se inauguró el 19 de junio de 2005. Según el INEGI, en 2022 el metrobús prestó servicio a 33.8 millones de personas de manera mensual. Se considera que es el tipo de transporte que le sigue en importancia al STC Metro.</div><br>', unsafe_allow_html=True,)
            st.write(r'El metrobús cuenta con:')
            st.markdown("""
                        - 7 líneas.
                        - 283 estaciones.
                        - 125 km.""")
            st.write(r'Horarios de operación: ')
            st.markdown("""
                        - Lunes a sábado: 4:30-0:00 horas.
                        - Domingos y días festivos: 5:00-0:00 horas.
                        """)
        with col_2:
            #st.markdown(
            #    r'<div style="{}"><img src="data:image/gif;base64,{}" alt="Imagen home" width=600 ></div>'.format(center_css, data_url_metrobus_map),
            #    unsafe_allow_html=True,
            #)
            st.image(r'./images/MAPA_METROBUS.png', use_column_width=True, output_format='PNG')
    

# Metro view.
def trends():
    st.markdown("""
        <style>
            [data-testid=stSidebar] {
                background-color: #e8540c;
            }
            [data-testid=stSidebar] h1 {
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)

    with st.container():
        st.header("🔥 Tendencias")
        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            transport = st.selectbox("Sistema", ["STC Metro", "Metrobús"])
            st.session_state.transport = transport

        with col2:
            level_div = st.selectbox("Nivel de filtrado", ["Zona", "Alcaldía", "Línea"])

        filter_div = []
        region_column = ""
        with col3:
            if level_div == 'Zona':
                filter_div = st.multiselect("Filtrado", zones_ls, placeholder='Selecciona zonas')
                region_column = 'zona'
                datageom = zones_gdf

            elif level_div == 'Alcaldía':
                filter_div = st.multiselect("Filtrado", munics_ls[st.session_state.transport], placeholder='Selecciona alcaldías')
                print(munics_gdf)
                region_column = 'NOMGEO'
                datageom = munics_gdf[munics_gdf[region_column].isin(munics_ls[transport])]

            elif level_div == 'Línea':
                filter_div = st.multiselect("Filtrado", lines_ls[st.session_state.transport], placeholder='Selecciona líneas')
                region_column = 'linea'
                datageom = munics_gdf

        st.write(
            f"🗓️ Las siguientes tendencias se toman a partir de los datos históricos en los días <b>{weekday.lower()}</b> de la <b>semana {week_year}</b> de años pasados.",
            unsafe_allow_html=True,
        )

        # Make SQL queries
        n = 10
        radio_int = 540 if transport == 'STC Metro' else 270
        df_top_stations_affluence_trends = query_top_stations_affluence_trends(transport, level_div, filter_div, weekday, week_year, n)
        df_top_stations_crime_trends = query_top_stations_crime_trends(transport, level_div, filter_div, weekday, week_year, radio_int, 1000)
        df_top_stations_crime_trends_aux = df_top_stations_crime_trends.head(n)

        col3, col4 = st.columns([2, 2])
        with col3:
            st.write("##### Mapa de puntos calientes delictivos")
            fig = plot_crime_trend_stations(datageom, df_top_stations_crime_trends, transport, region_column, filter_div)
            fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})

        with col4:
            st.write(f"##### Top {n} estaciones más peligrosas")
            fig_top_peligrosas = plot_top_stations_crime_trends(df_top_stations_crime_trends_aux, transport)
            fig_top_peligrosas.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
            st.plotly_chart(fig_top_peligrosas, use_container_width=True, config={'displayModeBar': False})

            st.write(f"##### Top {n} estaciones con mayor afluencia")
            fig_top_afluencia = plot_top_stations_affluence_trends(df_top_stations_affluence_trends, transport)
            fig_top_afluencia.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
            st.plotly_chart(fig_top_afluencia, use_container_width=True, config={'displayModeBar': False})
                    
# Metrobus view
def exploration():
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #c80f2e;
        }
        [data-testid=stSidebar] h1 {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.header("🔍 Explora las estaciones")
        
        col1, colmid, col2 = st.columns([20, 1, 20])
        # Map column.
        with col1:
            transport = st.selectbox("Sistema", ["STC Metro", "Metrobús"])
            df_stations_transport = df_stations_metro if transport == 'STC Metro' else df_stations_metrobus
            st.session_state.transport = transport
            radio_int = 540 if transport == 'STC Metro' else 270
            radio_ = '540' if transport == 'STC Metro' else '270'
            fig = plot_transport_stations(zones_gdf, transport)
            selected_click_cve_est = plotly_events(fig, click_event=True,)
            st.session_state.selected_id.append(selected_click_cve_est)
        
        with col2:
            # Second column validation.
            if 'selected_id' not in st.session_state:
                st.subheader("Seleccione una estación")
                st.write(f"Al seleccionar aparecerá la siguiente información sobre los hechos delictivos ocurridos dentro un radio de {radio_} metros alrededor de la estación seleccionada:")
                st.write(" - Top delitos más frecuentes")
                st.write(" - Comparación de géneros de víctimas")
                st.write(" - Rangos de edad más vulnerables")
                st.write(" - Comportamiento de la distancia delito-estación")
                st.write(" - Partes del día más delictivas")
            else:
                if st.session_state.selected_id is not None and st.session_state.selected_id[-1]:
                    last_selected_id = st.session_state.selected_id[-1]
                    if last_selected_id[0]['curveNumber'] >= len(lines_ls[transport]) + 1:
                        if transport == 'STC Metro':
                            df_stations_metro_filtered = df_stations_metro.iloc[last_selected_id[0]['pointIndex']]
                            cve_est = df_stations_metro_filtered['cve_est'] 
                        else:
                            df_stations_metrobus_filtered = df_stations_metrobus.iloc[last_selected_id[0]['pointIndex']]
                            cve_est = df_stations_metrobus_filtered['cve_est'] 
                        
                        name_est = get_station(df_stations_transport, cve_est, "nombre")
                        tipo_est = get_station(df_stations_transport, cve_est, "tipo")
                        linea_est = get_station(df_stations_transport, cve_est, "linea")
                        nivel_circul = get_station(df_stations_transport, cve_est, "nivel_circulacion_transporte")
                        name_est_line = name_est + f' (L{linea_est[1:]})'
                        st.subheader(f'{name_est_line.upper()}')
                        st.write(f'Tipo de estación: {tipo_est.capitalize()}')
                        st.write(f'Nivel de circulación: {nivel_circul.capitalize()}')
                        st.write("##### Top 10 delitos históricos más frecuentes (2019-2023)")
                        
                        # SQL queries
                        df_top_crimes_historical = query_top_crimes_historical(transport, cve_est, radio_int, 10)
                        fig = plot_top_crime_station(df_top_crimes_historical)
                        fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})

                    else:
                        st.subheader("Seleccione una estación")
                        st.write(f"Al seleccionar aparecerá la siguiente información sobre los hechos delictivos ocurridos dentro un radio de {radio_} metros alrededor de la estación seleccionada:")
                        st.write(" - Top delitos más frecuentes")
                        st.write(" - Comparación de géneros de víctimas")
                        st.write(" - Rangos de edad más vulnerables")
                        st.write(" - Comportamiento de la distancia delito-estación")
                        st.write(" - Partes del día más delictivas") 
                
                else:
                    radio_ = '540' if transport == 'STC Metro' else '270'
                    st.subheader("Seleccione una estación")
                    st.write(f"Al seleccionar aparecerá la siguiente información sobre los hechos delictivos ocurridos dentro un radio de {radio_} metros alrededor de la estación seleccionada:")
                    st.write(" - Top delitos más frecuentes")
                    st.write(" - Comparación de géneros de víctimas")
                    st.write(" - Rangos de edad más vulnerables")
                    st.write(" - Comportamiento de la distancia delito-estación")
                    st.write(" - Partes del día más delictivas") 
        
        if 'selected_id' in st.session_state:
            if st.session_state.selected_id is not None and st.session_state.selected_id[-1]:
                last_selected_id = st.session_state.selected_id[-1]
                if last_selected_id[0]['curveNumber'] >= len(lines_ls[transport]) + 1:
                    col3, col4 = st.columns([2, 3])
                    with col3:    
                        weekday_selected = st.selectbox('Día de la semana', weekdays_queries_ls, index=weekdays_queries_ls.index(weekday))
                    with col4:    
                        crime_var_selected = st.selectbox('Variable delito', crime_vars_queries_ls)
                    
                    df_crimes_exploration_gender = query_crimes_exploration_gender(transport, cve_est, radio_int, weekday_selected, crime_var_selected)
                    df_crimes_exploration_age_group = query_crimes_exploration_age_group(transport, cve_est, radio_int, weekday_selected, crime_var_selected)
                    df_crimes_exploration_distances = query_crimes_exploration_distances(transport, cve_est, radio_int, weekday_selected, crime_var_selected)    
                    df_crimes_exploration_part_of_day = query_crimes_part_of_day(transport, cve_est, radio_int, weekday_selected, crime_var_selected)
                    
                    total_rows_sample = df_crimes_exploration_gender['conteo_delitos'].sum()
                    
                    if total_rows_sample > 0:
                        st.markdown(f"{total_rows_sample} {' carpetas de investigación encontrada' if total_rows_sample == 1 else ' carpetas de investigación encontradas'}") 
                        col6, col7, col8, col9 = st.columns([2, 2, 3, 2])
                        with col6:
                            st.write("##### Comparación de género")
                            fig = plot_crime_exploration_gender(df_crimes_exploration_gender)
                            fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})

                        with col7:
                            st.write("##### Distribución de la edad")
                            fig = plot_crime_exploration_age_group(df_crimes_exploration_age_group)
                            fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})
                            
                        with col8:
                            st.write("##### Comparación de momentos del día")
                            fig = plot_crime_exploration_day_parts(df_crimes_exploration_part_of_day)
                            fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})
                            
                        with col9:
                            st.write("##### Distancia delito-estación")
                            fig = plot_crime_exploration_distances(df_crimes_exploration_distances)
                            fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})
                    else:
                        st.write('Se encontraron 0 registros coincidentes para los filtros aplicados')


# Predictions view
def predictions():
    
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #5751a9;
        }
        [data-testid=stSidebar] h1 {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 Predicción del nivel de riesgo delictivo")
    
    col1, col2 = st.columns([1, 5])
    
    # First container.
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        
        list_categ_crimes_ = {
            'Robo a transeúnte y pasajero en transporte público': 'Robo a transeúnte y pasajero en transporte público',
            'Robo de vehículo y autopartes': 'Robo de vehículo y autopartes',
            'Delitos sexuales': 'Delitos sexuales',
            'Lesiones': 'Lesiones',
            'Amenazas': 'Amenazas',
            'Fraude': 'Fraude',
        }
        
        transport = col1.selectbox("Sistema", [
            "STC Metro",
            "Metrobús",
        ])
        level_div = 'Alcaldía'
        
        categ_crime = col2.selectbox("Categoría delictiva", list_categ_crimes_.keys())
        categ_crime_ok = list_categ_crimes_[categ_crime]
        
        sex = col3.selectbox("Sexo a considerar en la predicción", ["Ambos", "Femenino", "Masculino"])
        id_sex = 0
        if sex == 'Femenino':
            id_sex = 1
        
        col4, col5, col6 = st.columns([4, 1, 2])
        weeks_forward = col4.slider(f'No. de semanas futuras a predecir', 0, last_week_of_year - int(week_year), 0)
        week_month_aux = week_of_month(get_monday_week_year(weeks_forward + int(week_year), year))
        
        text_num_week_forward = 'semana '
        if weeks_forward == 0:
            text_num_week_forward += 'actual'
        else:
            text_num_week_forward += str(weeks_forward + int(week_year))
        
        
        # Validate which grouping dataset belongs the model (3, 4, 6, 7)
        grouped_dataset_id = 0
        columns_input_model = []
        regions = []
        if level_div == 'Alcaldía' and sex == 'Ambos':
            print('Agrupamiento 3')
            grouped_dataset_id = 3
            columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'semana_1',]
            columns_weekly_count = ['semana_anio_completa', 'semana_anio', 'anio', 'alcaldia', 'categoria_delito_adaptada', 'conteo']
            if transport == 'STC Metro':
                regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tláhuac', 'Venustiano Carranza', 'Álvaro Obregón']
            else:
                regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tlalpan', 'Venustiano Carranza', 'Álvaro Obregón', 'Xochimilco']
                
            inputs_model_ls = []
            for region in regions:
                inputs_model_ls.append([week_month_aux, region, categ_crime_ok])
            input_model_df_partial = pd.DataFrame(inputs_model_ls, columns=columns_input_model[:-1])
            
            # weekly_crime_counts = load_weekly_crime_counts(transport, grouped_dataset_id)[columns_weekly_count]
            # weekly_crime_counts = weekly_crime_counts.groupby(columns_weekly_count[:-1])['conteo'].sum().reset_index(name='conteo')
            # print(weekly_crime_counts)
            # weekly_crime_counts_filtered = weekly_crime_counts[(weekly_crime_counts['categoria_delito_adaptada'] == categ_crime_ok)].sort_values(by=['anio', 'semana_anio'])
            # print('\n')
            # print(weekly_crime_counts_filtered)
            
            thresholds_pred = load_thresholds_crime_model(transport, grouped_dataset_id)
            threshold_grouped_dataset = thresholds_pred[(thresholds_pred['categ_delito'] == categ_crime_ok)]['percentil'].to_list()[0]
            
        
        elif level_div == 'Alcaldía' and sex != 'Ambos':
            print('Agrupamiento 4')
            grouped_dataset_id = 4
            columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'sexo_victima', 'semana_1',]
            columns_weekly_count = ['semana_anio_completa', 'semana_anio', 'anio', 'alcaldia', 'categoria_delito_adaptada', 'id_sexo', 'conteo']
            if transport == 'STC Metro':
                regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tláhuac', 'Venustiano Carranza', 'Álvaro Obregón']
            else:
                regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tlalpan', 'Venustiano Carranza', 'Álvaro Obregón', 'Xochimilco']
                
            inputs_model_ls = []
            for region in regions:
                inputs_model_ls.append([week_month_aux, region, categ_crime_ok, id_sex, ])
            input_model_df_partial = pd.DataFrame(inputs_model_ls, columns=columns_input_model[:-1])
            
            # weekly_crime_counts = load_weekly_crime_counts(transport, grouped_dataset_id)[columns_weekly_count]
            # weekly_crime_counts = weekly_crime_counts.groupby(columns_weekly_count[:-1])['conteo'].sum().reset_index(name='conteo')
            # print(weekly_crime_counts)
            # weekly_crime_counts_filtered = weekly_crime_counts[(weekly_crime_counts['categoria_delito_adaptada'] == categ_crime_ok)
            #                                                    & (weekly_crime_counts['id_sexo'] == id_sex)].sort_values(by=['anio', 'semana_anio'])
            
            thresholds_pred = load_thresholds_crime_model(transport, grouped_dataset_id)
            threshold_grouped_dataset = thresholds_pred[(thresholds_pred['categ_delito'] == categ_crime_ok) & (thresholds_pred['sexo'] == id_sex)]['percentil'].to_list()[0]
            
        if level_div == 'Sector policial' and sex == 'Ambos':
            print('Agrupamiento 6')
            grouped_dataset_id = 6
            columns_input_model = ['semana_mes', 'sector', 'categoria_delito_adaptada', 'semana_1',]
            columns_weekly_count = ['semana_anio_completa', 'semana_anio', 'anio', 'sector', 'categoria_delito_adaptada', 'conteo']
            if transport == 'STC Metro':
                regions = ['Abasto-Reforma', 'Alameda', 'Aragón', 'Arenal', 'Asturias',
                        'Buenavista', 'Centro', 'Churubusco', 'Clavería', 'Congreso',
                        'Consulado', 'Coyoacán', 'Cuchilla', 'Del Valle', 'Estrella',
                        'Granjas', 'Hormiga', 'Iztaccíhuatl', 'La Raza', 'Lindavista',
                        'Merced-Balbuena', 'Mixquic', 'Moctezuma', 'Morelos',
                        'Narvarte-Álamos', 'Nativitas', 'Nápoles', 'Oasis', 'Pantitlán',
                        'Plateros', 'Polanco-Castillo', 'Portales', 'Quiroga', 'Roma',
                        'San Ángel', 'Tacuba', 'Tacubaya', 'Taxqueña', 'Teotongo',
                        'Tepeyac', 'Tezonco', 'Tlacotal', 'Tlatelolco', 'Universidad',
                        'Zapotitla', 'Zaragoza', 'Ángel', 'Ticomán']
            else:
                pass
        
        elif level_div == 'Sector policial' and sex != 'Ambos':
            print('Agrupamiento 7')
            grouped_dataset_id = 7
            columns_input_model = ['semana_mes', 'sector', 'categoria_delito_adaptada', 'sexo_victima', 'semana_1',]
            columns_weekly_count = ['semana_anio_completa', 'semana_anio', 'anio', 'sector', 'categoria_delito_adaptada', 'id_sexo', 'conteo']
            if transport == 'STC Metro':
                regions = ['Abasto-Reforma', 'Alameda', 'Aragón', 'Arenal', 'Asturias',
                        'Buenavista', 'Centro', 'Churubusco', 'Clavería', 'Congreso',
                        'Consulado', 'Coyoacán', 'Cuchilla', 'Del Valle', 'Estrella',
                        'Granjas', 'Hormiga', 'Iztaccíhuatl', 'La Raza', 'Lindavista',
                        'Merced-Balbuena', 'Mixquic', 'Moctezuma', 'Morelos',
                        'Narvarte-Álamos', 'Nativitas', 'Nápoles', 'Oasis', 'Pantitlán',
                        'Plateros', 'Polanco-Castillo', 'Portales', 'Quiroga', 'Roma',
                        'San Ángel', 'Tacuba', 'Tacubaya', 'Taxqueña', 'Teotongo',
                        'Tepeyac', 'Tezonco', 'Tlacotal', 'Tlatelolco', 'Universidad',
                        'Zapotitla', 'Zaragoza', 'Ángel', 'Ticomán']
            else:
                pass
        
        print('Resultados')
        
        # Load of pickle model and reading of affluence predictions
        crime_model = load_crime_model(transport, grouped_dataset_id)
        afflu_fc_values = load_afflu_forecast_values(transport, grouped_dataset_id)
        afflu_fc_values_filtered = afflu_fc_values[afflu_fc_values['semana_anio'] == weeks_forward + int(week_year) - 1]
        input_model_df = input_model_df_partial.merge(afflu_fc_values_filtered, left_on=columns_input_model[1], right_on=['region'])
        input_model_df.rename(columns={'afluencia': 'semana_1'}, inplace=True)
        input_model_df = input_model_df[columns_input_model]
        print(input_model_df)
        
        # Predictions
        preds = crime_model.predict(input_model_df)
        print('Predicciones para semana {}:'.format(weeks_forward + int(week_year)))
        df_preds = pd.DataFrame(columns=['valor'])
        df_preds['valor'] = preds
        df_preds['valor'] = df_preds['valor'].replace({'High': 'Riesgo elevado', 'Low': 'Riesgo moderado'})
        print(df_preds)
        input_model_df_preds = pd.concat([input_model_df, df_preds], axis=1)
        
        # Get the spatial granularity for then pass it as an arg to the predictive map plot function
        if level_div == 'Alcaldía':
            region_gdf_merge = munics_gdf.merge(input_model_df_preds[['alcaldia', 'valor']], left_on=['NOMGEO'], right_on=['alcaldia'])
            region_column = 'CVE_MUN'
            region_column_name = 'NOMGEO'
            region_column_name_2 = 'alcaldia'
        else:
            region_gdf_merge = munics_gdf.merge(input_model_df_preds[['sector', 'valor']], left_on=['sector'], right_on=['sector'])
            region_column = 'sector'
            region_column_name = 'sector'
            region_column_name_2 = 'sector'
        region_gdf_merge = region_gdf_merge.sort_values(by=['valor'])
        print(region_gdf_merge)
    
    with st.container():
        metric_unique_values = np.sort(region_gdf_merge['valor'].unique())
        col7, col78, col8 = st.columns([20, 1, 30])
        with col7:
            fig = plot_predictive_map(region_gdf_merge, transport, metric_unique_values)
            fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
            # Interactive map was done thanks to
            # https://discuss.streamlit.io/t/interactive-plot-get-which-point-a-user-clicked/14596
            # https://github.com/null-jones/streamlit-plotly-events
            # https://orosz-attila-covid-19-dashboard-streamlit-app-kmmvgj.streamlit.app/
            #https://towardsdatascience.com/highlighting-click-data-on-plotly-choropleth-map-377e721c5893
            selected_click_aux = plotly_events(fig, click_event=True,)
            st.session_state.selected_click_pred_map.append(selected_click_aux)
            # print('click', st.session_state.selected_click_pred_map)
           
        with col8:
            if len(st.session_state.selected_click_pred_map) > 0:
                last_click = st.session_state.selected_click_pred_map[-1]
                if len(last_click) > 0:
                    #col8.write(last_click[0])
                    metric_value = metric_unique_values[last_click[0]['curveNumber']]
                    region_gdf_merge_ = region_gdf_merge[region_gdf_merge['valor'] == metric_value]
                    region_gdf_aux = region_gdf_merge_.iloc[last_click[0]['pointIndex']]
                    region_gdf_aux = pd.DataFrame(region_gdf_aux.values.reshape(1, -1), columns=region_gdf_aux.index)
                    region_name = region_gdf_aux[region_column_name].to_list()[0]
                    level_risk = region_gdf_aux['valor'].to_list()[0]
                    print(region_name)
                    print(level_risk)
                    #col8.write(munics_gdf_aux)
                    #col8.write(st.session_state.selected_click_pred_map)
                    #col8.markdown(f'##\t{region_name}')
                    if level_risk == 'Riesgo elevado':
                        threshold_txt = f'{str(int(threshold_grouped_dataset + 1))}+ hechos delictivos'
                        # Checar luego para alinear texto si queremos
                        col8.markdown(f'<span style="color: black; font-size: 20px; font-weight: bold;">{level_risk} ({threshold_txt})</span> <br><span style="color: black; font-size: 14px;">para las estaciones en <b>{region_name}</b> durante la <b>{text_num_week_forward}</b></span> <span style="color: #a5a5a5; font-size: 14px;">({get_week_date_range(weeks_forward + int(week_year), year)})</span>', unsafe_allow_html=True)
                    else:
                        if threshold_grouped_dataset > 0:
                            threshold_txt = f'0 a {str(int(threshold_grouped_dataset))} hechos delictivos'
                        else:
                            threshold_txt = 'casi nulo'
                        # Checar luego para alinear texto si queremos
                        col8.markdown(f'<span style="color: black; font-size: 20px; font-weight: bold;">{level_risk} ({threshold_txt})</span> <br><span style="color: black; font-size: 14px;">para las estaciones en <b>{region_name}</b> durante la <b>{text_num_week_forward}</b></span> <span style="color: #a5a5a5; font-size: 14px;">({get_week_date_range(weeks_forward + int(week_year), year)})</span>', unsafe_allow_html=True)
                    
                    fig2 = plot_complementary_predictive_map(region_gdf_merge, transport, region_gdf_aux, region_column)
                    fig2.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False,})
                    
                    
                    with st.expander("Estaciones afectadas", expanded=False):
                        if region_column == 'CVE_MUN':
                            df_stations_metro_aux = df_stations_metro[(df_stations_metro['alcaldia'] == region_name)]
                        else:
                            df_stations_metro_aux = df_stations_metro[(df_stations_metro['sector'] == region_name)]
                        
                        # Stations within the region
                        lines = df_stations_metro_aux['linea'].unique()
                        
                        list_text_stations_per_line = []
                        for l in lines:
                            stations_per_line = []
                            stations_line_aux = df_stations_metro_aux[(df_stations_metro_aux['linea'] == l)]
                            for ind, row in stations_line_aux.iterrows():
                                name_st = row['nombre']
                                stations_per_line.append(name_st)
                            text_aux = ""
                            if len(stations_per_line) == 1:
                                text_aux = l + ": " + stations_per_line[0]
                            else:
                                text_aux = l + ": " + ", ".join(stations_per_line[:-1]) + " y " + stations_per_line[-1]
                            list_text_stations_per_line.append(text_aux)

                        for elemento in list_text_stations_per_line:
                            st.write(f'<li style="font-size: 12px;">{elemento}</li>', unsafe_allow_html=True)
                        radio_ = '540' if transport == 'STC Metro' else '270'
                        
                        st.write(f'<span style="font-size: 12px; color: #a5a5a5"><b>Nota:</b> El nivel de riesgo predictivo se predice considerando la evidencia delictiva tanto dentro como en un radio de {radio_}m alrededor de las estaciones, para el caso del {transport}.</span>', unsafe_allow_html=True)
                else:
                    st.subheader("Seleccione una alcaldía")
                    st.write("Al seleccionar se muestran las estaciones afectadas en tal alcaldía, así como detalles sobre la predicción")                   
            else:
                st.subheader("Seleccione una alcaldía")
                st.write("Al seleccionar se muestran las estaciones afectadas en tal alcaldía, así como detalles sobre la predicción")                   
       
    with st.container():
        pass

# SIDEBAR
# Auxiliar links to generate CSS at streamlit
# https://discuss.streamlit.io/t/button-css-for-streamlit/45888/3
# https://discuss.streamlit.io/t/button-size-in-sidebar/36132/2
# https://discuss.streamlit.io/t/image-and-text-next-to-each-other/7627/7

# Style for buttons sidebar
st.markdown(
    """
    <style>
        /* Estilo para los botones del sidebar */
        section[data-testid="stSidebar"] div.stButton button {
            background-color: transparent;
            border: none;
            width: 200px;
            color: black;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        /* Estilo para el botón seleccionado */
        section[data-testid="stSidebar"] div.stButton button:active {
            background-color: #f0f0f080;
            color: black;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        section[data-testid="stSidebar"] div.stButton button:focus {
            background-color: #f0f0f080;
            color: black;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        section[data-testid="stSidebar"] div.stButton button:hover {
            /* background-color: #f0f0f0; */
            color: white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        /* Alineación horizontal de los elementos del div con data-testid="stVerticalBlock" */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Script to manage the selected state of buttons (not successful)
st.markdown(
    """
    <script>
        const buttons = document.querySelectorAll('section[data-testid="stSidebar"] div.stButton button');

        buttons.forEach(button => {
            button.addEventListener('click', () => {
                buttons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
            });
        });

        // Evitar que los botones pierdan el estado activo al hacer clic en otro lugar de la página
        document.addEventListener('click', (event) => {
            if (!event.target.closest('section[data-testid="stSidebar"]')) {
                buttons.forEach(btn => btn.classList.remove('active'));
            }
        });
    </script>
    """,
    unsafe_allow_html=True
)

# Change the view state when a sidebar's button is clicked
with st.sidebar:
    st.title("Menú")
    
    if st.button("Inicio"):
        st.session_state.selection = "INICIO"
    if st.button("Tendencias"):
        st.session_state.selection = "TENDENCIAS"
        st.session_state.transport = 'STC Metro'
    if st.button("Explora"):
        st.session_state.selection = "EXPLORA"
        st.session_state.selected_id = []
    if st.button("Predicciones"):
        st.session_state.selection = "PREDICCIONES"
        st.session_state.selected_click_pred_map = []

# Execute action depending on the above selection
if "selection" not in st.session_state:
    home()
else:
    if st.session_state.selection == "INICIO":
        home()
    if st.session_state.selection == "TENDENCIAS":
        trends()
    if st.session_state.selection == "EXPLORA":
        exploration()
    if st.session_state.selection == "PREDICCIONES":
        #if len(st.session_state.selected_click_pred_map) > 0:
        #    pass
        #else:
        #    st.session_state.selected_click_pred_map = []
        predictions()
        
        
