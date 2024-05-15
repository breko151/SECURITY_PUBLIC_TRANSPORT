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
import geopandas
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
import seaborn as sns
import matplotlib.pyplot as plt

# Datetime
from datetime import datetime, timedelta

# Constants
from colors import LINESM, LINESMB, LINESM_aux, LINESMB_aux

# Auxiliar proper modules
from plots import plot_top_stations_affluence_trends, plot_top_stations_crime_trends
from querys import query_top_stations_affluence_trends, query_top_stations_crime_trends

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

#Successful trial to remove top blankspace
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
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))

# Get the monday for a certain week of the year (auxiliary to show the range of a certain week)
def get_monday_week_year(week, year):
    first_day_year = datetime(year, 1, 1)
    monday_first_week = first_day_year - timedelta(days=first_day_year.weekday())
    monday_week_date = monday_first_week + timedelta(weeks=week - 1)

    return monday_week_date

# Get the week range (init_date_week to final_date_week)
def get_week_date_range(week_number, year):
    start_date = datetime(year, 1, 1)
    days_offset = (7 - start_date.weekday()) % 7
    start_date += timedelta(days=days_offset)
    start_week_date = start_date + timedelta(weeks=week_number - 1)
    end_week_date = start_week_date + timedelta(days=6)
    #start_date_str = start_week_date.strftime('%d-%m-%Y')
    #end_date_str = end_week_date.strftime('%d-%m-%Y')
    start_date_str = f"{start_week_date.day} de {month_names[start_week_date.month]} de {start_week_date.year}"
    end_date_str = f"{end_week_date.day} de {month_names[end_week_date.month]} de {end_week_date.year}"
    
    return f"{start_date_str} al {end_date_str}"

# Get columns from cve_est.
def get_station(df, cve_est, column=None):
    if column is None:
        result = df[df['cve_est'] == cve_est]
    else:
        filter = df[df['cve_est'] == cve_est]
        print(filter)
        result = filter[column].to_list()[0]
    
    return result


# INITIALIZATION OF VARIABLES

# List of default regions and lines
zones_ls = ["Ciudad de México", "Centro", "Norte", "Sur", "Oriente", "Poniente"]
munics_ls = {
    'STC Metro': ['Álvaro Obregón', 'Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
       'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
       'Tláhuac', 'Venustiano Carranza'],
    'Metrobús': ['Álvaro Obregón', 'Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
       'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
       'Tlalpan', 'Venustiano Carranza', ],
}
lines_ls = {
    'STC Metro': ['Línea 1', 'Línea 2', 'Línea 3', 'Línea 4', 'Línea 5', 'Línea 6', 'Línea 7', 'Línea 8',
                  'Línea 9', 'Línea 12', 'Línea A', 'Línea B',],
    'Metrobús': ['Línea 1', 'Línea 2', 'Línea 3', 'Línea 4', 'Línea 5', 'Línea 6', 'Línea 7',],
}

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

# Clicks at prediction map
if 'selected_click_pred_map' not in st.session_state:
    st.session_state.selected_click_pred_map = []

# Geodata
df_stations = pd.read_csv("./fact_constellation_schema/dim_estaciones_espacio_ok.csv")
df_stations['cve_mun_inegi'] = df_stations['cve_mun_inegi'].astype(str).str.zfill(3)
metro_lines = geopandas.read_file('./shapefiles/metro/STC_Metro_lineas_utm14n_repr.shp', index=False)
mb_lines = geopandas.read_file('./shapefiles/mb/Metrobus_lineas_utm14n_repr.shp', index=False)
munics_gdf = geopandas.read_file('./shapefiles/alcaldias/alcaldias_geo.shp', index=False)
police_sectors_gdf = geopandas.read_file('./shapefiles/cuadrantes/sectores_agrupados_ssc.shp', index=False)
lineas_cdmx = geopandas.read_file('./images/cdmx.json', encoding='utf-8')
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
file_image_home = open(image_home_logo_url, "rb")
contents = file_image_home.read()
data_url_image_home = base64.b64encode(contents).decode("utf-8")
file_image_home.close()

metro_logo_url = "./images/logo_metro.png"
file_metro_logo = open(metro_logo_url, "rb")
contents = file_metro_logo.read()
data_url_metro_logo = base64.b64encode(contents).decode("utf-8")
file_metro_logo.close()

metrobus_logo_url = "./images/logo_metrobus.png"
file_metrobus_logo = open(metrobus_logo_url, "rb")
contents = file_metrobus_logo.read()
data_url_metrobus_logo = base64.b64encode(contents).decode("utf-8")
file_metrobus_logo.close()


# LOAD DATA OR MODELS FUNCTIONS

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
        if grouped_dataset_id == 3 or grouped_dataset_id == 4:
            df = pd.read_csv('./predictions_sarima/predicciones_afluencia_alcaldia_semana_metro.csv')
        elif grouped_dataset_id == 6 or grouped_dataset_id == 7:
            df = pd.read_csv('./predictions_sarima/predicciones_afluencia_sector_policial_semana_metro.csv')
    else:
        if grouped_dataset_id == 3 or grouped_dataset_id == 4:
            df = pd.read_csv('./predictions_sarima/predicciones_afluencia_alcaldia_semana_metrobus.csv')
        elif grouped_dataset_id == 6 or grouped_dataset_id == 7:
            df = pd.read_csv('./predictions_sarima/predicciones_afluencia_sector_policial_semana_metrobus.csv')
    
    return df

# Load of weekly crime counts values
@st.cache_resource
def load_weekly_crime_counts(transport: str, grouped_dataset_id: int):
    if transport == 'STC Metro':
        df = pd.read_csv('./datasets_aux/test/carpetas_afluencia_metro_grupo_{}_wm_final_red.csv'.format(grouped_dataset_id))
    else:
        df = pd.read_csv('./datasets_aux/test/carpetas_afluencia_metrobus_grupo_{}_wm_final_red.csv'.format(grouped_dataset_id))
    
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

# Initialize map of cdmx
def init_map(center=(19.4326018, -99.1332049), zoom_start=11, map_type="cartodbpositron", height=300, width=100):
    
    return folium.Map(location=center, zoom_start=zoom_start, tiles=map_type, height=height)

# Plot map of stations
def plot_stations(df, folium_map, type):
    # Validation of type of transport.
    if type == 'METRO':
        # Add every station.
        for i, row in df.iterrows():
            # Customize icon.
            icon = folium.CustomIcon(
                POINTSM[row.linea],
                icon_size=(15, 15)
            )
            folium.Marker(
                [row.latitud, row.longitud],
                icon=icon,
                tooltip=f'{row.cve_est}: {row.nombre} (línea {row.linea[1:]})'
            ).add_to(folium_map)
        
        # Add every line.
        metro_lines['geometry_yx'] = metro_lines['geometry'].apply(lambda line: LineString([(point[1], point[0]) for point in line.coords]))
        for i, row in metro_lines.iterrows():
            coords_pts = [list(coord) for coord in row['geometry_yx'].coords]
            folium.PolyLine(coords_pts, color=LINESM[row['LINEA']]).add_to(folium_map)
    
    elif type == 'METROBUS':
        # Add every station.
        for i, row in df.iterrows():
            # Customize icon.
            icon = folium.CustomIcon(
                POINTSMB[row.linea],
                icon_size=(15, 15)
            )
            folium.Marker(
                [row.latitud, row.longitud],
                icon=icon,
                tooltip=f'{row.cve_est}: {row.nombre} (línea {row.linea[1:]})'
            ).add_to(folium_map)
        
        # Add every line.
        mb_lines['geometry_yx'] = mb_lines['geometry'].apply(lambda line: LineString([(point[1], point[0]) for point in line.coords])) 
        for i, row in mb_lines.iterrows():
            coords_pts = [list(coord) for coord in row['geometry_yx'].coords]
            folium.PolyLine(coords_pts, color=LINESMB[row['LINEA']]).add_to(folium_map)
        
    # Add CDMX shapefile to map.
    folium.GeoJson(lineas_cdmx_json, 
               name=None, 
               style_function=lambda x: {'color': 'black', 'weight': 2, 'fillColor': 'transparent'}).add_to(folium_map) 

    return folium_map



# PREDICTION SECTION

# Plot of time series weekly crime
def plot_ts_weekly_crime(df, title, threshold):
    fig = go.Figure()
    x = df['semana_anio_completa']
    print(df)
    y = df['conteo']
    
    # Bicolor graph
    # https://stackoverflow.com/questions/65931888/how-to-fill-colors-on-a-plotly-chart-based-on-y-axis-values
    
    colors_ = ['#5bc7da', '#b21800']
    risks_ = ['Riesgo moderado', 'Riesgo alto']
    colors = [colors_[0] if val <= threshold else colors_[1] for val in y]
    
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='', showlegend=False,
                             line=dict(color='#b9b9b9', width=1)))
    
    fig.add_shape(type='line', x0=x.iloc[0], x1=x.iloc[-1], y0=threshold + 0.0, y1=threshold + 0.0,
                  line=dict(color='#ff6c00', width=1.5, ), name='Límite de riesgo moderado', showlegend=True)
    
    for i in range(len(colors_)):
        color_ = colors_[i]
        risk_ = risks_[i]
        color_mask = [c == color_ for c in colors]
        fig.add_trace(go.Scatter(
                x=x[color_mask],
                y=y[color_mask],
                mode='markers',
                name=risk_, 
                marker=dict(color=color_, size=4,),
                #line=dict(color='#b9b9b9', width=1,),
                showlegend=True,
            )
        )
    
    tick_indices = list(range(0, len(x), 52))
    tick_labels = [x.iloc[i] for i in tick_indices]

    fig.update_xaxes(tickvals=tick_indices, ticktext=tick_labels)

    fig.update_layout(
        title=title,
        title_font=dict(
            #family="Arial", 
            size=14,      
            color="navy",
        ),
        xaxis_title='Semana',
        yaxis_title='Conteo de delitos',
        height=300,
        margin=dict(
            t=30,
            b=0,
            l=0,
            r=0,
        ),
        legend=dict(
            x = 0.51,
            y = 1.42,
            #xanchor = 'right',
            #yanchor = 'top',
            #xanchor='auto', yanchor='auto',
            #bgcolor='rgba(255, 255, 255, 0.5)', 
            #bordercolor='rgba(0, 0, 0, 0.5)',
            # borderwidth=1,
        ),
    )

    return fig
    
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

    # Las figuras con go las pude hacer gracias a:
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
    
    # Si todo el mapa va a tener el mismo color validamos porque sino hay un bug de visualización
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
    # Si el mapa si tiene valores de ambas clases
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
        mapbox=dict(center=dict(lat=center_geom.y, lon=center_geom.x),zoom=9.8),
    )

    # Las figuras con go las pude hacer gracias a:
    # https://community.plotly.com/t/annotations-on-plotly-choropleth-choropleth-mapbox-scattermapbox/74556/6
    # https://stackoverflow.com/questions/68709936/how-to-plot-a-shp-file-with-plotly-go-choroplethmapbox
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Choroplethmapbox.html
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.choroplethmapbox.html#plotly.graph_objects.choroplethmapbox.ColorBar
    
    # Mostrar estaciones y líneas
    if transport == 'STC Metro':
        lines_unique = df_stations_metro['linea'].unique()
        
        # (Enfoque omitido, puesto que ahora se grafican todas las líneas y estaciones)
        # Se filtran aquellas líneas que intersectan (caen dentro) con la demarcación de la alcaldía en cuestión
        # for line in lines_unique:
        #     metro_lines_aux = metro_lines[metro_lines['LINEA'] == line[1:]]
        #     metro_lines_aux['geometry_yx'] = metro_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
        #     lines_ = metro_lines_aux['geometry_yx'].iloc[0]
        #     polygon_ = region_gdf_cp.loc[region_gdf_cp[region_column] == cve_region_selected, 'geometry'].to_list()[0]
        #     intersection = lines_.intersection(polygon_)
        #     if intersection.type == 'MultiLineString':
        #         # Oro molido
        #         # https://gis.stackexchange.com/questions/456266/error-of-multilinestring-object-is-not-iterable
        #         for ind_line in list(intersection.geoms):
        #             if ind_line.is_empty:
        #                 #print("No hay intersección")
        #                 pass
        #             else:
        #                 #print('Intersecta')
        #                 coords_pts = [[coord[0], coord[1]] for coord in ind_line.coords]
        #                 line_trace = go.Scattermapbox(
        #                     mode='lines',
        #                     lon = [coord[0] for coord in coords_pts],
        #                     lat = [coord[1] for coord in coords_pts],
        #                     line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
        #                     hoverinfo='none',
        #                 )
        #                 fig.add_trace(line_trace)
                
        #     else:
        #         if intersection.is_empty:
        #             #print("No hay intersección")
        #             pass
        #         else:
        #             #print('Intersecta')
        #             coords_pts = [[coord[0], coord[1]] for coord in intersection.coords]
        #             line_trace = go.Scattermapbox(
        #                 mode='lines',
        #                 lon = [coord[0] for coord in coords_pts],
        #                 lat = [coord[1] for coord in coords_pts],
        #                 line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
        #                 hoverinfo='none',
        #             )
                    
        #             fig.add_trace(line_trace)
        
        # # Se filtran las estaciones que caen dentro de la alcaldía
        # for line in lines_unique:
        #     if region_column == 'CVE_MUN':
        #         region_column_aux = 'cve_mun_inegi'
        #     else:
        #         region_column_aux = 'sector'
        #     df_stations_metro_aux = df_stations_metro[(df_stations_metro['linea'] == line) & (df_stations_metro[region_column_aux] == cve_region_selected)]
        #     lats = df_stations_metro_aux['latitud']
        #     lons = df_stations_metro_aux['longitud']
        #     ids = df_stations_metro_aux['cve_est']
        #     lines = df_stations_metro_aux['linea']
        #     names = df_stations_metro_aux['nombre']
        #     scatter_trace = go.Scattermapbox(lat=lats,
        #             lon=lons,
        #             mode='markers',
        #             #text=ids,
        #             customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
        #             textposition='top center',
        #             marker_size=4,
        #             marker_color='black',
        #             hovertext=ids,
        #             hoverlabel=dict(namelength=0),
        #             hovertemplate=''
        #     )
        #     # scatter_trace_2 = go.Scattermapbox(lat=lats,
        #     #         lon=lons,
        #     #         mode='markers',
        #     #         #text=ids,
        #     #         customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
        #     #         textposition='top center',
        #     #         marker_size=10,
        #     #         marker_color='white',
        #     #         hovertext=ids,
        #     #         hoverlabel=dict(namelength=0),
        #     #         hovertemplate=''
        #     # )
        #     scatter_trace_3 = go.Scattermapbox(lat=lats,
        #             lon=lons,
        #             mode='markers',
        #             #text=ids,
        #             customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
        #             textposition='top center',
        #             marker_size=3,
        #             marker_color=LINESM_aux[line],
        #             hovertext=ids,
        #             hoverlabel=dict(namelength=0),
        #             hovertemplate='<b>Estación:</b> %{customdata[2]} (%{customdata[1]})<br>'
        #     )
        #     fig.add_trace(scatter_trace)
        #     #fig.add_trace(scatter_trace_2)
        #     fig.add_trace(scatter_trace_3)
        
        # Mapeamos todas las líneas
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
                    
        
        # Se filtran las estaciones que caen dentro de la región seleccionada para pintarlas de cierto color, y las demás de otro
        for line in lines_unique:
            if region_column == 'CVE_MUN':
                region_column_aux = 'cve_mun_inegi'
            else:
                region_column_aux = 'sector'
            
            # Estaciones dentro de la región
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
                    marker_size=8,
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
                    marker_size=5,
                    marker_color='white',
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
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
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>'
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)
            fig.add_trace(scatter_trace_3)
            
            
            # Estaciones fuera de la región
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
                    marker_size=7,
                    marker_color='black',
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
                    marker_size=5,
                    marker_color='white',
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_3_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker_size=3,
                    marker_color=LINESM_aux[line],
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>'
            )
            #fig.add_trace(scatter_trace_out)
            fig.add_trace(scatter_trace_2_out)
            fig.add_trace(scatter_trace_3_out)

    else:
        lines_unique = df_stations_metrobus['linea'].unique()
        #print(lines_unique)
        #print(mb_lines['LINEA'].unique())
        # Mapeamos todas las líneas
        for line in lines_unique:
            #print(line)
            metrobus_lines_aux = mb_lines[mb_lines['LINEA'].str[-1] == line[1:]]
            metrobus_lines_aux['geometry_yx'] = metrobus_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
            #print(metrobus_lines_aux)
            lines_ = metrobus_lines_aux['geometry_yx']
            #print(metrobus_lines_aux['geometry_yx'])
            #print(len(lines_.type))
            
            for i in range(len(lines_)):
                line_ = lines_.iloc[i]
                coords_pts = [[coord[0], coord[1]] for coord in line_.coords]
                #print('llega')
                #print(LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]])
                line_trace = go.Scattermapbox(
                    mode='lines',
                    lon = [coord[0] for coord in coords_pts],
                    lat = [coord[1] for coord in coords_pts],
                    line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                    hoverinfo='none',
                )
                
                fig.add_trace(line_trace)
                
            
        
        # Se filtran las estaciones que caen dentro de la región seleccionada para pintarlas de cierto color, y las demás de otro
        for line in lines_unique:
            if region_column == 'CVE_MUN':
                region_column_aux = 'cve_mun_inegi'
            else:
                region_column_aux = 'sector'
            
            # Estaciones dentro de la región
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
                    marker_size=8,
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
                    marker_size=5,
                    marker_color='white',
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
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
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>'
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)
            fig.add_trace(scatter_trace_3)
            
            
            # Estaciones fuera de la región
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
                    marker_size=7,
                    marker_color='black',
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
                    marker_size=5,
                    marker_color='white',
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_3_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker_size=3,
                    marker_color=LINESM_aux[line],
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>'
            )
            #fig.add_trace(scatter_trace_out)
            fig.add_trace(scatter_trace_2_out)
            fig.add_trace(scatter_trace_3_out)
    
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
        height = 300,
        autosize=True,
        dragmode=False,
    )

    # Trazar contorno de la región seleccionada
    geometry_ = region_gdf_cp['geometry'].iloc[0]
    lon_geom, lat_geom = geometry_.exterior.xy
    lon_geom = np.array(lon_geom).tolist()
    lat_geom = np.array(lat_geom).tolist()
    
    colorscales = [
        'rgba(255,84,84,0.3)',
        #'rgba(176,242,244,0.3)',
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
def home():
    
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
    #st.image("./images/MapaCDMX.png", width=300)
    st.markdown(
        r'<div style="{}"><img src="data:image/gif;base64,{}" alt="Imagen home" width=500 ></div>'.format(center_css, data_url_image_home),
        unsafe_allow_html=True,
    )
    

# Metro view.
def metro():
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
    
    # if st.session_state.transport == 'STC Metro':
    #     st.markdown("""
    #     <style>
    #         [data-testid=stSidebar] {
    #             background-color: #e8540c;
    #         }
    #         [data-testid=stSidebar] h1 {
    #             color: white;
    #         }
    #     </style>
    #     """, unsafe_allow_html=True)
        
    #     col1, col2 = st.columns([1, 5])
    #     with col1:
    #         st.markdown("<div style='{}'><img src='data:image/gif;base64,{}' width='90' style='margin-right:0px;'></div>".format(center_css, data_url_metro_logo), unsafe_allow_html=True)
    #     with col2:
    #         st.title('STC Metro')
    #         #st.markdown("<div style='display: flex; justify-content: left; align-items: center;'><h1 style='text-align: left;'>STC Metro</h1></div>", unsafe_allow_html=True)
    #     st.markdown("<br>", unsafe_allow_html=True)
    # else:
    #     st.markdown("""
    #     <style>
    #         [data-testid=stSidebar] {
    #             background-color: #c80f2e;
    #         }
    #         [data-testid=stSidebar] h1 {
    #             color: white;
    #         }
    #     </style>
    #     """, unsafe_allow_html=True)
        
    #     st.title("Metrobús")
    
    # First container.
    with st.container():
        st.header("🔥 Tendencias")
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            transport = st.selectbox("Sistema", ["STC Metro", "Metrobús"])
            st.session_state.transport = transport
        
        with col2:
            level_div = st.selectbox("Nivel de filtrado", ["Zona", "Alcaldía", "Línea"])
            
        with col3:
            if level_div == 'Zona':
                zone = st.selectbox("Escoge una zona", zones_ls)
            elif level_div == 'Alcaldía':
                zone = st.selectbox("Escoge una alcaldía", munics_ls[st.session_state.transport])
            elif level_div == 'Línea':
                zone = st.selectbox("Escoge una línea", lines_ls[st.session_state.transport])
        
        df_top_stations_affluence_trends = query_top_stations_affluence_trends('STC Metro', zone, weekday, week_year, 10)
        df_top_stations_crime_trends = query_top_stations_crime_trends('STC Metro', zone, weekday, week_year, 540, 10)
        
        
        col3, col4 = st.columns(2)
        # Column 1.
        col3.write("##### Top 10 estaciones con más afluencia")
        col3.write("🗓️ Datos históricos para <b>{}</b> de la <b>semana {}</b> de años pasados.".format(weekday.lower(), week_year), unsafe_allow_html=True)
        col3.plotly_chart(plot_top_stations_affluence_trends(df_top_stations_affluence_trends, 'STC Metro'), use_container_width=True)
        
        # Column 2.
        col4.write("##### Top 10 estaciones más delictivas")
        col4.write("🗓️ Datos históricos para <b>{}</b> de la <b>semana {}</b> de años pasados.".format(weekday.lower(), week_year), unsafe_allow_html=True)
        col4.plotly_chart(plot_top_stations_crime_trends(df_top_stations_crime_trends, 'STC Metro'), use_container_width=True)
        
    #st.write('poner selector de tipo desglose: zona, alcaldía, línea y en otro selector los posibles valores')
    
    # Second container.
    with st.container():
        st.header("🔍 Explora las estaciones")
        col1, col2 = st.columns(2)
        # Map column.
        with col1:
            m = init_map()
            m = plot_stations(df_stations_metro, m, 'METRO')
            level1_map_data = st_folium(m)
            st.session_state.selected_id = level1_map_data['last_object_clicked_tooltip']
        # Second column validation.
        if 'selected_id' not in st.session_state:
            col2.subheader("Es necesario seleccionar una estación")
            col2.write("Al seleccionar alguna aparecerá la siguiente información sobre los hechos delictivos ocurridos dentro un radio de 540 metros alrededor de las estaciones:")
            col2.write(" - Top delitos más frecuentes")
            col2.write(" - Comparación de géneros de víctimas")
            col2.write(" - Rangos de edad más vulnerables")
            col2.write(" - Comportamiento de la distancia delito-estación")
            col2.write(" - Partes del día más delictivas")
        else:
            if st.session_state.selected_id is not None:
                cve_est = st.session_state.selected_id.split(":")[0]
                name_est = get_station(df_stations_metro, cve_est, "nombre")
                tipo_est = get_station(df_stations_metro, cve_est, "tipo")
                linea_est = get_station(df_stations_metro, cve_est, "linea")
                col2.subheader(f'{name_est.upper()}')
                col2.write(f'LÍNEA: {linea_est[1:]}')
                col2.write(f'ESTACIÓN TIPO: {tipo_est.upper()}')
                col2.write("##### TOP DELITOS")
                data = {
                    'Tipo de Delito': ['Robo', 'Asalto', 'Homicidio'],
                    'Frecuencia': [20, 15, 5]  # Ejemplo de frecuencia de delitos
                }
                df = pd.DataFrame(data)
                plt.barh(df['Tipo de Delito'], df['Frecuencia'], color='skyblue', edgecolor='black')
                plt.xlabel('Tipo de Delitos')
                plt.ylabel('Frecuencia de Delitos')
                plt.title('Histograma de Frecuencia de Delitos')
                plt.grid(True)
                col2.pyplot(plt)
                col2.write("##### COMPARACIÓN DE GÉNEROS")
                # Datos de ejemplo
                labels = ['HOMBRE', 'MUJER']
                sizes = [45, 55]
                # Configuración de colores
                colors = sns.color_palette('pastel')
                # Crear gráfico de pastel
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
                ax.axis('equal')  # Equal aspect ratio para asegurar que es un círculo
                ax.set_title('Ejemplo de Gráfico de Pastel con Seaborn', fontsize=16)
                col2.pyplot(fig)
                col2.write("###### EDAD")
                data2 = {
                    'Edad': ['[10-20]', '[40-50]', '[90-100]'],
                    'Frecuencia': [20, 15, 5]  # Ejemplo de frecuencia de delitos
                }
                df2 = pd.DataFrame(data2)
                fig2, ax2 = plt.subplots()
                ax2.bar(df2['Edad'], df2['Frecuencia'], color='skyblue', edgecolor='black')
                col2.pyplot(fig2)
                col2.divider()
                col2.write("###### DISTANCIAS DE LOS DELITOS")
                # Generar datos aleatorios para el histograma
                np.random.seed(0)
                data = np.random.randn(1000)
                # Definir el número de bins para el histograma
                plt.figure(figsize=(8, 6))
                plt.hist(data, bins=100, orientation='horizontal', color='skyblue')
                plt.xlabel('Frecuencia')
                plt.ylabel('xlabel')
                plt.title('Histograma Horizontal')
                plt.grid(True)
                col2.pyplot(plt)
            else:
                col2.subheader("Es necesario seleccionar una estación")
                col2.write("Al seleccionar alguna aparecerá la siguiente información sobre los hechos delictivos ocurridos dentro un radio de 540 metros alrededor de las estaciones:")
                col2.write(" - Top delitos más frecuentes")
                col2.write(" - Comparación de géneros de víctimas")
                col2.write(" - Rangos de edad más vulnerables")
                col2.write(" - Comportamiento de la distancia delito-estación")
                col2.write(" - Partes del día más delictivas")

            
# Metrobus view
def metrobus():
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
    
    st.title("Metrobús")
    # First container.
    with st.container():
        st.header("TENDENCIAS")
        opt = st.selectbox("ESCOGE UNA ZONA", ["CIUDAD DE MÉXICO", "NORTE", "SUR", "ORIENTE", "PONIENTE"])
        st.write(opt)
        col1, col2 = st.columns(2)
        # Column 1.
        col1.write("##### TOP AFLUENCIAS")
        data = {
            'Estación': ['Hidalgo', 'Juárez', 'Centro Médico'],
            'Frecuencia': [20, 15, 5]  # Ejemplo de frecuencia de delitos
        }
        df = pd.DataFrame(data)
        plt.barh(df['Estación'], df['Frecuencia'], color='skyblue', edgecolor='black')
        plt.xlabel('Estación')
        plt.ylabel('Frecuencia de Afluencias')
        plt.title('Histograma de Frecuencia de Afluencias')
        plt.grid(True)
        col1.pyplot(plt)
        # Column 2.
        col2.write("##### TOP DELECTIVAS")
        data = {
            'Estación': ['Hidalgo', 'Juárez', 'Centro Médico'],
            'Frecuencia': [20, 15, 5]  # Ejemplo de frecuencia de delitos
        }
        df = pd.DataFrame(data)
        plt.barh(df['Estación'], df['Frecuencia'], color='skyblue', edgecolor='black')
        plt.xlabel('Estación')
        plt.ylabel('Frecuencia de Afluencias')
        plt.title('Histograma de Frecuencia de Délitos')
        plt.grid(True)
        col2.pyplot(plt)
    # Second container.
    with st.container():
        st.header("ESTACIONES")
        col1, col2 = st.columns(2)
        # Map column.
        with col1:
            m = init_map()
            m = plot_stations(df_stations_metrobus, m, 'METROBUS')
            level1_map_data = st_folium(m)
            st.session_state.selected_id = level1_map_data['last_object_clicked_tooltip']
        # Second column validation.
        if 'selected_id' not in st.session_state:
            col2.subheader("ES NECESARIO SELECCIONAR UNA ESTACIÓN")
            col2.subheader("AL SELECCIONAR APARECERÁ LA SIGUIENTE INFORMACIÓN:")
            col2.write("##### TOP DELITOS")
            col2.write("##### COMPARACIÓN DE GÉNEROS")
            col2.write("##### EDAD")
            col2.write("##### DISTANCIAS DE LOS DELITOS")
        else:
            if st.session_state.selected_id is not None:
                cve_est = st.session_state.selected_id.split(":")[0]
                name_est = get_station(df_stations_metrobus, cve_est, "nombre")
                tipo_est = get_station(df_stations_metrobus, cve_est, "tipo")
                linea_est = get_station(df_stations_metrobus, cve_est, "linea")
                col2.subheader(f'{name_est.upper()}')
                col2.write(f'LÍNEA: {linea_est[1:]}')
                col2.write(f'ESTACIÓN TIPO: {tipo_est.upper()}')
                col2.write("##### TOP DELITOS")
                data = {
                    'Tipo de Delito': ['Robo', 'Asalto', 'Homicidio'],
                    'Frecuencia': [20, 15, 5]  # Ejemplo de frecuencia de delitos
                }
                df = pd.DataFrame(data)
                plt.barh(df['Tipo de Delito'], df['Frecuencia'], color='skyblue', edgecolor='black')
                plt.xlabel('Tipo de Delitos')
                plt.ylabel('Frecuencia de Delitos')
                plt.title('Histograma de Frecuencia de Delitos')
                plt.grid(True)
                col2.pyplot(plt)
                col2.write("##### COMPARACIÓN DE GÉNEROS")
                # Datos de ejemplo
                labels = ['HOMBRE', 'MUJER']
                sizes = [45, 55]
                # Configuración de colores
                colors = sns.color_palette('pastel')
                # Crear gráfico de pastel
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
                ax.axis('equal')  # Equal aspect ratio para asegurar que es un círculo
                ax.set_title('Ejemplo de Gráfico de Pastel con Seaborn', fontsize=16)
                col2.pyplot(fig)
                col2.write("###### EDAD")
                data2 = {
                    'Edad': ['[10-20]', '[40-50]', '[90-100]'],
                    'Frecuencia': [20, 15, 5]  # Ejemplo de frecuencia de delitos
                }
                df2 = pd.DataFrame(data2)
                fig2, ax2 = plt.subplots()
                ax2.bar(df2['Edad'], df2['Frecuencia'], color='skyblue', edgecolor='black')
                col2.pyplot(fig2)
                col2.divider()
                col2.write("###### DISTANCIAS DE LOS DELITOS")
                # Generar datos aleatorios para el histograma
                np.random.seed(0)
                data = np.random.randn(1000)
                # Definir el número de bins para el histograma
                plt.figure(figsize=(8, 6))
                plt.hist(data, bins=100, orientation='horizontal', color='skyblue')
                plt.xlabel('Frecuencia')
                plt.ylabel('xlabel')
                plt.title('Histograma Horizontal')
                plt.grid(True)
                col2.pyplot(plt)
            else:
                col2.subheader("ES NECESARIO SELECCIONAR UNA ESTACIÓN")
                col2.subheader("AL SELECCIONAR APARECERÁ LA SIGUIENTE INFORMACIÓN:")
                col2.write("##### TOP DELITOS")
                col2.write("##### COMPARACIÓN DE GÉNEROS")
                col2.write("##### EDAD")
                col2.write("##### DISTANCIAS DE LOS DELITOS")


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
    #st.markdown("<br>", unsafe_allow_html=True)
    
    # First container.
    with st.container():
        #st.markdown('<b>Parámetros para la predicción</b>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 3, 1])
        
        list_categ_crimes = {
            'Robo simple': 'Robo simple',
            'Robo a pasajero en transporte público colectivo o individual': 'Robo en transporte público individual o colectivo',
            'Robo a transeúnte': 'Robo a transeúnte',
            'Delitos sexuales (acoso, abuso, violación y otros)': 'Acoso, abuso, violación y otros delitos sexuales',
            'Robo de vehículo, autopartes y en transporte individual': 'Robo de vehículo, autopartes y en transporte individual',
            'Robo a negocio': 'Robo a negocio',
            'Lesiones': 'Lesiones',
            'Homicidio': 'Homicidio',
            'Fraude': 'Fraude',
            'Amenazas': 'Amenazas',
        }
        
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
        # level_div = col2.selectbox("Nivel de división", [
        #     "Alcaldía",
        #     #"Sector policial"
        # ])
        
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
        
        
        
        # Validar a que agrupamiento corresponde determinado modelo predictivo
        grouped_dataset_id = 0
        columns_input_model = []
        regions = []
        if level_div == 'Alcaldía' and sex == 'Ambos':
            print('Agrupamiento 3')
            grouped_dataset_id = 3
            #columns_input_model = ['alcaldia', 'categoria_delito_adaptada', 'semana_1',]
            columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'semana_1',]
            columns_weekly_count = ['semana_anio_completa', 'semana_anio', 'anio', 'alcaldia', 'categoria_delito_adaptada', 'conteo']
            if transport == 'STC Metro':
                regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tláhuac', 'Venustiano Carranza', 'Álvaro Obregón']
            else:
                regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tlalpan', 'Venustiano Carranza', 'Álvaro Obregón']
                
            inputs_model_ls = []
            for region in regions:
                inputs_model_ls.append([week_month_aux, region, categ_crime_ok])
            input_model_df_partial = pd.DataFrame(inputs_model_ls, columns=columns_input_model[:-1])
            
            weekly_crime_counts = load_weekly_crime_counts(transport, grouped_dataset_id)[columns_weekly_count]
            weekly_crime_counts = weekly_crime_counts.groupby(columns_weekly_count[:-1])['conteo'].sum().reset_index(name='conteo')
            print(weekly_crime_counts)
            weekly_crime_counts_filtered = weekly_crime_counts[(weekly_crime_counts['categoria_delito_adaptada'] == categ_crime_ok)].sort_values(by=['anio', 'semana_anio'])
            print('\n')
            print(weekly_crime_counts_filtered)
            thresholds_pred = load_thresholds_crime_model(transport, grouped_dataset_id)
            
            print('Thresholds')
            print(thresholds_pred)
            threshold_grouped_dataset = thresholds_pred[(thresholds_pred['categ_delito'] == categ_crime_ok)]['percentil'].to_list()[0]
            print(threshold_grouped_dataset)
            
        
        elif level_div == 'Alcaldía' and sex != 'Ambos':
            print('Agrupamiento 4')
            grouped_dataset_id = 4
            #columns_input_model = ['alcaldia', 'categoria_delito_adaptada', 'sexo_victima', 'semana_1',]
            columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'sexo_victima', 'semana_1',]
            columns_weekly_count = ['semana_anio_completa', 'semana_anio', 'anio', 'alcaldia', 'categoria_delito_adaptada', 'id_sexo', 'conteo']
            if transport == 'STC Metro':
                regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tláhuac', 'Venustiano Carranza', 'Álvaro Obregón']
            else:
                regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tlalpan', 'Venustiano Carranza', 'Álvaro Obregón']
                
            inputs_model_ls = []
            for region in regions:
                inputs_model_ls.append([week_month_aux, region, categ_crime_ok, id_sex, ])
            input_model_df_partial = pd.DataFrame(inputs_model_ls, columns=columns_input_model[:-1])
            
            weekly_crime_counts = load_weekly_crime_counts(transport, grouped_dataset_id)[columns_weekly_count]
            weekly_crime_counts = weekly_crime_counts.groupby(columns_weekly_count[:-1])['conteo'].sum().reset_index(name='conteo')
            print(weekly_crime_counts)
            weekly_crime_counts_filtered = weekly_crime_counts[(weekly_crime_counts['categoria_delito_adaptada'] == categ_crime_ok)
                                                               & (weekly_crime_counts['id_sexo'] == id_sex)].sort_values(by=['anio', 'semana_anio'])
            
            thresholds_pred = load_thresholds_crime_model(transport, grouped_dataset_id)
            print('Thresholds')
            print(thresholds_pred)
            threshold_grouped_dataset = thresholds_pred[(thresholds_pred['categ_delito'] == categ_crime_ok) & (thresholds_pred['sexo'] == id_sex)]['percentil'].to_list()[0]
            print(threshold_grouped_dataset)
            
        if level_div == 'Sector policial' and sex == 'Ambos':
            print('Agrupamiento 6')
            grouped_dataset_id = 6
            #columns_input_model = ['sector', 'categoria_delito_adaptada', 'semana_1',]
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
            #columns_input_model = ['sector', 'categoria_delito_adaptada', 'sexo_victima', 'semana_1',]
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
        
        # Carga de modelo respectivo y lectura de afluencia predicha para este año
        crime_model = load_crime_model(transport, grouped_dataset_id)
        afflu_fc_values = load_afflu_forecast_values(transport, grouped_dataset_id)
        afflu_fc_values_filtered = afflu_fc_values[afflu_fc_values['semana_anio'] == weeks_forward + int(week_year) - 1]
        input_model_df = input_model_df_partial.merge(afflu_fc_values_filtered, left_on=columns_input_model[1], right_on=['region'])
        input_model_df.rename(columns={'afluencia': 'semana_1'}, inplace=True)
        input_model_df = input_model_df[columns_input_model]
        #print(crime_model)
        print(input_model_df)
        preds = crime_model.predict(input_model_df)
        #print(input_model_df)
        print('Predicciones para semana {}:'.format(weeks_forward + int(week_year)))
        df_preds = pd.DataFrame(columns=['valor'])
        df_preds['valor'] = preds
        df_preds['valor'] = df_preds['valor'].replace({'High': 'Riesgo elevado', 'Low': 'Riesgo moderado'})
        print(df_preds)
        #print(munics_gdf)
        #print(police_sectors_gdf)
        input_model_df_preds = pd.concat([input_model_df, df_preds], axis=1)
        
        # Obtener la granularidad espacial para luego pasarla como argumento al gráfico de mapa
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
     
    bool_plot_ts_weekly_crime = False
    with st.container():
        metric_unique_values = np.sort(region_gdf_merge['valor'].unique())
        col7, col78, col8 = st.columns([20, 1, 40])
        with col7:
            fig = plot_predictive_map(region_gdf_merge, transport, metric_unique_values)
            fig.update_layout({"uirevision": "foo"}, overwrite=True)
            # https://discuss.streamlit.io/t/interactive-plot-get-which-point-a-user-clicked/14596
            # https://github.com/null-jones/streamlit-plotly-events
            # https://orosz-attila-covid-19-dashboard-streamlit-app-kmmvgj.streamlit.app/
            #https://towardsdatascience.com/highlighting-click-data-on-plotly-choropleth-map-377e721c5893
            selected_click_aux = plotly_events(fig, click_event=True)
            st.session_state.selected_click_pred_map.append(selected_click_aux)
            print('click', st.session_state.selected_click_pred_map)
           
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
                    
                    #col8.markdown(f'<span style="color: black; font-size: 14px;">para las estaciones de <b>{region_name}</b> durante la <b>{text_num_week_forward}</b></span> <span style="color: #a5a5a5; font-size: 14px;">({get_week_date_range(weeks_forward + int(week_year), year)})</span>', unsafe_allow_html=True)
                    
                    print(weekly_crime_counts_filtered)
                    weekly_crime_counts_filtered = weekly_crime_counts_filtered[weekly_crime_counts_filtered[region_column_name_2] == region_name]
                    print(weekly_crime_counts_filtered)
                    #fig3 = plot_ts_weekly_crime(weekly_crime_counts_filtered, f'Conteo semanal de {categ_crime_ok.lower()} (2019-2023)', threshold_grouped_dataset)
                    fig3 = plot_ts_weekly_crime(weekly_crime_counts_filtered, f'Conteo semanal delictivo (2019-2023)', threshold_grouped_dataset)
                    
                    col9, col10, col11 = col8.columns([40, 1, 80])
                    with col9:
                        fig2 = plot_complementary_predictive_map(region_gdf_merge, transport, region_gdf_aux, region_column)
                        fig2.update_layout({"uirevision": "foo"}, overwrite=True)
                        
                        col9.plotly_chart(fig2, use_container_width=True)
                    with col11:
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    with st.expander("Estaciones afectadas", expanded=True):
                        #st.markdown('<span style="color: black; font-size: 14px;">Estaciones afectadas:</span>', unsafe_allow_html=True)
                        if region_column == 'CVE_MUN':
                            df_stations_metro_aux = df_stations_metro[(df_stations_metro['alcaldia'] == region_name)]
                        else:
                            df_stations_metro_aux = df_stations_metro[(df_stations_metro['sector'] == region_name)]
                        
                        # Estaciones dentro de la región
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
                        #st.write('<br>', unsafe_allow_html=True)
                    # weekly_crime_counts_filtered = weekly_crime_counts_filtered[weekly_crime_counts_filtered[region_column_name_2] == region_name]
                    # fig3 = plot_ts_weekly_crime(weekly_crime_counts_filtered, f'Conteo semanal de {categ_crime_ok.lower()} (2019-2023)', threshold_grouped_dataset)
                    
                    # col8.plotly_chart(fig3, use_container_width=True)
                            
                    
        
    with st.container():
        
        pass


# SIDEBAR
# LINKS AUXILIARES CSS EN STREAMLIT
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
    if st.button("STC Metro"):
        st.session_state.selection = "METRO"
        st.session_state.transport = 'STC Metro'
    if st.button("Metrobús"):
        st.session_state.selection = "METROBÚS"
    if st.button("Predicciones"):
        st.session_state.selection = "PREDICCIONES"
        st.session_state.selected_click_pred_map = []

# Execute action depending on the above selection
if "selection" not in st.session_state:
    home()
else:
    if st.session_state.selection == "INICIO":
        home()
    if st.session_state.selection == "METRO":
        metro()
    if st.session_state.selection == "METROBÚS":
        metrobus()
    if st.session_state.selection == "PREDICCIONES":
        #if len(st.session_state.selected_click_pred_map) > 0:
        #    pass
        #else:
        #    st.session_state.selected_click_pred_map = []
        predictions()
        
        
