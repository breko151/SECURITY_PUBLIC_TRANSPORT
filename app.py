# Libraries needed.
# Web.
import streamlit as st
from streamlit_folium import st_folium
# Data.
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import LineString
import folium
# Data visualization.
import seaborn as sns
import matplotlib.pyplot as plt


# Global variables.
df_stations = pd.read_csv("./fact_constellation_schema/coordanadas_estaciones.csv")
metro_lines = geopandas.read_file('./shapefiles/metro/STC_Metro_lineas_utm14n_repr.shp', index=False)
mb_lines = geopandas.read_file('./shapefiles/mb/Metrobus_lineas_utm14n_repr.shp', index=False)
mb_lines = mb_lines[mb_lines['LINEA'] != '01 y 02']
lineas_cdmx = geopandas.read_file('./images/cdmx.json', encoding='utf-8')
lineas_cdmx_json = lineas_cdmx.to_json()

df_stations_metro = df_stations[df_stations['sistema'] == 'STC Metro']
df_stations_metrobus = df_stations[df_stations['sistema'] == 'Metrobús']


# Map initialization.
def init_map(center=(19.4325019109759, -99.1322510732777), zoom_start=10, map_type="cartodbpositron"):
    return folium.Map(location=center, zoom_start=zoom_start, tiles=map_type)


# DataImages for stations.
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


# DataColors for stations.
LINESM = {
    '1': '#e55f91',
    '2': '#0071d0',
    '3': '#c1b405',
    '4': '#8fc0ba',
    '5': '#f0d405',
    '6': '#a81b1e',
    '7': '#e48310',
    '8': '#008d4d',
    '9': '#522400',
    'A': '#652782',
    'B': '#d7d7d7',
    '12': '#b89038',
}


LINESMB = {
    '01': '#a9343a',
    '01 y 02': '#FFFFFF',
    '02': '#862b92',
    '03': '#799b3d',
    '04': '#ff9400',
    '05': '#05367d',
    '06': '#e2188e',
    '07': '#01642d'
}


# Plot map.
def plot_from_df(df, folium_map, type):
    # Validation of type of transport.
    if type == 'METRO':
        # Add every station.
        for i, row in df.iterrows():
            # Customize icon.
            icon = folium.CustomIcon(
                POINTSM[row.linea],
                icon_size=(15, 15)
            )
            folium.Marker([row.latitud, row.longitud],
                        icon=icon,
                        tooltip=f'{row.cve_est}: {row.nombre}, Línea: {row.linea[1:]}').add_to(folium_map)
            # 
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
            folium.Marker([row.latitud, row.longitud],
                        icon=icon,
                        tooltip=f'{row.cve_est}: {row.nombre}, Línea: {row.linea[1:]}').add_to(folium_map)
            # 
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


# Get columns from cve_est.
def get_station(df, cve_est, column=None):
    if column is None:
        result = df[df['cve_est'] == cve_est]
    else:
        filter = df[df['cve_est'] == cve_est]
        print(filter)
        result = filter[column].to_list()[0]
    return result


# Page config.
st.set_page_config(page_title="Página Principal",
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


# Different Views.
# home view.
def home():
    # Style config.
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #2B2A25;
        }
    </style>
    """, unsafe_allow_html=True)
    st.title("BIENVENIDO A TU TRANSPORTE SEGURO")
    st.image("./images/MapaCDMX.png")
    

# metro view.
def metro():
    st.title("METRO")
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
            m = plot_from_df(df_stations_metro, m, 'METRO')
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
                col2.subheader("ES NECESARIO SELECCIONAR UNA ESTACIÓN")
                col2.subheader("AL SELECCIONAR APARECERÁ LA SIGUIENTE INFORMACIÓN:")
                col2.write("##### TOP DELITOS")
                col2.write("##### COMPARACIÓN DE GÉNEROS")
                col2.write("##### EDAD")
                col2.write("##### DISTANCIAS DE LOS DELITOS")
                

def metrobus():
    st.title("METROBÚS")
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
            m = plot_from_df(df_stations_metrobus, m, 'METROBUS')
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


def predictions():
    st.title("PREDICCIONES")


# Sidebar elements.
with st.sidebar:
    # Sidebar title.
    st.title("ESCOGE UNA OPCIÓN")
    # Sidebar buttons.
    if st.button("INICIO"):
        st.session_state.selection = "INICIO"
    if st.button("METRO"):
        st.session_state.selection = "METRO"
    if st.button("METROBÚS"):
        st.session_state.selection = "METROBÚS"
    if st.button("PREDICCIONES"):
        st.session_state.selection = "PREDICCIONES"

# Options.
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
        predictions()