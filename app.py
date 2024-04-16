# Libraries needed.

# Web.
import streamlit as st
from streamlit_folium import st_folium
import base64

# Data.
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import LineString
import folium

# Data visualization.
import seaborn as sns
import matplotlib.pyplot as plt

# Datetime
from datetime import datetime, timedelta

# Constants
from colors import LINESM, LINESMB

# Auxiliar proper modules
from plots import plot_top_stations_affluence_trends, plot_top_stations_crime_trends

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

# Global variables.
df_stations = pd.read_csv("./fact_constellation_schema/coordanadas_estaciones.csv")
metro_lines = geopandas.read_file('./shapefiles/metro/STC_Metro_lineas_utm14n_repr.shp', index=False)
mb_lines = geopandas.read_file('./shapefiles/mb/Metrobus_lineas_utm14n_repr.shp', index=False)
mb_lines = mb_lines[mb_lines['LINEA'] != '01 y 02']
lineas_cdmx = geopandas.read_file('./images/cdmx.json', encoding='utf-8')
lineas_cdmx_json = lineas_cdmx.to_json()

df_stations_metro = df_stations[df_stations['sistema'] == 'STC Metro']
df_stations_metrobus = df_stations[df_stations['sistema'] == 'Metrobús']

dict_weekday = {
    0: 'Lunes',
    1: 'Martes',
    2: 'Miércoles',
    3: 'Jueves',
    4: 'Viernes',
    5: 'Sábado',
    6: 'Domingo',
}

today_ = datetime.now()
weekday = dict_weekday[today_.weekday()]
week_year = today_.strftime("%W")
month = today_.month


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
                        tooltip=f'{row.cve_est}: {row.nombre} Línea: {row.linea[1:]}').add_to(folium_map)
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
                        tooltip=f'{row.cve_est}: {row.nombre} Línea {row.linea[1:]}').add_to(folium_map)
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

# Load of images to display at dashboard without st.image()
image_home_logo_url = "./images/MapaCDMX.png"
file_image_home = open(image_home_logo_url, "rb")
contents = file_image_home.read()
data_url_image_home = base64.b64encode(contents).decode("utf-8")
file_image_home.close()

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

#Aux center CSS for columns
#CSS style to center horizontally and vertically
center_css = """
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: row;
"""

# Different Views.
# home view.
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
    
    

# metro view.
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
    
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("<div style='{}'><img src='data:image/gif;base64,{}' width='90' style='margin-right:0px;'></div>".format(center_css, data_url_metro_logo), unsafe_allow_html=True)
    with col2:
        st.title('STC Metro')
        #st.markdown("<div style='display: flex; justify-content: left; align-items: center;'><h1 style='text-align: left;'>STC Metro</h1></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # First container.
    with st.container():
        st.header("🔥 Tendencias")
        st.write("🗓️ Datos históricos para <b>{}</b> de la <b>semana {}</b> de años pasados.".format(weekday.lower(), week_year), unsafe_allow_html=True)
        opt = st.selectbox("Escoge una zona", ["Ciudad de México", "Centro", "Norte", "Sur", "Oriente", "Poniente"])
        st.write(opt)
        col1, col2 = st.columns(2)
        # Column 1.
        
        col1.write("##### Top 10 estaciones con más afluencia")
        #data = {
        #    'Estación': ['Hidalgo', 'Juárez', 'Centro Médico'],
        #    'Frecuencia': [20, 15, 5]  # Ejemplo de frecuencia de delitos
        #}
        #df = pd.DataFrame(data)
        #plt.barh(df['Estación'], df['Frecuencia'], color='skyblue', edgecolor='black')
        #plt.xlabel('Estación')
        #plt.ylabel('Frecuencia de Afluencias')
        #plt.title('Histograma de Frecuencia de Afluencias')
        #plt.grid(True)
        #col1.pyplot(plt)
        
        data_df = {'nombre': ['Politécnico', 'Constitución de 1917', 'Indios Verdes', 'Pantitlán'],
                'línea': ['L5', 'L8', 'L3', 'L1'],
                'afluencia_promedio': [10.2, 9.3, 8.2, 7.3],}
        df_top_stations_affluence_trends = pd.DataFrame(data_df).sort_values(by=['afluencia_promedio'], ascending=True)
        
        col1.plotly_chart(plot_top_stations_affluence_trends(df_top_stations_affluence_trends, 'STC Metro'), use_container_width=True)
        
        # Column 2.
        col2.write("##### Top 10 estaciones más delictivas")
        #data = {
        #    'Estación': ['Hidalgo', 'Juárez', 'Centro Médico'],
        #    'Frecuencia': [20, 15, 5]  # Ejemplo de frecuencia de delitos
        #}
        #df = pd.DataFrame(data)
        #plt.barh(df['Estación'], df['Frecuencia'], color='skyblue', edgecolor='black')
        #plt.xlabel('Estación')
        #plt.ylabel('Frecuencia de Afluencias')
        #plt.title('Histograma de Frecuencia de Délitos')
        #plt.grid(True)
        #col2.pyplot(plt)
        
        data_df = {'nombre': ['Politécnico', 'Constitución de 1917', 'Indios Verdes', 'Pantitlán'],
                'línea': ['L5', 'L8', 'L3', 'L1'],
                'conteo_delitos': [5, 3, 6, 8],}
        df_top_stations_crime_trends = pd.DataFrame(data_df).sort_values(by=['conteo_delitos'], ascending=True)
        
        col2.plotly_chart(plot_top_stations_crime_trends(df_top_stations_crime_trends, 'STC Metro'), use_container_width=True)
        
    # Second container.
    with st.container():
        st.header("🔍 Explora las estaciones")
        col1, col2 = st.columns(2)
        # Map column.
        with col1:
            m = init_map()
            m = plot_from_df(df_stations_metro, m, 'METRO')
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
    
    st.title("Predicciones")


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

# Script JavaScript para manejar el estado activo de los botones
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

# Sidebar elements.
with st.sidebar:
    # Sidebar title.
    st.title("Menú")
    # Sidebar buttons.
    if st.button("Inicio"):
        st.session_state.selection = "INICIO"
    if st.button("STC Metro"):
        st.session_state.selection = "METRO"
    if st.button("Metrobús"):
        st.session_state.selection = "METROBÚS"
    if st.button("Predicciones"):
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
        
        
# LINKS AUXILIARES CSS EN STREAMLIT
# https://discuss.streamlit.io/t/button-css-for-streamlit/45888/3
# https://discuss.streamlit.io/t/button-size-in-sidebar/36132/2
# https://discuss.streamlit.io/t/image-and-text-next-to-each-other/7627/7