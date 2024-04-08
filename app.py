# Libraries needed.
import streamlit as st
import pandas as pd
import geopandas
import folium
from shapely.geometry import Point
from streamlit_folium import st_folium


# Map initialization.
def init_map(center=(19.4325019109759, -99.1322510732777), zoom_start=10, map_type="cartodbpositron"):
    return folium.Map(location=center, zoom_start=zoom_start, tiles=map_type)


# Plot map.
def plot_from_df(df, folium_map):
    for i, row in df.iterrows():
        folium.Marker([row.latitud, row.longitud],
                      tooltip=f'{row.cve_est}: {row.linea}, {row.nombre}').add_to(folium_map)
    return folium_map


# Page config.
st.set_page_config(page_title="Página Principal",
    initial_sidebar_state="expanded",
    layout="wide",
    page_icon="🚈")
# Hide the legend of "Made with streamlit" and hamburger menu
# https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/2
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
df_stations = pd.read_csv("./fact_constellation_schema/coordanadas_estaciones.csv")
df_stations_metro = df_stations[df_stations['sistema'] == 'STC Metro']
df_stations_metrobus = df_stations[df_stations['sistema'] == 'Metrobús']


# Different Views.
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
    

def metro():
    st.title("METRO")
    # First container.
    with st.container():
        st.title("TENDENCIAS")
        st.selectbox("ESCOGE UNA ZONA", ["CIUDAD DE MÉXICO", "NORTE", "SUR", "ORIENTE", "PONIENTE"])
        col1, col2 = st.columns(2)
        col1.write("TOP AFLUENCIAS")
        col2.write("TOP DELECTIVAS")
    with st.container():
        st.title("ESTACIONES")
        col1, col2 = st.columns(2)
        with col1:
            m = init_map()
            m = plot_from_df(df_stations_metro, m)
            level1_map_data = st_folium(m)
            st.session_state.selected_id = level1_map_data['last_object_clicked_tooltip']
        if 'selected_id' not in st.session_state:
            col2.title("NOMBRE ESTACIÓN")
            col2.write("TOP DELITOS")
            col2.write("COMPARACIÓN DE GÉNEROS")
            col2.write("EDAD")
            col2.write("DISTANCIAS DE LOS DELITOS")
        else:
            if st.session_state.selected_id is not None:
                col2.title(f'{st.session_state.selected_id}')
                col2.write("TOP DELITOS")
                col2.write("COMPARACIÓN DE GÉNEROS")
                col2.write("EDAD")
                col2.write("DISTANCIAS DE LOS DELITOS")
            else:
                col2.title("NOMBRE ESTACIÓN")
                col2.write("TOP DELITOS")
                col2.write("COMPARACIÓN DE GÉNEROS")
                col2.write("EDAD")
                col2.write("DISTANCIAS DE LOS DELITOS")
                


def metrobus():
    st.title("METROBÚS")
    # First container.
    with st.container():
        st.title("TENDENCIAS")
        st.selectbox("ESCOGE UNA ZONA", ["CIUDAD DE MÉXICO", "NORTE", "SUR", "ORIENTE", "PONIENTE"])
        col1, col2 = st.columns(2)
        col1.write("TOP AFLUENCIAS")
        col2.write("TOP DELECTIVAS")
    with st.container():
        st.title("ESTACIONES")
        col1, col2 = st.columns(2)
        col1.image("./images/MapaCDMX.png")
        col2.title("NOMBRE ESTACIÓN")
        col2.write("TOP DELITOS")
        col2.write("COMPARACIÓN DE GÉNEROS")
        col2.write("EDAD")
        col2.write("DISTANCIAS DE LOS DELITOS")


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