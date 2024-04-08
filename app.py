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
def plot_from_df(df, folium_map, DICT_COLORS):
    for i, row in df.iterrows():
        icon = folium.CustomIcon(
            DICT_COLORS[row.linea],
            icon_size=(15, 15)
        )
        folium.Marker([row.latitud, row.longitud],
                      icon=icon,
                    tooltip=f'{row.cve_est}: {row.nombre}, Línea: {row.linea[1:]}').add_to(folium_map)
    return folium_map


# Get columns from cve_est.
def get_station(df, cve_est, column=None):
    if column is None:
        result = df[df['cve_est'] == cve_est]
    else:
        filter = df_stations_metro[df_stations_metro['cve_est'] == cve_est]
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
            m = plot_from_df(df_stations_metro, m, POINTSM)
            level1_map_data = st_folium(m)
            st.session_state.selected_id = level1_map_data['last_object_clicked_tooltip']
        if 'selected_id' not in st.session_state:
            col2.title("ES NECESARIO SELECCIONAR UNA ESTACIÓN")
            col2.subheader("AL SELECCIONAR APARECERÁ LA SIGUIENTE INFORMACIÓN:")
            col2.write("TOP DELITOS")
            col2.write("COMPARACIÓN DE GÉNEROS")
            col2.write("EDAD")
            col2.write("DISTANCIAS DE LOS DELITOS")
        else:
            if st.session_state.selected_id is not None:
                cve_est = st.session_state.selected_id.split(":")[0]
                name_est = get_station(df_stations_metro, cve_est, "nombre")
                col2.title(f'{name_est}')
                col2.write("TOP DELITOS")
                col2.write("COMPARACIÓN DE GÉNEROS")
                col2.write("EDAD")
                col2.write("DISTANCIAS DE LOS DELITOS")
            else:
                col2.title("ES NECESARIO SELECCIONAR UNA ESTACIÓN")
                col2.subheader("AL SELECCIONAR APARECERÁ LA SIGUIENTE INFORMACIÓN:")
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