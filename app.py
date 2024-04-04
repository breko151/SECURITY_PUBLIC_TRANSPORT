# Libraries needed.
import streamlit as st

# Page config.
st.set_page_config(page_title="Página Principal")
selection = "INICIO"

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
    st.title("BIENVENIDO A TU ESPACIO SEGURO")
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
        col1.image("./images/MapaCDMX.png")
        col2.title("NOMBRE ESTACIÓN")
        col2.write("TOP DELITOS")
        col2.write("COMPARACIÓN DE GÉNEROS")
        col2.write("EDAD")
        col2.write("DISTANCIAS DE LOS DELITOS")


def metrobus():
    st.title("Metrobus")


def predictions():
    st.title("Predicciones")


# Sidebar elements.
with st.sidebar:
    # Sidebar title.
    st.title("ESCOGE UNA OPCIÓN")
    # Sidebar buttons.
    if st.button("INICIO"):
        selection = "INICIO"
    if st.button("METRO"):
        selection = "METRO"
    if st.button("METROBÚS"):
        selection = "METROBÚS"
    if st.button("PREDICCIONES"):
        selection = "PREDICCIONES"

# Options.
if selection == "INICIO":
    home()
if selection == "METRO":
    metro()
if selection == "METROBÚS":
    metrobus()
if selection == "PREDICCIONES":
    predictions()