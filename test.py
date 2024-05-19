# Web.
import streamlit as st
import base64

image_home_logo_url = "./images/MapaCDMX.png"
file_image_home = open(image_home_logo_url, "rb")
contents = file_image_home.read()
center_css = """
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: row;
"""
data_url_image_home = base64.b64encode(contents).decode("utf-8")
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
    col_1, col_2 = st.columns([0.40, 0.60])
    with col_1:
        st.image(r'./images/MapaCDMX.png')
        # st.markdown(
        #     r'<div style="{}"><img src="data:image/gif;base64,{}" alt="Imagen home" width=200 ></div>'.format(center_css, data_url_image_home),
        #     unsafe_allow_html=True,
        # )
    with col_2:
        st.subheader("La delincuencia en el transporte público de la Ciudad de México")
        st.write(r'El transporte público es un elemento esencial en la vida cotidiana de las personas. En particular, para la Ciudad de México el STC Metro y Metrobús son los medios de transporte más utilizados, por lo que, es importante garantizar la seguridad y satisfacción de los usuarios. Sin embargo, debido al crecimiento en la red de transporte público, se ha generado una alta concentración de personas en las instalaciones de ambos medios de transporte, lo que ha propiciado un aumento en la incidencia delictiva.')
        st.write(r'Lo anterior requiere de un proceso de conocer la dinámica de los delitos por medio de un proceso de integración de datos públicos donde se considera:')
        st.markdown("""
                    - Carpetas de investigación de la Fiscalía General de Justicia de la Ciudad de México.
                    - Datos geoespaciales de las estaciones de Metro y Metrobús
                    - Datos de afluencia de las estaciones de Metro y Metrobús
                    """)
        st.write(r'Con la finalidad de conocer el fenómeno de los delitos que ocurrieron dentro y en las cercanías de las estaciones de ambos medios de transporte.')
    level_div = st.selectbox("Tipo de transporte", ["Metro", "Metrobús"])
    if level_div == "Metro":
        col_1, col_2 = st.columns([0.50, 0.50])
        with col_1:
            st.write(r'El Sistema de Transporte Colectivo Metro es una red de transporte público subterráneo que se encuentra en la Ciudad de México y parte de su área metropolitana. Según la INEGI, 90.2 millones de personas usaban mensualmente este transporte en 2022, por lo que lo vuelve en el transporte público más utilizado en la ciudad y su área metropolitana.')
            st.write(r'El Metro cuenta:')
            st.markdown("""
                        - 12 líneas.
                        - 195 estaciones.
                        - 269.52 km.""")
            st.write(r'Horarios de operación: ')
            st.markdown("""
                        - Lunes a viernes: 5:00-00:00 horas.
                        - Sábados: 6:00-0:00 horas.
                        - Domingos y días festivos: 7:00-0:00 horas.
                        """)
        with col_2:
            st.image(r'./images/MAPA_METRO.png')
    elif level_div == "Metrobús":
        col_1, col_2 = st.columns([0.50, 0.50])
        with col_1:
            st.write(r'El metrobús es un sistema de autobús, infraestructura dedica, carriles exclusivo y sistemas de control que se inauguró el 19 de junio de 2005. Según la INEGI el metrobus presto servicio a 22.2 millones de personas de manera mensual en el año 2021. Se considera que es el tipo de transporte que le sigue en importancia al Metro.')
            st.write(r'El metrobús cuenta:')
            st.markdown("""
                        - 7 líneas.
                        - 283 estaciones.
                        - 125 km.""")
            st.write(r'Horarios de operación: ')
            st.markdown("""
                        - Lunes a sabado: 4:30-00:00 horas.
                        - Domigos y días festivos: 5:00-0:00 horas.
                        """)
        with col_2:
            st.image(r'./images/MAPA_METROBUS.png')
with st.sidebar: 
    st.title("Menú")
home()