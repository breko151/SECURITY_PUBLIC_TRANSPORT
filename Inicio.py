import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from dotenv import load_dotenv
import os

# Credenciales
load_dotenv()
PASSWORD = os.getenv('PASSWORD')

# GENERAL SETTINGS OF DASHBOARD
with streamlit_analytics.track(unsafe_password=f'{PASSWORD}'):
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
        st.markdown('<br>', unsafe_allow_html=True,)
        col_1, col_mid, col_2 = st.columns([0.30, 0.05, 0.65])
        with col_1:
            st.image(r'./assets/images/MapaCDMX.png',  use_container_width=True, output_format='PNG')
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
                _extracted_from_home_36(
                    r'El metro cuenta con:',
                    """
                            - 12 líneas.
                            - 195 estaciones.
                            - 269.52 km.""",
                    """
                            - Lunes a viernes: 5:00-0:00 horas.
                            - Sábados: 6:00-0:00 horas.
                            - Domingos y días festivos: 7:00-0:00 horas.
                            """,
                )
            with col_2:
                st.image(r'./assets/images/MAPA_METRO.png', use_container_width=True, output_format='PNG')
        elif level_div == "Metrobús":
            col_1, col_mid, col_2 = st.columns([0.45, 0.05, 0.45])
            with col_1:
                st.write(r'<div style="text-align: justify;">El Metrobús es un sistema de autobuses con infraestructura dedicada, carriles exclusivos y sistemas de control, que se inauguró el 19 de junio de 2005. Según el INEGI, en 2022 el metrobús prestó servicio a 33.8 millones de personas de manera mensual. Se considera que es el tipo de transporte que le sigue en importancia al STC Metro.</div><br>', unsafe_allow_html=True,)
                _extracted_from_home_36(
                    r'El metrobús cuenta con:',
                    """
                            - 7 líneas.
                            - 283 estaciones.
                            - 125 km.""",
                    """
                            - Lunes a sábado: 4:30-0:00 horas.
                            - Domingos y días festivos: 5:00-0:00 horas.
                            """,
                )
            with col_2:
                st.image(r'./assets/images/MAPA_METROBUS.png', use_container_width=True, output_format='PNG')


    # TODO Rename this here and in `home`
    def _extracted_from_home_36(arg0, arg1, arg2):
        st.write(arg0)
        st.markdown(arg1)
        st.write(r'Horarios de operación: ')
        st.markdown(arg2)

    if __name__ == "__main__":
        home()


