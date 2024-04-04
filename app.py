# Libraries needed.
import streamlit as st

# Page config.
st.set_page_config(page_title="Página Principal")
selection = "INICIO"

# Different Views.
def home():
    st.title("BIENVENIDO A TU ESPACIO SEGURO")
    st.image("./images/MapaCDMX.png")
    

def metro():
    st.title("Metro")


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

# st.markdown("""
# <style>
#     [data-testid=stSidebar] {
#         background-color: #ff000050;
#     }
# </style>
# """, unsafe_allow_html=True)


# streamlit_style = """
# 			<style>
# 			@import url('https://fonts.googleapis.com/css2?family=Metrophobic&display=swap');

# 			html, body, [class*="css"]  {
# 			font-family: 'Metrophobic', sans-serif;
# 			}
# 			</style>
# 			"""
# st.markdown(streamlit_style, unsafe_allow_html=True)

# # # Pages of the app (sidebar)
# # show_pages(
# #             [
# #                 Page("app.py", "Inicio"),
# #                 Page("sections/sociodemografico.py", "Sociodemográfico"),
# #                 Page("sections/socioelectoral.py", "Socioelectoral"),
# #                 Page("sections/upload_files.py", "Archivos"),
# #                 Page("sections/register.py", "Registro"),
# #                 Page("sections/forms.py", "Formulario"),
# #                 Page("sections/vis_forms.py", "VISUALIZACIÓN")
# #             ]
# #         )

# # Main contain.
# st.title("BIENVENIDO A TU ESPACIO SEGURO")
# st.image("./images/MapaCDMX.png")