import streamlit as st
from src.config import zones_ls, munics_ls, lines_ls, get_current_date_info
from src.loader import load_geodata
from src.querys import query_top_stations_affluence_trends, query_top_stations_crime_trends
from src.plots import plot_crime_trend_stations, plot_top_stations_crime_trends, plot_top_stations_affluence_trends

st.set_page_config(page_title="Tendencias", page_icon="🔥", layout="wide")

# Load data
geodata = load_geodata()
zones_gdf = geodata["zones_gdf"]
munics_gdf = geodata["munics_gdf"]

# Date info
date_info = get_current_date_info()
weekday = date_info["weekday"]
week_year = date_info["week_year"]

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
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

    with col1:
        transport = st.selectbox("Sistema", ["STC Metro", "Metrobús"])
        st.session_state.transport = transport

    with col3:
        level_div = st.selectbox("Nivel de filtrado", ["Zona", "Alcaldía", "Línea"])
    
    with col2:
        sex = st.selectbox("Sexo", ["Ambos", "Femenino", "Masculino",])

    filter_div = []
    region_column = ""
    with col4:
        if level_div == 'Zona':
            filter_div = st.multiselect("Filtrado", zones_ls, placeholder='Selecciona zonas')
            region_column = 'zona'
            datageom = zones_gdf

        elif level_div == 'Alcaldía':
            filter_div = st.multiselect("Filtrado", munics_ls[st.session_state.transport], placeholder='Selecciona alcaldías')
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
    df_top_stations_crime_trends = query_top_stations_crime_trends(transport, level_div, filter_div, sex, weekday, week_year, radio_int, 1000)
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
