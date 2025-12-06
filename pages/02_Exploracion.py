import streamlit as st
from streamlit_plotly_events import plotly_events
from src.config import lines_ls, weekdays_queries_ls, crime_vars_queries_ls, get_current_date_info, get_station
from src.loader import load_geodata
from src.querys import query_top_crimes_historical, query_crimes_exploration_gender, query_crimes_exploration_age_group, query_crimes_exploration_distances, query_crimes_part_of_day
from src.plots import plot_transport_stations, plot_top_crime_station, plot_crime_exploration_gender, plot_crime_exploration_age_group, plot_crime_exploration_distances, plot_crime_exploration_day_parts

st.set_page_config(page_title="Exploración", page_icon="🔍", layout="wide")

# Load data
geodata = load_geodata()
zones_gdf = geodata["zones_gdf"]
df_stations_metro = geodata["df_stations_metro"]
df_stations_metrobus = geodata["df_stations_metrobus"]

# Date info
date_info = get_current_date_info()
weekday = date_info["weekday"]

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

if 'selected_id' not in st.session_state:
    st.session_state.selected_id = []

with st.container():
    st.header("🔍 Explora las estaciones")
    
    col1, colmid, col2 = st.columns([20, 1, 20])
    # Map column.
    with col1:
        transport = st.selectbox("Sistema", ["STC Metro", "Metrobús"])
        df_stations_transport = df_stations_metro if transport == 'STC Metro' else df_stations_metrobus
        st.session_state.transport = transport
        radio_int = 540 if transport == 'STC Metro' else 270
        radio_ = '540' if transport == 'STC Metro' else '270'
        fig = plot_transport_stations(zones_gdf, transport)
        selected_click_cve_est = plotly_events(fig, click_event=True,)
        st.session_state.selected_id.append(selected_click_cve_est)
    
    with col2:
        # Second column validation.
        if 'selected_id' not in st.session_state:
            st.subheader("Seleccione una estación")
            st.write(f"Al seleccionar aparecerá la siguiente información sobre los hechos delictivos ocurridos dentro un radio de {radio_} metros alrededor de la estación seleccionada:")
            st.write(" - Top delitos más frecuentes")
            st.write(" - Comparación de géneros de víctimas")
            st.write(" - Rangos de edad más vulnerables")
            st.write(" - Comportamiento de la distancia delito-estación")
            st.write(" - Partes del día más delictivas")
        else:
            if st.session_state.selected_id is not None and st.session_state.selected_id[-1]:
                last_selected_id = st.session_state.selected_id[-1]
                if last_selected_id[0]['curveNumber'] >= len(lines_ls[transport]) + 1:
                    if transport == 'STC Metro':
                        df_stations_metro_filtered = df_stations_metro.iloc[last_selected_id[0]['pointIndex']]
                        cve_est = df_stations_metro_filtered['cve_est'] 
                    else:
                        df_stations_metrobus_filtered = df_stations_metrobus.iloc[last_selected_id[0]['pointIndex']]
                        cve_est = df_stations_metrobus_filtered['cve_est'] 
                    
                    name_est = get_station(df_stations_transport, cve_est, "nombre")
                    tipo_est = get_station(df_stations_transport, cve_est, "tipo")
                    linea_est = get_station(df_stations_transport, cve_est, "linea")
                    nivel_circul = get_station(df_stations_transport, cve_est, "nivel_circulacion_transporte")
                    name_est_line = name_est + f' (L{linea_est[1:]})'
                    st.subheader(f'{name_est_line.upper()}')
                    st.write(f'Tipo de estación: {tipo_est.capitalize()}')
                    st.write(f'Nivel de circulación: {nivel_circul.capitalize()}')
                    st.write("##### Top 10 delitos históricos más frecuentes (2019-2023)")
                    
                    # SQL queries
                    df_top_crimes_historical = query_top_crimes_historical(transport, cve_est, radio_int, 10)
                    fig = plot_top_crime_station(df_top_crimes_historical)
                    fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})

                else:
                    st.subheader("Seleccione una estación")
                    st.write(f"Al seleccionar aparecerá la siguiente información sobre los hechos delictivos ocurridos dentro un radio de {radio_} metros alrededor de la estación seleccionada:")
                    st.write(" - Top delitos más frecuentes")
                    st.write(" - Comparación de géneros de víctimas")
                    st.write(" - Rangos de edad más vulnerables")
                    st.write(" - Comportamiento de la distancia delito-estación")
                    st.write(" - Partes del día más delictivas") 
            
            else:
                radio_ = '540' if transport == 'STC Metro' else '270'
                st.subheader("Seleccione una estación")
                st.write(f"Al seleccionar aparecerá la siguiente información sobre los hechos delictivos ocurridos dentro un radio de {radio_} metros alrededor de la estación seleccionada:")
                st.write(" - Top delitos más frecuentes")
                st.write(" - Comparación de géneros de víctimas")
                st.write(" - Rangos de edad más vulnerables")
                st.write(" - Comportamiento de la distancia delito-estación")
                st.write(" - Partes del día más delictivas") 
    
    if 'selected_id' in st.session_state and (st.session_state.selected_id is not None and st.session_state.selected_id[-1]):
        last_selected_id = st.session_state.selected_id[-1]
        if last_selected_id[0]['curveNumber'] >= len(lines_ls[transport]) + 1:
            col3, col4 = st.columns([2, 3])
            with col3:    
                weekday_selected = st.selectbox('Día de la semana', weekdays_queries_ls, index=weekdays_queries_ls.index(weekday))
            with col4:    
                crime_var_selected = st.selectbox('Variable delito', crime_vars_queries_ls)
            
            df_crimes_exploration_gender = query_crimes_exploration_gender(transport, cve_est, radio_int, weekday_selected, crime_var_selected)
            df_crimes_exploration_age_group = query_crimes_exploration_age_group(transport, cve_est, radio_int, weekday_selected, crime_var_selected)
            df_crimes_exploration_distances = query_crimes_exploration_distances(transport, cve_est, radio_int, weekday_selected, crime_var_selected)    
            df_crimes_exploration_part_of_day = query_crimes_part_of_day(transport, cve_est, radio_int, weekday_selected, crime_var_selected)
            
            total_rows_sample = df_crimes_exploration_gender['conteo_delitos'].sum()
            
            if total_rows_sample > 0:
                st.markdown(f"{total_rows_sample} {' carpetas de investigación encontrada' if total_rows_sample == 1 else ' carpetas de investigación encontradas'}") 
                col6, col7, col8, col9 = st.columns([2, 2, 3, 2])
                with col6:
                    st.write("##### Comparación de género")
                    fig = plot_crime_exploration_gender(df_crimes_exploration_gender)
                    fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})
    
                with col7:
                    st.write("##### Distribución de la edad")
                    fig = plot_crime_exploration_age_group(df_crimes_exploration_age_group)
                    fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})
                    
                with col8:
                    st.write("##### Comparación de momentos del día")
                    fig = plot_crime_exploration_day_parts(df_crimes_exploration_part_of_day)
                    fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})
                    
                with col9:
                    st.write("##### Distancia delito-estación")
                    fig = plot_crime_exploration_distances(df_crimes_exploration_distances)
                    fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})
            else:
                st.write('Se encontraron 0 registros coincidentes para los filtros aplicados')
