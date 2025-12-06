import streamlit as st
import pandas as pd
import numpy as np
from streamlit_plotly_events import plotly_events
from src.config import (
    week_year, 
    year, 
    last_week_of_year, 
    week_of_month, 
    get_monday_week_year, 
    get_week_date_range
)
from src.loader import (
    load_geodata, 
    load_thresholds_crime_model, 
    load_crime_model, 
    load_afflu_forecast_values
)
from src.plots import plot_predictive_map, plot_complementary_predictive_map

st.set_page_config(page_title="Predicciones", page_icon="📈", layout="wide")

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

st.title("📈 Predicción del nivel de riesgo delictivo")

# Initialize session state for map clicks if not exists
if 'selected_click_pred_map' not in st.session_state:
    st.session_state.selected_click_pred_map = []

# Load geodata
geodata = load_geodata()
munics_gdf = geodata["munics_gdf"]
df_stations_metro = geodata["df_stations_metro"]
df_stations_metrobus = geodata["df_stations_metrobus"]

col1, col2 = st.columns([1, 5])

# First container.
with st.container():
    col1, col2, col3 = st.columns([1, 3, 1])
    
    list_categ_crimes_ = {
        'Robo a transeúnte y pasajero en transporte público': 'Robo a transeúnte y pasajero en transporte público',
        'Robo de vehículo y autopartes': 'Robo de vehículo y autopartes',
        'Delitos sexuales': 'Delitos sexuales',
        'Lesiones': 'Lesiones',
        'Amenazas': 'Amenazas',
        'Fraude': 'Fraude',
    }
    
    transport = col1.selectbox("Sistema", [
        "STC Metro",
        "Metrobús",
    ])
    level_div = 'Alcaldía'
    
    categ_crime = col2.selectbox("Categoría delictiva", list_categ_crimes_.keys())
    categ_crime_ok = list_categ_crimes_[categ_crime]
    
    sex = col3.selectbox("Sexo a considerar en la predicción", ["Ambos", "Femenino", "Masculino"])
    id_sex = 0
    if sex == 'Femenino':
        id_sex = 1
    
    col4, col5, col6 = st.columns([4, 1, 2])
    weeks_forward = col4.slider(f'No. de semanas futuras a predecir', 0, last_week_of_year - int(week_year), 0)
    week_month_aux = week_of_month(get_monday_week_year(weeks_forward + int(week_year), year))
    
    text_num_week_forward = 'semana '
    if weeks_forward == 0:
        text_num_week_forward += 'actual'
    else:
        text_num_week_forward += str(weeks_forward + int(week_year))
    
    
    # Validate which grouping dataset belongs the model (3, 4, 6, 7)
    grouped_dataset_id = 0
    columns_input_model = []
    regions = []
    if level_div == 'Alcaldía' and sex == 'Ambos':
        # print('Agrupamiento 3')
        grouped_dataset_id = 3
        columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'semana_1',]
        if transport == 'STC Metro':
            regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                'Tláhuac', 'Venustiano Carranza', 'Álvaro Obregón']
        else:
            regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                'Tlalpan', 'Venustiano Carranza', 'Álvaro Obregón', 'Xochimilco']
            
        inputs_model_ls = []
        for region in regions:
            inputs_model_ls.append([week_month_aux, region, categ_crime_ok])
        input_model_df_partial = pd.DataFrame(inputs_model_ls, columns=columns_input_model[:-1])
        
        thresholds_pred = load_thresholds_crime_model(transport, grouped_dataset_id)
        threshold_grouped_dataset = thresholds_pred[(thresholds_pred['categ_delito'] == categ_crime_ok)]['percentil'].to_list()[0]
        
    
    elif level_div == 'Alcaldía' and sex != 'Ambos':
        # print('Agrupamiento 4')
        grouped_dataset_id = 4
        columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'sexo_victima', 'semana_1',]
        if transport == 'STC Metro':
            regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                'Tláhuac', 'Venustiano Carranza', 'Álvaro Obregón']
        else:
            regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                'Tlalpan', 'Venustiano Carranza', 'Álvaro Obregón', 'Xochimilco']
            
        inputs_model_ls = []
        for region in regions:
            inputs_model_ls.append([week_month_aux, region, categ_crime_ok, id_sex, ])
        input_model_df_partial = pd.DataFrame(inputs_model_ls, columns=columns_input_model[:-1])
        
        thresholds_pred = load_thresholds_crime_model(transport, grouped_dataset_id)
        threshold_grouped_dataset = thresholds_pred[(thresholds_pred['categ_delito'] == categ_crime_ok) & (thresholds_pred['sexo'] == id_sex)]['percentil'].to_list()[0]
        
    
    # print('Resultados')
    
    # Load of pickle model and reading of affluence predictions
    crime_model = load_crime_model(transport, grouped_dataset_id)
    afflu_fc_values = load_afflu_forecast_values(transport, grouped_dataset_id)
    afflu_fc_values_filtered = afflu_fc_values[afflu_fc_values['semana_anio'] == weeks_forward + int(week_year) - 1]
    input_model_df = input_model_df_partial.merge(afflu_fc_values_filtered, left_on=columns_input_model[1], right_on=['region'])
    input_model_df.rename(columns={'afluencia': 'semana_1'}, inplace=True)
    input_model_df = input_model_df[columns_input_model]
    # print(input_model_df)
    
    # Predictions
    preds = crime_model.predict(input_model_df)
    # print('Predicciones para semana {}:'.format(weeks_forward + int(week_year)))
    df_preds = pd.DataFrame(columns=['valor'])
    df_preds['valor'] = preds
    df_preds['valor'] = df_preds['valor'].replace({'High': 'Riesgo elevado', 'Low': 'Riesgo moderado'})
    # print(df_preds)
    input_model_df_preds = pd.concat([input_model_df, df_preds], axis=1)
    
    # Get the spatial granularity for then pass it as an arg to the predictive map plot function
    if level_div == 'Alcaldía':
        region_gdf_merge = munics_gdf.merge(input_model_df_preds[['alcaldia', 'valor']], left_on=['NOMGEO'], right_on=['alcaldia'])
        region_column = 'CVE_MUN'
        region_column_name = 'NOMGEO'
        region_column_name_2 = 'alcaldia'
    else:
        region_gdf_merge = munics_gdf.merge(input_model_df_preds[['sector', 'valor']], left_on=['sector'], right_on=['sector'])
        region_column = 'sector'
        region_column_name = 'sector'
        region_column_name_2 = 'sector'
    region_gdf_merge = region_gdf_merge.sort_values(by=['valor'])
    # print(region_gdf_merge)

with st.container():
    metric_unique_values = np.sort(region_gdf_merge['valor'].unique())
    col7, col78, col8 = st.columns([20, 1, 30])
    with col7:
        fig = plot_predictive_map(region_gdf_merge, transport, metric_unique_values)
        fig.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
        # Interactive map was done thanks to
        # https://discuss.streamlit.io/t/interactive-plot-get-which-point-a-user-clicked/14596
        # https://github.com/null-jones/streamlit-plotly-events
        # https://orosz-attila-covid-19-dashboard-streamlit-app-kmmvgj.streamlit.app/
        #https://towardsdatascience.com/highlighting-click-data-on-plotly-choropleth-map-377e721c5893
        selected_click_aux = plotly_events(fig, click_event=True,)
        st.session_state.selected_click_pred_map.append(selected_click_aux)
        # print('click', st.session_state.selected_click_pred_map)
    
    with col8:
        if len(st.session_state.selected_click_pred_map) > 0:
            last_click = st.session_state.selected_click_pred_map[-1]
            if len(last_click) > 0:
                #col8.write(last_click[0])
                metric_value = metric_unique_values[last_click[0]['curveNumber']]
                region_gdf_merge_ = region_gdf_merge[region_gdf_merge['valor'] == metric_value]
                region_gdf_aux = region_gdf_merge_.iloc[last_click[0]['pointIndex']]
                region_gdf_aux = pd.DataFrame(region_gdf_aux.values.reshape(1, -1), columns=region_gdf_aux.index)
                region_name = region_gdf_aux[region_column_name].to_list()[0]
                level_risk = region_gdf_aux['valor'].to_list()[0]
                # print(region_name)
                # print(level_risk)
                #col8.write(munics_gdf_aux)
                #col8.write(st.session_state.selected_click_pred_map)
                #col8.markdown(f'##\t{region_name}')
                if level_risk == 'Riesgo elevado':
                    threshold_txt = f'{str(int(threshold_grouped_dataset + 1))}+ hechos delictivos'
                    # Checar luego para alinear texto si queremos
                    col8.markdown(f'<span style="color: black; font-size: 20px; font-weight: bold;">{level_risk} ({threshold_txt})</span> <br><span style="color: black; font-size: 14px;">para las estaciones en <b>{region_name}</b> durante la <b>{text_num_week_forward}</b></span> <span style="color: #a5a5a5; font-size: 14px;">({get_week_date_range(weeks_forward + int(week_year), year)})</span>', unsafe_allow_html=True)
                else:
                    if threshold_grouped_dataset > 0:
                        threshold_txt = f'0 a {str(int(threshold_grouped_dataset))} hechos delictivos'
                    else:
                        threshold_txt = 'casi nulo'
                    # Checar luego para alinear texto si queremos
                    col8.markdown(f'<span style="color: black; font-size: 20px; font-weight: bold;">{level_risk} ({threshold_txt})</span> <br><span style="color: black; font-size: 14px;">para las estaciones en <b>{region_name}</b> durante la <b>{text_num_week_forward}</b></span> <span style="color: #a5a5a5; font-size: 14px;">({get_week_date_range(weeks_forward + int(week_year), year)})</span>', unsafe_allow_html=True)
                
                fig2 = plot_complementary_predictive_map(region_gdf_merge, transport, region_gdf_aux, region_column)
                fig2.update_layout({"uirevision": "foo"}, overwrite=True, dragmode=False)
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False,})
                
                
                with st.expander("Estaciones afectadas", expanded=False):
                    if region_column == 'CVE_MUN':
                        if transport == 'Metrobús':
                            df_stations_metro_aux = df_stations_metrobus[(df_stations_metrobus['alcaldia'] == region_name)]
                        else:
                            df_stations_metro_aux = df_stations_metro[(df_stations_metro['alcaldia'] == region_name)]
                    else:
                        df_stations_metro_aux = df_stations_metro[(df_stations_metro['sector'] == region_name)]
                        
                    
                    # Stations within the region
                    lines = df_stations_metro_aux['linea'].unique()
                    
                    list_text_stations_per_line = []
                    for l in lines:
                        stations_per_line = []
                        stations_line_aux = df_stations_metro_aux[(df_stations_metro_aux['linea'] == l)]
                        for ind, row in stations_line_aux.iterrows():
                            name_st = row['nombre']
                            stations_per_line.append(name_st)
                        text_aux = ""
                        if len(stations_per_line) == 1:
                            text_aux = l + ": " + stations_per_line[0]
                        else:
                            text_aux = l + ": " + ", ".join(stations_per_line[:-1]) + " y " + stations_per_line[-1]
                        list_text_stations_per_line.append(text_aux)

                    for elemento in list_text_stations_per_line:
                        st.write(f'<li style="font-size: 12px;">{elemento}</li>', unsafe_allow_html=True)
                    radio_ = '540' if transport == 'STC Metro' else '270'
                    
                    st.write(f'<span style="font-size: 12px; color: #a5a5a5"><b>Nota:</b> El nivel de riesgo predictivo se predice considerando la evidencia delictiva tanto dentro como en un radio de {radio_}m alrededor de las estaciones, para el caso del {transport}.</span>', unsafe_allow_html=True)
            else:
                st.subheader("Seleccione una alcaldía")
