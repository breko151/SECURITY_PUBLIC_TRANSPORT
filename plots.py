import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from colors import LINESM_aux, LINESMB_aux

def plot_top_stations_affluence_trends(df: pd.DataFrame, transport: str):
    df['estacion'] = df['nombre'] + ' ' + df['linea']
    df.reset_index(inplace=True, drop=True)
    df.set_index('estacion', inplace=True)
    # Quien sabe porque al graficar las pone en otro orden los valores, por eso ajustamos
    df = df.sort_values(by=['afluencia_promedio'])
    
    if transport == 'STC Metro':
        dict_colours = LINESM_aux
    else:
        dict_colours = LINESMB_aux
    
    max_affluence = max(df['afluencia_promedio'])
    min_affluence = min(df['afluencia_promedio'])
    x_range = [min_affluence - 0.2 * (max_affluence - min_affluence), max_affluence + 0.1 * (max_affluence - min_affluence)]
    
    fig = go.Figure()
    fig = go.Figure(go.Bar(
        x=df['afluencia_promedio'],
        y=df.index,
        orientation='h',
        width=0.5,
        marker=dict(
            color=[dict_colours[linea] for linea in df['linea']],
        ),
    ))
    
    fig.update_layout(
        title='',
        xaxis_title='Afluencia promedio',
        yaxis_title='Estación',
        margin=dict(t=0), 
        height=300,
    )
    
    fig.update_xaxes(range=x_range)
    
    return fig

# Por si se quiere redondear las gráficas de barras en su punta
# https://mebaysan.medium.com/rounded-edge-bar-charts-in-plotly-4caf54779cc




def plot_top_stations_crime_trends(df: pd.DataFrame, transport: str):
    df['estacion'] = df['nombre'] + ' ' + df['linea']
    df.reset_index(inplace=True, drop=True)
    df.set_index('estacion', inplace=True)
    # Quien sabe porque al graficar las pone en otro orden los valores, por eso ajustamos
    df = df.sort_values(by=['promedio_delitos'])
    
    if transport == 'STC Metro':
        dict_colours = LINESM_aux
    else:
        dict_colours = LINESMB_aux
    
    max_value = max(df['promedio_delitos'])
    min_value = min(df['promedio_delitos'])
    x_range = [min_value - 0.2 * (max_value - min_value), max_value + 0.1 * (max_value - min_value)]
    
    fig = go.Figure()
    fig = go.Figure(go.Bar(
        x=df['promedio_delitos'],
        y=df.index,
        orientation='h',
        width=0.5,
        marker=dict(
            color=[dict_colours[linea] for linea in df['linea']],
        ),
    ))
    
    fig.update_layout(
        title='',
        xaxis_title='Promedio de hechos delictivos',
        yaxis_title='Estación',
        margin=dict(t=0), 
        height=300,
    )
    
    fig.update_xaxes(range=x_range)
    
    return fig


