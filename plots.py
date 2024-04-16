import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from colors import LINESM_aux, LINESMB_aux

def plot_top_stations_affluence_trends(df: pd.DataFrame, transport: str):
    df['estacion'] = df['nombre'] + ' ' + df['línea']
    df.reset_index(inplace=True)
    df.set_index('estacion', inplace=True)
    
    if transport == 'STC Metro':
        dict_colours = LINESM_aux
    else:
        dict_colours = LINESMB_aux
    
    fig = go.Figure()
    fig = go.Figure(go.Bar(
        x=df['afluencia_promedio'],
        y=df.index,
        orientation='h',
        width=0.5,
        marker=dict(
            color=[dict_colours[linea] for linea in df['línea']],
        ),
    ))
    
    fig.update_layout(
        title='',
        xaxis_title='Afluencia promedio',
        yaxis_title='Estación',
        margin=dict(t=0), 
        height=200,
    )
    
    return fig

# Por si se quiere redondear las gráficas de barras en su punta
# https://mebaysan.medium.com/rounded-edge-bar-charts-in-plotly-4caf54779cc


def plot_top_stations_crime_trends(df: pd.DataFrame, transport: str):
    df['estacion'] = df['nombre'] + ' ' + df['línea']
    df.reset_index(inplace=True)
    df.set_index('estacion', inplace=True)
    
    if transport == 'STC Metro':
        dict_colours = LINESM_aux
    else:
        dict_colours = LINESMB_aux
    
    fig = go.Figure()
    fig = go.Figure(go.Bar(
        x=df['conteo_delitos'],
        y=df.index,
        orientation='h',
        width=0.5,
        marker=dict(
            color=[dict_colours[linea] for linea in df['línea']],
        ),
    ))
    
    fig.update_layout(
        title='',
        xaxis_title='Conteo histórico de delitos',
        yaxis_title='Estación',
        margin=dict(t=0), 
        height=200,
    )
    
    return fig


