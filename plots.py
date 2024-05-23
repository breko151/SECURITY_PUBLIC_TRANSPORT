import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from colors import LINESM_aux, LINESMB_aux

# Trend graphs
def plot_top_stations_affluence_trends(df: pd.DataFrame, transport: str):
    df['estacion'] = df['linea'] + ' - ' + df['nombre']
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
        #yaxis_title='Estación',
        margin=dict(t=0, b=0, l=200), 
        height=222,
    )
    
    fig.update_xaxes(range=x_range)
    
    return fig

# Por si se quiere redondear las gráficas de barras en su punta
# https://mebaysan.medium.com/rounded-edge-bar-charts-in-plotly-4caf54779cc

def plot_top_stations_crime_trends(df: pd.DataFrame, transport: str):
    df = df[df['promedio_delitos'] > 0]
    
    df['estacion'] = df['linea'] + ' - ' + df['nombre']
    df.reset_index(inplace=True, drop=True)
    df.set_index('estacion', inplace=True)
    df['promedio_delitos'] = df['promedio_delitos'].apply(float)
    
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
        #yaxis_title='Estación',
        margin=dict(t=0, b=0, l=200), 
        height=222,
    )
    
    fig.update_xaxes(range=x_range)
    
    return fig

# Auxiliary func to make color gradients
def interpolate_color(color1, color2, factor: float):
    def hex_to_rgb(hex_color):
        return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    def rgb_to_hex(rgb_color):
        return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    interpolated = tuple(int(a + (b - a) * factor) for a, b in zip(rgb1, rgb2))
    return rgb_to_hex(interpolated)

# Exploration graphs

def plot_top_crime_station(df: pd.DataFrame):
    df.reset_index(inplace=True, drop=True)
    df.set_index('clase_delito', inplace=True)
    df = df.sort_values(by=['conteo_delitos'])
    
    colors = [interpolate_color('#f0be40', '#d60000', i / (len(df) - 1)) for i in range(len(df))]
    
    fig = go.Figure()
    fig = go.Figure(go.Bar(
        x=df['conteo_delitos'],
        y=df.index,
        orientation='h',
        width=0.5,
        marker=dict(color=colors)
    ))

    fig.update_layout(
        title='',
        xaxis_title='Conteo de hechos delictivos',
        yaxis_title='Clase delito',
        margin=dict(t=0,b=0,), 
        height=321,
        dragmode=False,
    )
    
    return fig

def plot_crime_exploration_gender(df: pd.DataFrame):
    colors = ['#3974c8' if label == 'Masculino' else '#d779c3' for label in df['sexo_victima']]
    
    fig = go.Figure(data=[go.Pie(
        labels=df['sexo_victima'],
        values=df['conteo_delitos'],
        hole=.6,
        hoverinfo='none',
        hovertemplate='%{label}: %{value}<extra></extra>',
        marker=dict(colors=colors)
    )])

    fig.update_layout(
        margin=dict(t=0), 
        height=300,
        annotations=[dict(text='', x=0.5, y=0.5, font_size=20, showarrow=False)],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        ),
        dragmode=False,
    )

    return fig

def plot_crime_exploration_age_group(df: pd.DataFrame):
    df = df.sort_values(by=['grupo_quinquenal_inegi'], ascending=False)
    df = df[df['conteo_delitos'] > 0]
    
    if len(df) > 1:
        colors = [interpolate_color('#30679e', '#32cd71', i / (len(df) - 1)) for i in range(len(df))]
    else:
        colors = ['#30679e']
    
    fig = go.Figure(go.Bar(
        y=df['grupo_quinquenal_inegi'],
        x=df['conteo_delitos'],
        orientation='h',
        width=0.5,
        marker=dict(color=colors),
        hoverinfo='none',
        hovertemplate='%{y}: %{x}<extra></extra>',
    ))

    fig.update_layout(
        xaxis_title='Conteo de grupo',
        yaxis_title='Grupo de edad',
        margin=dict(t=0), 
        height=300,
        dragmode=False,
    )

    return fig

def plot_crime_exploration_day_parts(df: pd.DataFrame):
    #grupos_horas = ['Madrugada: (0-7 hrs)', 'Mañana (7-12 hrs)', 'Tarde (12-19 hrs)', 'Noche (19-24 hrs)']
    dict_part_days_aux = {
        'Madrugada': 0,
        'Mañana': 1,
        'Tarde': 2,
        'Noche': 3,
    }
    dict_part_days_complete = {
        'Madrugada': 'Madrugada (0-7hrs)',
        'Mañana': 'Mañana (7-12hrs)',
        'Tarde': 'Tarde (12-19hrs)',
        'Noche': 'Noche (19-24hrs)',
    }
    df['parte_dia_aux'] = df['parte_dia'].map(dict_part_days_aux)
    df['parte_dia_completa'] = df['parte_dia'].map(dict_part_days_complete)
    df = df.sort_values(by=['parte_dia_aux'], ascending=True)
    grupos_horas = df['parte_dia_completa']
    valores = df['conteo_delitos']
    
    max_value = max(valores)
    grosor = [0.10 if valor == max_value else 0.03 for valor in valores]

    colores_relleno = ['#7e58ad', '#7dccdc', '#ff8f00', '#5c74a7']
    colores_contorno = ['#482c6a', '#518893', '#b56500', '#324264']
    bordes_ancho = [3 if valor == max_value else 0 for valor in valores]

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=grupos_horas,
        values=valores,
        hole=0.3,
        textinfo='percent',
        #insidetextorientation='radial',
        hoverinfo='none',
        hovertemplate='%{label}: %{value}<extra></extra>',
        marker=dict(colors=colores_relleno, line=dict(color=colores_contorno, width=bordes_ancho)),
        pull=grosor,
        direction='clockwise',
        rotation=0,
        sort=False,
    ))
    fig.update_layout(
        margin=dict(t=0), 
        height=300,
        showlegend=False,
        dragmode=False,
    )

    return fig

def plot_crime_exploration_distances(df: pd.DataFrame):
    x = df['distancia']
    min_value = 0
    max_value = x.max()
    
    # Crear bins manualmente
    bins = np.arange(min_value, max_value + 100, 100)
    counts, edges = np.histogram(x, bins=bins)
    

    bin_labels = [f'{int(edges[i])}-{int(edges[i+1])}' for i in range(len(edges)-1)]
    colors = [interpolate_color('#af3131', '#4d238d', i / (len(bin_labels) - 1)) for i in range(len(bin_labels))]

    fig = go.Figure(go.Bar(
        x=bin_labels,
        y=counts,
        marker=dict(color=colors),
        hoverinfo='none',
        hovertemplate='%{x}m: %{y}<extra></extra>',
    ))

    fig.update_layout(
        xaxis_title='Distancia en metros',
        yaxis_title='Conteo de distancias',
        margin=dict(t=0), 
        height=280,
        showlegend=False,
        dragmode=False,
    )

    return fig
