import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiPolygon, Polygon, MultiPoint
import json
from .colors import LINESM_aux, LINESMB_aux, LINESM, LINESMB
from src.loader import load_geodata

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

# Helper functions for trends
def normalize_size(value, min_value, max_value, min_size=4, max_size=8):
    return ((value - min_value) / (max_value - min_value)) * (max_size - min_size) + min_size

def generate_geom_grouped(geom_df, level):
    # Group regions by a certain level

    groups = geom_df.groupby(level)
    grouped_geoms = []
    sectors = []

    for sector, group in groups:
        grouped_geom = unary_union(group['geometry'])
        grouped_geoms.append(grouped_geom)
        sectors.append(sector)

    data_dict_grouped_geoms = {
        level: sectors,
        'geometry': grouped_geoms
    }

    #print(data_dict_grouped_geoms)

    return gpd.GeoDataFrame(data_dict_grouped_geoms, geometry='geometry')

def label_percentiles_3_parts(row, percentiles):
    if row <= percentiles.iloc[0]:
        return 1
    elif row <= percentiles.iloc[1]:
        return 2
    else:
        return 3

def generate_ranges_labels_percentiles_3_parts(percentiles, min_value, max_value):
    sorted_values = sorted(percentiles)
    ranges = []
    ranges.append(f'{min_value} - {round(sorted_values[0], 2)}')
    
    for i in range(1, len(sorted_values)):
        ranges.append(f'{round(round(sorted_values[i-1], 2) + 0.01, 2)} - {round(sorted_values[i], 2)}')

    ranges.append(f'{round(round(sorted_values[-1], 2) + 0.1, 2)} - {max_value}')
    
    return ranges

# Plot the stations showing criminal trends
def plot_crime_trend_stations(datageom, df: pd.DataFrame, transport: str, level_div: str, filter_div: list):
    geodata = load_geodata()
    df_stations_metro = geodata["df_stations_metro"]
    df_stations_metrobus = geodata["df_stations_metrobus"]
    metro_lines = geodata["metro_lines"]
    mb_lines = geodata["mb_lines"]
    lineas_cdmx = geodata["lineas_cdmx"]

    region_gdf_cp = datageom.copy(deep=True)
    center_geom = region_gdf_cp['geometry'].to_list()[0].centroid
    df['promedio_delitos'] = df['promedio_delitos'].astype(float)
    
    dict_colors_percentile = {
        1: '#8fce00',
        2: '#ffa200',
        3: 'red',
    }
    
    fig = go.Figure()
    
    fig.update_layout(
        mapbox=dict(center=dict(lat=center_geom.y, lon=center_geom.x),zoom=9.8),
    )

    # Show stations and lines
    if transport == 'STC Metro':
        df_stations_metro_complete = df_stations_metro.merge(df[['linea', 'nombre', 'promedio_delitos']], on=['linea', 'nombre'])
        min_value = df_stations_metro_complete['promedio_delitos'].min()
        max_value = df_stations_metro_complete['promedio_delitos'].max()
        percentile_values = df_stations_metro_complete['promedio_delitos'].quantile([0.33, 0.66])
        df_stations_metro_complete['clase'] = df_stations_metro_complete['promedio_delitos'].apply(label_percentiles_3_parts, percentiles=percentile_values)
        colors_ls = [dict_colors_percentile[class_] for class_ in df_stations_metro_complete['clase'].to_list()]
        
        lines_unique = df_stations_metro_complete['linea'].unique()
        colors_classes_unique = list(dict_colors_percentile.keys())
        ranges_classes_unique = generate_ranges_labels_percentiles_3_parts(percentile_values, min_value, max_value)
        
        # Map boundaries of regions selected
        if level_div != 'linea':
            # Filter geometry of choroplet
            if filter_div != []:
                region_gdf_cp_aux = region_gdf_cp[region_gdf_cp[level_div].isin(filter_div)]
            else:
                region_gdf_cp_aux = region_gdf_cp
            
            for ind, row in region_gdf_cp_aux.iterrows():
                geometry_ = row['geometry']
                lon_geom, lat_geom = geometry_.exterior.xy
                lon_geom = np.array(lon_geom).tolist()
                lat_geom = np.array(lat_geom).tolist()
                trace_boundary = go.Scattermapbox(
                    lon=lon_geom,
                    lat=lat_geom,
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(196, 223, 225, 0.15)',
                    line=dict(color='#65999d', width=2),
                    hoverinfo='none',
                    showlegend=False,
                )
                fig.add_trace(trace_boundary)
                
            region_gdf_cp_aux[level_div] = 'union'
            region_gdf_grouped = generate_geom_grouped(region_gdf_cp_aux, level_div)
            polygon_ = region_gdf_grouped['geometry'].to_list()[0]
            
            # Filter just the fragment of line which intersects the area of the choroplet
            for line in lines_unique:
                metro_lines_aux = metro_lines[metro_lines['LINEA'] == line[1:]]
                metro_lines_aux['geometry_yx'] = metro_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
                lines_ = metro_lines_aux['geometry_yx'].iloc[0]
                
                intersection = lines_.intersection(polygon_)
                if intersection.type == 'MultiLineString':
                    for ind_line in list(intersection.geoms):
                        if ind_line.is_empty:
                            pass
                        else:
                            coords_pts = [[coord[0], coord[1]] for coord in ind_line.coords]
                            line_trace = go.Scattermapbox(
                                mode='lines',
                                lon = [coord[0] for coord in coords_pts],
                                lat = [coord[1] for coord in coords_pts],
                                line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                                hoverinfo='none',
                                showlegend=False,
                            )
                            fig.add_trace(line_trace)
                else:
                    if intersection.is_empty:
                        pass
                    else:
                        coords_pts = [[coord[0], coord[1]] for coord in intersection.coords]
                        line_trace = go.Scattermapbox(
                            mode='lines',
                            lon = [coord[0] for coord in coords_pts],
                            lat = [coord[1] for coord in coords_pts],
                            line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                            hoverinfo='none',
                            showlegend=False,
                        )
        else:
            geometry_ = lineas_cdmx['geometry'].to_list()[0]
            lon_geom, lat_geom = geometry_.exterior.xy
            lon_geom = np.array(lon_geom).tolist()
            lat_geom = np.array(lat_geom).tolist()
            trace_boundary = go.Scattermapbox(
                lon=lon_geom,
                lat=lat_geom,
                mode='lines',
                fill='toself',
                fillcolor='rgba(196, 223, 225, 0.15)',
                line=dict(color='#65999d', width=2),
                hoverinfo='none',
                showlegend=False,
            )
            fig.add_trace(trace_boundary)
            
            # Map all the lines
            for line in lines_unique:
                metro_lines_aux = metro_lines[metro_lines['LINEA'] == line[1:]]
                metro_lines_aux['geometry_yx'] = metro_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
                lines_ = metro_lines_aux['geometry_yx'].iloc[0]
                coords_pts = [[coord[0], coord[1]] for coord in lines_.coords]
                line_trace = go.Scattermapbox(
                    mode='lines',
                    lon = [coord[0] for coord in coords_pts],
                    lat = [coord[1] for coord in coords_pts],
                    line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                    hoverinfo='none',
                    showlegend=False,
                )
                fig.add_trace(line_trace)
        
        # Plot stations with different sizes and colors
        for color_cl, range_cl in zip(colors_classes_unique, ranges_classes_unique):
            df_stations_metro_aux = df_stations_metro_complete[(df_stations_metro_complete['clase'] == color_cl)]
            lats = df_stations_metro_aux['latitud']
            lons = df_stations_metro_aux['longitud']
            ids = df_stations_metro_aux['cve_est']
            lines = df_stations_metro_aux['linea']
            names = df_stations_metro_aux['nombre']
            values_ = df_stations_metro_aux['promedio_delitos']
            marker_sizes_colors = [normalize_size(value, min_value, max_value, min_size=10, max_size=16) for value in values_]
            marker_sizes_white = [normalize_size(value, min_value, max_value, min_size=4, max_size=9) for value in values_]
            
            scatter_trace = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=marker_sizes_colors,
                        color=dict_colors_percentile[color_cl],
                        opacity = 1.0,
                    ),
                    hoverinfo='none',
                    name=range_cl,
                    showlegend=True,
            )
            scatter_trace_2 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=marker_sizes_white,
                        color='white',
                        opacity = 1.0,
                    ),
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
                    showlegend=False,
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)

    else:
        df_stations_metrobus_complete = df_stations_metrobus.merge(df[['linea', 'nombre', 'promedio_delitos']], on=['linea', 'nombre'])
        min_value = df_stations_metrobus_complete['promedio_delitos'].min()
        max_value = df_stations_metrobus_complete['promedio_delitos'].max()
        percentile_values = df_stations_metrobus_complete['promedio_delitos'].quantile([0.33, 0.66])
        df_stations_metrobus_complete['clase'] = df_stations_metrobus_complete['promedio_delitos'].apply(label_percentiles_3_parts, percentiles=percentile_values)
        colors_ls = [dict_colors_percentile[class_] for class_ in df_stations_metrobus_complete['clase'].to_list()]
        
        lines_unique = df_stations_metrobus_complete['linea'].unique()
        colors_classes_unique = list(dict_colors_percentile.keys())
        ranges_classes_unique = generate_ranges_labels_percentiles_3_parts(percentile_values, min_value, max_value)
        
        # Map boundaries of regions selected
        if level_div != 'linea':
            # Filter geometry of choroplet
            if filter_div != []:
                region_gdf_cp_aux = region_gdf_cp[region_gdf_cp[level_div].isin(filter_div)]
            else:
                region_gdf_cp_aux = region_gdf_cp
            
            for ind, row in region_gdf_cp_aux.iterrows():
                geometry_ = row['geometry']
                lon_geom, lat_geom = geometry_.exterior.xy
                lon_geom = np.array(lon_geom).tolist()
                lat_geom = np.array(lat_geom).tolist()
                trace_boundary = go.Scattermapbox(
                    lon=lon_geom,
                    lat=lat_geom,
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(196, 223, 225, 0.15)',
                    line=dict(color='#65999d', width=2),
                    hoverinfo='none',
                    showlegend=False,
                )
                fig.add_trace(trace_boundary)
                
            region_gdf_cp_aux[level_div] = 'union'
            region_gdf_grouped = generate_geom_grouped(region_gdf_cp_aux, level_div)
            polygon_ = region_gdf_grouped['geometry'].to_list()[0]
            
            # Filter just the fragment of line which intersects the area of the choroplet
            for line in lines_unique:
                metrobus_lines_aux = mb_lines[mb_lines['LINEA'].str[-1] == line[1:]]
                metrobus_lines_aux['geometry_yx'] = metrobus_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
                lines_ = metrobus_lines_aux['geometry_yx']
                
                for i in range(len(lines_)):
                    line_ = lines_.iloc[i]
                    intersection = line_.intersection(polygon_)
                    if intersection.type == 'MultiLineString':
                        for ind_line in list(intersection.geoms):
                            if ind_line.is_empty:
                                pass
                            else:
                                coords_pts = [[coord[0], coord[1]] for coord in ind_line.coords]
                                line_trace = go.Scattermapbox(
                                    mode='lines',
                                    lon = [coord[0] for coord in coords_pts],
                                    lat = [coord[1] for coord in coords_pts],
                                    line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                                    hoverinfo='none',
                                    showlegend=False,
                                )
                                fig.add_trace(line_trace)
                    else:
                        if intersection.is_empty:
                            pass
                        else:
                            coords_pts = [[coord[0], coord[1]] for coord in intersection.coords]
                            line_trace = go.Scattermapbox(
                                mode='lines',
                                lon = [coord[0] for coord in coords_pts],
                                lat = [coord[1] for coord in coords_pts],
                                line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                                hoverinfo='none',
                                showlegend=False,
                            )
                            fig.add_trace(line_trace) 
        else:
            geometry_ = lineas_cdmx['geometry'].to_list()[0]
            lon_geom, lat_geom = geometry_.exterior.xy
            lon_geom = np.array(lon_geom).tolist()
            lat_geom = np.array(lat_geom).tolist()
            trace_boundary = go.Scattermapbox(
                lon=lon_geom,
                lat=lat_geom,
                mode='lines',
                fill='toself',
                fillcolor='rgba(196, 223, 225, 0.15)',
                line=dict(color='#65999d', width=2),
                hoverinfo='none',
                showlegend=False,
            )
            fig.add_trace(trace_boundary)
            
            # Map all the lines
            for line in lines_unique:
                metrobus_lines_aux = mb_lines[mb_lines['LINEA'].str[-1] == line[1:]]
                metrobus_lines_aux['geometry_yx'] = metrobus_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
                lines_ = metrobus_lines_aux['geometry_yx']
                
                for i in range(len(lines_)):
                    line_ = lines_.iloc[i]
                    coords_pts = [[coord[0], coord[1]] for coord in line_.coords]
                    line_trace = go.Scattermapbox(
                        mode='lines',
                        lon = [coord[0] for coord in coords_pts],
                        lat = [coord[1] for coord in coords_pts],
                        line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                        hoverinfo='none',
                        showlegend=False,
                    )
                    fig.add_trace(line_trace)

        # Plot stations with different sizes and colors
        for color_cl, range_cl in zip(colors_classes_unique, ranges_classes_unique):
            df_stations_metrobus_aux = df_stations_metrobus_complete[(df_stations_metrobus_complete['clase'] == color_cl)]
            lats = df_stations_metrobus_aux['latitud']
            lons = df_stations_metrobus_aux['longitud']
            ids = df_stations_metrobus_aux['cve_est']
            lines = df_stations_metrobus_aux['linea']
            names = df_stations_metrobus_aux['nombre']
            values_ = df_stations_metrobus_aux['promedio_delitos']
            marker_sizes_colors = [normalize_size(value, min_value, max_value, min_size=8, max_size=16) for value in values_]
            marker_sizes_white = [normalize_size(value, min_value, max_value, min_size=4, max_size=7) for value in values_]
            
            scatter_trace = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=marker_sizes_colors,
                        color=dict_colors_percentile[color_cl],
                        opacity = 1.0,
                    ),
                    hoverinfo='none',
                    name=range_cl,
                    showlegend=True,
            )
            scatter_trace_2 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=marker_sizes_white,
                        color='white',
                        opacity = 1.0,
                    ),
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
                    showlegend=False,
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)
    
    fig.update_layout(
        title_text='',
        margin=dict(t=0, l=0, r=21, b=0),
        title=dict(
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
        ),
        legend=dict(
            title=dict(
                text='Puntos calientes',
                font=dict(
                    size=14,
                    color='black',
                )
            ),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        legend_title=dict(side='top center'),
        showlegend=True,

        mapbox_style="carto-positron",
        mapbox=dict(
            pitch=0,
        ),
        height = 480,
        autosize=True,
        dragmode=False,
    )
    
    return fig

# Plot the stations showing criminal trends
def plot_transport_stations(datageom, transport: str):
    geodata = load_geodata()
    df_stations_metro = geodata["df_stations_metro"]
    df_stations_metrobus = geodata["df_stations_metrobus"]
    metro_lines = geodata["metro_lines"]
    mb_lines = geodata["mb_lines"]
    lineas_cdmx = geodata["lineas_cdmx"]

    region_gdf_cp = datageom.copy(deep=True)
    center_geom = region_gdf_cp['geometry'].to_list()[0].centroid
    
    fig = go.Figure()
    
    # Show stations and lines
    if transport == 'STC Metro':
        df_stations_metro_complete = df_stations_metro
        lines_unique = df_stations_metro_complete['linea'].unique()
        
        # Map boundaries of regions selected
        geometry_ = lineas_cdmx['geometry'].to_list()[0]
        lon_geom, lat_geom = geometry_.exterior.xy
        lon_geom = np.array(lon_geom).tolist()
        lat_geom = np.array(lat_geom).tolist()
        trace_boundary = go.Scattermapbox(
            lon=lon_geom,
            lat=lat_geom,
            mode='lines',
            fill='toself',
            fillcolor='rgba(196, 223, 225, 0.15)',
            line=dict(color='#65999d', width=2),
            hoverinfo='none',
        )
        fig.add_trace(trace_boundary)
        
        # Map all the lines
        for line in lines_unique:
            metro_lines_aux = metro_lines[metro_lines['LINEA'] == line[1:]]
            metro_lines_aux['geometry_yx'] = metro_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
            lines_ = metro_lines_aux['geometry_yx'].iloc[0]
            coords_pts = [[coord[0], coord[1]] for coord in lines_.coords]
            line_trace = go.Scattermapbox(
                mode='lines',
                lon = [coord[0] for coord in coords_pts],
                lat = [coord[1] for coord in coords_pts],
                line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                hoverinfo='none',
            )
            fig.add_trace(line_trace)
        
        # Map all stations
        lats = df_stations_metro_complete['latitud'].tolist()
        lons = df_stations_metro_complete['longitud'].tolist()
        ids = df_stations_metro_complete['cve_est'].tolist()
        lines = df_stations_metro_complete['linea'].tolist()
        names = df_stations_metro_complete['nombre'].tolist()
        
        scatter_trace = go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                marker=dict(
                    size=8,
                    color='black',
                    opacity = 1.0,
                ),
                hoverinfo='none',
        )
        scatter_trace_2 = go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                marker=dict(
                    size=4,
                    color='white',
                    opacity = 1.0,
                ),
                hovertext=ids,
                hoverlabel=dict(namelength=0),
                hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
        )
        fig.add_trace(scatter_trace)
        fig.add_trace(scatter_trace_2)

    else:
        df_stations_metrobus_complete = df_stations_metrobus
        lines_unique = df_stations_metrobus_complete['linea'].unique()
        
        # Map boundaries of regions selected
        geometry_ = lineas_cdmx['geometry'].to_list()[0]
        lon_geom, lat_geom = geometry_.exterior.xy
        lon_geom = np.array(lon_geom).tolist()
        lat_geom = np.array(lat_geom).tolist()
        trace_boundary = go.Scattermapbox(
            lon=lon_geom,
            lat=lat_geom,
            mode='lines',
            fill='toself',
            fillcolor='rgba(196, 223, 225, 0.15)',
            line=dict(color='#65999d', width=2),
            hoverinfo='none',
        )
        fig.add_trace(trace_boundary)
        
        # Map all lines
        for line in lines_unique:
            metrobus_lines_aux = mb_lines[mb_lines['LINEA'].str[-1] == line[1:]]
            metrobus_lines_aux['geometry_yx'] = metrobus_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
            lines_ = metrobus_lines_aux['geometry_yx']
            
            for i in range(len(lines_)):
                line_ = lines_.iloc[i]
                coords_pts = [[coord[0], coord[1]] for coord in line_.coords]
                line_trace = go.Scattermapbox(
                    mode='lines',
                    lon = [coord[0] for coord in coords_pts],
                    lat = [coord[1] for coord in coords_pts],
                    line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                    hoverinfo='none',
                )
                fig.add_trace(line_trace)

        # Map all the stations
        lats = df_stations_metrobus_complete['latitud'].tolist()
        lons = df_stations_metrobus_complete['longitud'].tolist()
        ids = df_stations_metrobus_complete['cve_est'].tolist()
        lines = df_stations_metrobus_complete['linea'].tolist()
        names = df_stations_metrobus_complete['nombre'].tolist()
        
        scatter_trace = go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
            marker=dict(
                size=8,
                color='black',
                opacity = 1.0,
            ),
            hoverinfo='none',
        )
        scatter_trace_2 = go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
            marker=dict(
                size=4,
                color='white',
                opacity = 1.0,
            ),
            hovertext=ids,
            hoverlabel=dict(namelength=0),
            hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
        )
        fig.add_trace(scatter_trace)
        fig.add_trace(scatter_trace_2)
    
    fig.update_layout(
        title_text='',
        margin=dict(t=0, l=0, r=0, b=0),
        legend=dict(
            title='',
            traceorder='normal',
            orientation='h',
            y=0,
            x=0,
            xanchor='center',
            yanchor='bottom',
            itemsizing='constant',
            itemwidth=30,
            bgcolor='rgba(255, 255, 255, 0)'
        ),
        legend_title=dict(side='top right'),
        showlegend=False,

        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=center_geom.y, lon=center_geom.x),
            zoom=10.2,
            pitch=0,
        ),
        height = 444,
        autosize=True,
        dragmode=False,
    )
    
    return fig

# Plot crime level risk map
def plot_predictive_map(datageom, transport: str, unique_values_metric):
    metric = 'valor'

    centroids = []
    for index, row in datageom.iterrows():
        multi_polygon = row['geometry']
        if isinstance(multi_polygon, MultiPolygon):
            for polygon in multi_polygon.geoms:
                if isinstance(polygon, Polygon):
                    centroids.append(polygon.centroid)
        elif isinstance(multi_polygon, Polygon):
            centroids.append(multi_polygon.centroid)

    center_geom = MultiPoint(centroids).centroid

    fig = go.Figure()

    colorscales = [
        ((0.0, '#ff5454'), (1.0, '#ff5454')),
        ((0.0, '#ffc862'), (1.0, '#ffc862')),
    ]
    colorborders = [
        '#b21800', '#FFA500',
    ]
    markerlinewidths = [
        3, 1
    ]

    if datageom[metric].value_counts().iloc[0] == len(datageom):
        dfp = datageom
        label_ = datageom[metric].unique()[0]
        varaux = 1 if label_ == 'Riesgo moderado' else 0
        fig.add_trace(
            go.Choroplethmapbox(
                    geojson=json.loads(dfp.to_json()), 
                    locations=dfp.index,
                    z=[varaux,] * len(dfp[metric]),
                    customdata=dfp['NOMGEO'],
                    colorscale=colorscales[varaux],
                    marker_opacity=0.8,
                    marker_line_width=markerlinewidths[varaux],
                    hoverlabel_bgcolor='white',
                    marker_line_color=colorborders[varaux],
                    hovertext=dfp[metric],
                    hoverlabel=dict(namelength=0),
                    hovertemplate = '<b>%{customdata}</b>: %{hovertext}',
                    showlegend=False,
                    showscale=False,
            )
        )
    else:
        for i, label_ in enumerate(unique_values_metric):
            dfp = datageom[datageom[metric] == label_]
            fig.add_trace(
                go.Choroplethmapbox(
                        geojson=json.loads(dfp.to_json()), 
                        locations=dfp.index,
                        z=[i,] * len(dfp[metric]),
                        customdata=dfp['NOMGEO'],
                        colorscale=colorscales[i],
                        marker_opacity=0.8,
                        marker_line_width=markerlinewidths[i],
                        hoverlabel_bgcolor='white',
                        marker_line_color=colorborders[i],
                        hovertext=dfp[metric],
                        hoverlabel=dict(namelength=0),
                        hovertemplate = '<b>%{customdata}</b>: %{hovertext}',
                        showlegend=False,
                        showscale=False,
                )
            )

    fig.update_layout(
        title_text='',
        margin=dict(t=0, l=0, r=0, b=0),
        title=dict(
            y=0.5,
            x=0.5,
            xanchor='center',
            yanchor='top',
        ),
        legend=dict(
            title='',
            traceorder='normal',
            orientation='h',
            y=0.5,
            x=0.5,
            xanchor='center',
            yanchor='top',
            itemsizing='constant',
            itemwidth=30,
            bgcolor='rgba(255, 255, 255, 0)'
        ),
        mapbox_style="carto-positron",
        height = 450,
        autosize=True,
        mapbox=dict(center=dict(lat=center_geom.y, lon=center_geom.x),zoom=9.0),
        dragmode=False
    )

    return fig

# Plot the view of the region selected with its stations inside
def plot_complementary_predictive_map(datageom, transport: str, region_gdf_aux: pd.DataFrame, region_column: str):
    geodata = load_geodata()
    df_stations_metro = geodata["df_stations_metro"]
    df_stations_metrobus = geodata["df_stations_metrobus"]
    metro_lines = geodata["metro_lines"]
    mb_lines = geodata["mb_lines"]

    region_gdf_cp = datageom.copy(deep=True)
    metric = 'valor_'
    cve_region_selected = region_gdf_aux[region_column].to_list()[0]
    region_gdf_cp.loc[region_gdf_cp['CVE_MUN'] == cve_region_selected, metric] = 1
    region_gdf_cp = region_gdf_cp.dropna()
    
    center_geom = region_gdf_cp['geometry'].to_list()[0].centroid
    
    fig = go.Figure()
    
    fig.update_layout(
        mapbox=dict(center=dict(lat=center_geom.y, lon=center_geom.x),zoom=9.8),dragmode=False
    )

    # Show stations and lines
    if transport == 'STC Metro':
        lines_unique = df_stations_metro['linea'].unique()
        
        # Mapping all the lines
        for line in lines_unique:
            metro_lines_aux = metro_lines[metro_lines['LINEA'] == line[1:]]
            metro_lines_aux['geometry_yx'] = metro_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
            lines_ = metro_lines_aux['geometry_yx'].iloc[0]
            coords_pts = [[coord[0], coord[1]] for coord in lines_.coords]
            line_trace = go.Scattermapbox(
                mode='lines',
                lon = [coord[0] for coord in coords_pts],
                lat = [coord[1] for coord in coords_pts],
                line=dict(color=LINESM[metro_lines_aux['LINEA'].to_list()[0]], width=4),
                hoverinfo='none',
            )
            
            fig.add_trace(line_trace)
                    
        
        # Filter just the stations within the selected region to mark it differently
        for line in lines_unique:
            if region_column == 'CVE_MUN':
                region_column_aux = 'cve_mun_inegi'
            else:
                region_column_aux = 'sector'
            
            # Stations within the region
            df_stations_metro_aux = df_stations_metro[(df_stations_metro['linea'] == line) & (df_stations_metro[region_column_aux] == cve_region_selected)]
            lats = df_stations_metro_aux['latitud']
            lons = df_stations_metro_aux['longitud']
            ids = df_stations_metro_aux['cve_est']
            lines = df_stations_metro_aux['linea']
            names = df_stations_metro_aux['nombre']
            scatter_trace = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=10,
                        color='black',
                    ),
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_2 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=6,
                        color='white',
                    ),
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>'
            )
            scatter_trace_3 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=3,
                        color=LINESM_aux[line],
                    ),
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)
            #fig.add_trace(scatter_trace_3)
            
            # Stations outside the region
            df_stations_metro_aux_out = df_stations_metro[(df_stations_metro['linea'] == line) & (df_stations_metro[region_column_aux] != cve_region_selected)]
            lats_out = df_stations_metro_aux_out['latitud']
            lons_out = df_stations_metro_aux_out['longitud']
            ids_out = df_stations_metro_aux_out['cve_est']
            lines_out = df_stations_metro_aux_out['linea']
            names_out = df_stations_metro_aux_out['nombre']
            scatter_trace_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker=dict(
                        size=6,
                        color='gray',
                    ),
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_2_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker=dict(
                        size=4,
                        color='white',
                    ),
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
            )
            scatter_trace_3_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker=dict(
                        size=3,
                        color=LINESM_aux[line],
                    ),
                    hovertemplate=''
            )
            fig.add_trace(scatter_trace_out)
            fig.add_trace(scatter_trace_2_out)
            #fig.add_trace(scatter_trace_3_out)

    else:
        lines_unique = df_stations_metrobus['linea'].unique()
        
        # Mapping all the lines
        for line in lines_unique:
            metrobus_lines_aux = mb_lines[mb_lines['LINEA'].str[-1] == line[1:]]
            metrobus_lines_aux['geometry_yx'] = metrobus_lines_aux['geometry'].apply(lambda line: LineString([(point[0], point[1]) for point in line.coords])) 
            lines_ = metrobus_lines_aux['geometry_yx']
            
            for i in range(len(lines_)):
                line_ = lines_.iloc[i]
                coords_pts = [[coord[0], coord[1]] for coord in line_.coords]
                line_trace = go.Scattermapbox(
                    mode='lines',
                    lon = [coord[0] for coord in coords_pts],
                    lat = [coord[1] for coord in coords_pts],
                    line=dict(color=LINESMB[metrobus_lines_aux['LINEA'].to_list()[0]], width=4),
                    hoverinfo='none',
                )
                
                fig.add_trace(line_trace)

        # Filter just the stations within the selected region to mark it differently
        for line in lines_unique:
            if region_column == 'CVE_MUN':
                region_column_aux = 'cve_mun_inegi'
            else:
                region_column_aux = 'sector'
            
            # Stations within the region
            df_stations_metrobus_aux = df_stations_metrobus[(df_stations_metrobus['linea'] == line) & (df_stations_metrobus[region_column_aux] == cve_region_selected)]
            lats = df_stations_metrobus_aux['latitud']
            lons = df_stations_metrobus_aux['longitud']
            ids = df_stations_metrobus_aux['cve_est']
            lines = df_stations_metrobus_aux['linea']
            names = df_stations_metrobus_aux['nombre']
            scatter_trace = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=10,
                        color='black',
                    ),
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_2 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=6,
                        color='white',
                    ),
                    hovertext=ids,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
            )
            scatter_trace_3 = go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids, lines, names)],
                    textposition='top center',
                    marker=dict(
                        size=3,
                        color=LINESM_aux[line],
                    ),
                    hovertemplate=''
            )
            fig.add_trace(scatter_trace)
            fig.add_trace(scatter_trace_2)
            #fig.add_trace(scatter_trace_3)
            
            
            # Stations outside the region
            df_stations_metrobus_aux_out = df_stations_metrobus[(df_stations_metrobus['linea'] == line) & (df_stations_metrobus[region_column_aux] != cve_region_selected)]
            lats_out = df_stations_metrobus_aux_out['latitud']
            lons_out = df_stations_metrobus_aux_out['longitud']
            ids_out = df_stations_metrobus_aux_out['cve_est']
            lines_out = df_stations_metrobus_aux_out['linea']
            names_out = df_stations_metrobus_aux_out['nombre']
            scatter_trace_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker=dict(
                        size=6,
                        color='gray',
                    ),
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate=''
            )
            scatter_trace_2_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker=dict(
                        size=4,
                        color='white',
                    ),
                    hovertext=ids_out,
                    hoverlabel=dict(namelength=0),
                    hovertemplate='%{customdata[2]} (%{customdata[1]})<br>',
            )
            scatter_trace_3_out = go.Scattermapbox(
                    lat=lats_out,
                    lon=lons_out,
                    mode='markers',
                    customdata=[[i, l, n] for i, l, n in zip(ids_out, lines_out, names_out)],
                    textposition='top center',
                    marker=dict(
                        size=3,
                        color=LINESM_aux[line],
                    ),
                    hovertemplate=''
            )
            fig.add_trace(scatter_trace_out)
            fig.add_trace(scatter_trace_2_out)
            #fig.add_trace(scatter_trace_3_out)
    
    fig.update_layout(
        title_text='',
        margin=dict(t=0, l=0, r=0, b=0),
        title=dict(
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
        ),
        legend=dict(
            title='',
            traceorder='normal',
            orientation='h',
            y=0.5,
            x=0.5,
            xanchor='center',
            yanchor='top',
            itemsizing='constant',
            itemwidth=30,
            bgcolor='rgba(255, 255, 255, 0)'
        ),
        legend_title=dict(side='top right'),
        showlegend=False,

        mapbox_style="carto-positron",
        mapbox=dict(
            pitch=20,
        ),
        height = 260,
        autosize=True,
        dragmode=False,
    )

    # Trace border of region selected
    geometry_ = region_gdf_cp['geometry'].iloc[0]
    lon_geom, lat_geom = geometry_.exterior.xy
    lon_geom = np.array(lon_geom).tolist()
    lat_geom = np.array(lat_geom).tolist()
    
    colorscales = [
        'rgba(255,84,84,0.3)',
        'rgba(255, 165, 0, 0.3)',

    ]
    colorborders = [
        '#b21800', '#FFA500',
    ]
    markerlinewidths = [
        2.5, 2.5
    ]

    if region_gdf_aux['valor'].to_list()[0] == 'Riesgo elevado':
        cs = colorscales[0]
        cb = colorborders[0]
        mklw = markerlinewidths[0]
    else:
        cs = colorscales[1]
        cb = colorborders[1]
        mklw = markerlinewidths[1]

    trace_boundary = go.Scattermapbox(
        lon=lon_geom,
        lat=lat_geom,
        mode='lines',
        fill='toself',
        fillcolor=cs,
        line=dict(color=cb, width=mklw),
        hoverinfo='none',
    )
    
    fig.add_trace(trace_boundary)
    
    return fig
