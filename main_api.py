import json
import pandas as pd
from math import ceil
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Creating instance of api flask
app = Flask(__name__)


# Crime models
def load_models_once(func):
    def wrapper(*args, **kwargs):
        if not hasattr(app, 'crime_metro_model_1'):
            with open('./models_trained/final/clf_crime_metro_dataset_{}_wm_2_mas_perc.pkl'.format(3), 'rb') as file:
                app.crime_metro_model_1 = pickle.load(file)
        if not hasattr(app, 'crime_metro_model_2'):
            with open('./models_trained/final/clf_crime_metro_dataset_{}_wm_2_mas_perc.pkl'.format(4), 'rb') as file:
                app.crime_metro_model_2 = pickle.load(file)
        if not hasattr(app, 'crime_metrobus_model_1'):
            with open('./models_trained/final/clf_crime_metrobus_dataset_{}_wm_2_mas_perc.pkl'.format(3), 'rb') as file:
                app.crime_metrobus_model_1 = pickle.load(file)
        if not hasattr(app, 'crime_metrobus_model_2'):
            with open('./models_trained/final/clf_crime_metrobus_dataset_{}_wm_2_mas_perc.pkl'.format(4), 'rb') as file:
                app.crime_metrobus_model_2 = pickle.load(file)
            
        return func(*args, **kwargs)

    # Renaming the function name:
    wrapper.__name__ = func.__name__
    return wrapper

# Load of affluence forecasting values
def load_afflu_forecast_munic_values(transport: str):
    if transport == 'STC Metro':
        df = pd.read_csv('./predictions_sarima/predicciones_afluencia_alcaldia_semana_metro.csv')
    else:
        df = pd.read_csv('./predictions_sarima/predicciones_afluencia_alcaldia_semana_metrobus_final.csv')

    return df

# Get number of the week of month given a date
def week_of_month(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))

# Get the monday for a certain week of the year (auxiliary to show the range of a certain week)
def get_monday_week_year(week, year):
    first_day_year = datetime(year, 1, 1)
    monday_first_week = first_day_year - timedelta(days=first_day_year.weekday())
    return monday_first_week + timedelta(weeks=week - 1)

# Validate if the text is a number
def validate_number(text):
    try:
        num = float(text)
        num_int = int(num)
        
        return num_int
    
    except ValueError:
        return -1

@app.route('/crime_model')
@load_models_once
def predict_with_crime_model():
    # Read params    
    transport = request.args.get('transport')
    year = request.args.get('year')
    week_year_to_predict = request.args.get('week_year_to_predict')
    categ_crime = request.args.get('categ_crime')
    sex = request.args.get('sex')
    id_sex = 0
    
    transport_none = False
    year_none = False
    week_year_to_predict_none = False
    categ_crime_none = False
    sex_none = False

    # Verify if param values were given, and also if they are valid
    if transport is None:
        print("Parameter 'transport' is missing")
        transport_none = True
        return jsonify({'Error': 'No se ingresó el parámetro "transport"'})
    else:
        if not transport in ['STC Metro', 'Metrobús']:
            return jsonify({'Error': 'No se ingresó un parámetro válido para la variable "transport"'})
        
    if year is None:
        print("Parameter 'year' is missing")
        year_none = True
        return jsonify({'Error': 'No se ingresó el parámetro "year"'})
    else:
        year = validate_number(year)
        if year == -1 or not year in [2024, ]:
            return jsonify({'Error': 'No se ingresó un parámetro válido para la variable "year"'})
        
    if week_year_to_predict is None:
        print("Parameter 'week_year_to_predict' is missing")
        week_year_to_predict_none = True
        return jsonify({'Error': 'No se ingresó el parámetro "week_year_to_predict"'})
    else:
        week_year_to_predict = validate_number(week_year_to_predict)
        if week_year_to_predict == -1 or not week_year_to_predict in [i + 1 for i in range(53)]:
            return jsonify({'Error': 'No se ingresó un parámetro válido para la variable "week_year_to_predict"'})
        
        # Adjust just to not load the real affluence from last week of last year (2023)
        if week_year_to_predict == 1:
            week_year_to_predict = 2
    
    if categ_crime is None:
        print("Parameter 'categ_crime' is missing")
        categ_crime_none = True
        return jsonify({'Error': 'No se ingresó el parámetro "categ_crime"'})
    else:
        if not categ_crime in ['Robo a transeúnte y pasajero en transporte público', 'Robo de vehículo y autopartes', 'Delitos sexuales', 'Lesiones', 'Amenazas', 'Fraude', ]:
            return jsonify({'Error': 'No se ingresó un parámetro válido para la variable "categ_crime"'})
        
    if sex is None:
        print("Parameter 'sex' is missing")
        sex_none = True
    else:
        if not sex in ['Masculino', 'Femenino']:
            return jsonify({'Error': 'No se ingresó un parámetro válido para la variable "sex"'})
        else:
            id_sex = 0 if sex == 'Masculino' else 1
        
    week_of_month_ = week_of_month(get_monday_week_year(week_year_to_predict, year))
    
    inputs_model_ls = []
    afflu_fc_values = load_afflu_forecast_munic_values(transport)
    if transport == 'STC Metro':
        regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tláhuac', 'Venustiano Carranza', 'Álvaro Obregón']
        if sex_none == True:
            columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'semana_1',]
            for region in regions:
                inputs_model_ls.append([week_of_month_, region, categ_crime])
            crime_model = app.crime_metro_model_1
            
        else:
            columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'sexo_victima', 'semana_1',]
            for region in regions:
                inputs_model_ls.append([week_of_month_, region, categ_crime, id_sex, ])
            crime_model = app.crime_metro_model_2
            
    else:
        regions = ['Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
                    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
                    'Tlalpan', 'Venustiano Carranza', 'Álvaro Obregón', 'Xochimilco']
        
        if sex_none == True:
            columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'semana_1',]
            for region in regions:
                inputs_model_ls.append([week_of_month_, region, categ_crime])
            crime_model = app.crime_metrobus_model_1
            
        else:
            columns_input_model = ['semana_mes', 'alcaldia', 'categoria_delito_adaptada', 'sexo_victima', 'semana_1',]
            for region in regions:
                inputs_model_ls.append([week_of_month_, region, categ_crime, id_sex, ])
            crime_model = app.crime_metrobus_model_2
    
    
    input_model_df_partial = pd.DataFrame(inputs_model_ls, columns=columns_input_model[:-1])
    
    # Retake the predictions from SARIMA models
    afflu_fc_values_filtered = afflu_fc_values[afflu_fc_values['semana_anio'] == week_year_to_predict - 1]
    input_model_df = input_model_df_partial.merge(afflu_fc_values_filtered, left_on=columns_input_model[1], right_on=['region'])
    input_model_df.rename(columns={'afluencia': 'semana_1'}, inplace=True)
    input_model_df = input_model_df[columns_input_model]
    
    # Predictions
    preds = crime_model.predict(input_model_df)
    df_preds = pd.DataFrame(columns=['prediccion'])
    df_preds['prediccion'] = preds
    df_preds['prediccion'] = df_preds['prediccion'].replace({'High': 'Riesgo elevado', 'Low': 'Riesgo moderado'})
    input_model_df.rename(columns={'semana_1': 'afluencia_total_predicha_semana_pasada',}, inplace=True)
    input_model_df['semana_anio'] = week_year_to_predict
    input_model_df.drop(columns=['sexo_victima'], inplace=True)
    input_model_df['sexo_victima'] = sex
    input_model_df_preds = pd.concat([input_model_df, df_preds], axis=1)
    result = input_model_df_preds.to_dict(orient='records')
    response = jsonify(result)
    
    return response

if __name__ == '__main__':
    app.run()