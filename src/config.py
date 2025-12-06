from datetime import datetime, timedelta
from math import ceil

# Dictionaries to fix values
dict_weekday = {
    0: 'Lunes',
    1: 'Martes',
    2: 'Miércoles',
    3: 'Jueves',
    4: 'Viernes',
    5: 'Sábado',
    6: 'Domingo',
}

dict_munics = {
    'AZCAPOTZALCO': 'Azcapotzalco',
    'COYOACAN': 'Coyoacán',
    'CUAJIMALPA DE MORELOS': 'Cuajimalpa de Morelos',
    'GUSTAVO A. MADERO': 'Gustavo A. Madero',
    'IZTACALCO': 'Iztacalco',
    'IZTAPALAPA': 'Iztapalapa',
    'MAGDALENA CONTRERAS': 'Magdalena Contreras',
    'MILPA ALTA': 'Milpa Alta',
    'ALVARO OBREGON': 'Álvaro Obregón',
    'TLAHUAC': 'Tláhuac',
    'TLALPAN': 'Tlalpan',
    'XOCHIMILCO': 'Xochimilco',
    'BENITO JUAREZ': 'Benito Juárez',
    'CUAUHTEMOC': 'Cuauhtémoc',
    'MIGUEL HIDALGO': 'Miguel Hidalgo',
    'VENUSTIANO CARRANZA': 'Venustiano Carranza',
}

month_names = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
}

# List of default regions and lines
zones_ls = ["Centro", "Norte", "Sur", "Oriente", "Poniente"]
munics_ls = {
    'STC Metro': ['Álvaro Obregón', 'Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
    'Tláhuac', 'Venustiano Carranza'],
    'Metrobús': ['Álvaro Obregón', 'Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuauhtémoc',
    'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa', 'Miguel Hidalgo',
    'Tlalpan', 'Venustiano Carranza', 'Xochimilco'],
}
lines_ls = {
    'STC Metro': ['Línea 1', 'Línea 2', 'Línea 3', 'Línea 4', 'Línea 5', 'Línea 6', 'Línea 7', 'Línea 8',
                'Línea 9', 'Línea 12', 'Línea A', 'Línea B',],
    'Metrobús': ['Línea 1', 'Línea 2', 'Línea 3', 'Línea 4', 'Línea 5', 'Línea 6', 'Línea 7',],
}

years_queries_ls = [2019, 2020, 2021, 2022, 2023]
months_queries_ls = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
weekdays_queries_ls = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
crime_classes_queries_ls = [
    'Robo a transeúnte en vía pública',
    'Robo a transeúnte en espacio abierto al público',
    'Robo en transporte público individual',
    'Robo en transporte público colectivo',
    'Robo en transporte individual',
    'Robo a persona en un lugar privado',
    'Robo simple',
    'Robo de vehículo',
    'Robo de autopartes',
    'Robo a institución bancaria',
    'Robo a negocio', 
    'Amenazas',
    'Fraude',
    'Extorsión',
    'Abuso sexual',
    'Acoso sexual',
    'Violación simple',
    'Violación equiparada',
    'Otro tipo de violación',
    'Estupro',
    'Otros delitos que atentan contra la libertad y la seguridad sexual',
    'Homicidio',
    'Feminicidio',
    'Lesiones',
]

crime_vars_queries_ls = [
    'Robo',
    'Amenazas',
    'Fraude',
    'Extorsión',
    'Abuso sexual',
    'Acoso sexual',
    'Violación',
    'Homicidio',
    'Lesiones',
]

# Dictionaries to load point images of stations
POINTSM = {
    'L1': './assets/images/circulos/STCMetro_L1.png',
    'L2': './assets/images/circulos/STCMetro_L2.png',
    'L3': './assets/images/circulos/STCMetro_L3.png',
    'L4': './assets/images/circulos/STCMetro_L4.png',
    'L5': './assets/images/circulos/STCMetro_L5.png',
    'L6': './assets/images/circulos/STCMetro_L6.png',
    'L7': './assets/images/circulos/STCMetro_L7.png',
    'L8': './assets/images/circulos/STCMetro_L8.png',
    'L9': './assets/images/circulos/STCMetro_L9.png',
    'LA': './assets/images/circulos/STCMetro_LA.png',
    'LB': './assets/images/circulos/STCMetro_LB.png',
    'L12': './assets/images/circulos/STCMetro_L12.png',
}

POINTSMB = {
    'L1': './assets/images/circulos/MB_L1.png',
    'L2': './assets/images/circulos/MB_L2.png',
    'L3': './assets/images/circulos/MB_L3.png',
    'L4': './assets/images/circulos/MB_L4.png',
    'L5': './assets/images/circulos/MB_L5.png',
    'L6': './assets/images/circulos/MB_L6.png',
    'L7': './assets/images/circulos/MB_L7.png',
}

# Helper functions
def week_of_month(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))

def get_monday_week_year(week, year):
    first_day_year = datetime(year, 1, 1)
    monday_first_week = first_day_year - timedelta(days=first_day_year.weekday())
    return monday_first_week + timedelta(weeks=week - 1)

def get_week_date_range(week_number, year):
    start_date = datetime(year, 1, 1)
    days_offset = (7 - start_date.weekday()) % 7
    start_date += timedelta(days=days_offset)
    start_week_date = start_date + timedelta(weeks=week_number - 1)
    end_week_date = start_week_date + timedelta(days=6)
    start_date_str = f"{start_week_date.day} de {month_names[start_week_date.month]} de {start_week_date.year}"
    end_date_str = f"{end_week_date.day} de {month_names[end_week_date.month]} de {end_week_date.year}"
    
    return f"{start_date_str} al {end_date_str}"

def get_station(df, cve_est, column=None):
    if column is None:
        return df[df['cve_est'] == cve_est]
    filter = df[df['cve_est'] == cve_est]
    # print(filter) # Removed print for cleaner output
    return filter[column].to_list()[0]

# Datetime values
today_ = datetime.now()
weekday = dict_weekday[today_.weekday()]
week_year = today_.strftime("%W")
month = today_.month
year = today_.year
last_day_of_year = datetime(year, 12, 31)
last_week_of_year = int(last_day_of_year.strftime("%W"))
week_month = week_of_month(today_)

def get_current_date_info():
    return {
        "today": today_,
        "weekday": weekday,
        "week_year": week_year,
        "month": month,
        "year": year,
        "last_week_of_year": last_week_of_year,
        "week_month": week_month
    }
