import pandas as pd
from dicts import metSeason_dict, weekend_dict

def import_and_preprocess_enercoop_load_profiles(path):
    df = pd.read_json(path)
    df = pd.melt(df, id_vars = 'Date', value_vars = df.columns[1:].tolist(), var_name = 'Profile', value_name = 'Consumed energy [Wh]')
    df['Profile'] = df['Profile'].astype(int)
    df['Date'] = pd.to_datetime(df['Date'])
    df[[
            'Year',                 'Meteorological season',                        'Month',                        'Week of the year',
            'Day of the month',     'Weekday',                                      'Hour of the day',              'Weekend'
    ]] = \
        pd.DataFrame([
            df['Date'].dt.year,     df['Date'].dt.month.replace(metSeason_dict),    df['Date'].dt.month_name(),     df['Date'].dt.isocalendar().week,
            df['Date'].dt.day,      df['Date'].dt.day_name(),                       df['Date'].dt.hour + 1,         df['Date'].dt.dayofweek.replace(weekend_dict)
        ]).T.values
    df = df[['Profile', 'Date'] + df.columns[3:].tolist() + ['Consumed energy [Wh]']]
    intCols = ['Profile', 'Year', 'Week of the year', 'Day of the month', 'Hour of the day']
    df[intCols] = df[intCols].astype(int)
    df['Weekend'] = df['Weekend'].astype('bool')
    return df