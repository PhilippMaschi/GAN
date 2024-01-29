from holidays import country_holidays
import pandas as pd
import numpy as np

#%% Dicts and lists

metSeason_dict = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4:"Spring",
    5:"Spring",
    6: 'Summer',
    7: 'Summer',
    8: 'Summer',
    9: 'Fall',
    10: 'Fall',
    11: 'Fall',
    12: "Winter",
}

weekend_dict = {
    0: False,
    1: False,
    2: False,
    3: False,
    4: False,
    5: True,
    6: True
}

holidays = country_holidays('ES', years = [2021, 2022]).keys()
newFeatures = [
    'date',
    'year',
    'meteorological season',
    'month',
    'month of the year',
    'week of the year',
    'day of the month',
    'weekday',
    'hour of the day',
    'day off',
    'weekend',
    'holiday'
]

#%% Functions
def import_and_preprocess_data(path):
    log_dict = {}
    df = pd.read_json(path) #import data from JSON file
    df.set_index('Date', inplace = True)    #set 'Date' column as index
    df = df.astype('float64')   #convert remaining columns to float64
    zeroConsumProfiles = df.columns[df.sum() == 0].values   #identify zero-consumption profiles
    log_dict['Zero-consumption profiles'] = zeroConsumProfiles.tolist()
    df.drop(columns = zeroConsumProfiles, inplace = True)   #remove zero-consumption profiles
    dupeTS_df = df[df.index.duplicated(keep = False)]   #identify duplicated timestamps
    log_dict['Duplicated timestamps'] = dupeTS_df.index.strftime('%Y-%m-%d %H:%M:%S').unique().tolist()
    df = df[~df.index.duplicated()] #remove duplicated timestamps (keep first occurance)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq = 'H'))  #reindex dataframe (sorts index and adds missing timestamps)
    missTS = df[df.isna().any(axis = 1)].index.strftime('%Y-%m-%d %H:%M:%S').tolist()   #identify missing timestamps
    log_dict['Missing timestamps'] = missTS
    df.fillna(method = 'ffill', inplace = True) #fill missing timestamps (propagate last valid observation forward)
    df.index.name = 'timestamp' #rename index
    df.reset_index(inplace = True)
    return df, log_dict


def create_and_add_datetime_features(df):
    profileCols = list(df.columns[1:])  #save profile columns for later
    df[newFeatures] = pd.DataFrame([    #create and add new columns/features
        df['timestamp'].dt.date,    #date
        df['timestamp'].dt.year,    #year
        df['timestamp'].dt.month.replace(metSeason_dict),   #meteorological season
        df['timestamp'].dt.month_name(),    #month
        df['timestamp'].dt.month,   #month of the year
        df['timestamp'].dt.isocalendar().week,  #week of the year
        df['timestamp'].dt.day, #day of the month
        df['timestamp'].dt.day_name(),  #weekday
        df['timestamp'].dt.hour + 1,    #hour of the day
        np.empty(len(df)),  #day off
        df['timestamp'].dt.dayofweek.replace(weekend_dict), #weekend
        df['timestamp'].dt.date.isin(holidays)  #holiday
    ]).T.values
    df['day off'] = (df['weekend'] + df['holiday']).astype(bool)    #fill column 'day off' (day off = weekend and/or holiday)
    df[['weekend', 'holiday', 'day off']] = df[['weekend', 'holiday', 'day off']].astype(int)   #convert Boolean values into integers
    df = df[['timestamp'] + newFeatures + profileCols]  #rearrange columns
    return df