from holidays import country_holidays
import pandas as pd
import numpy as np

from dictionaries import metSeason_dict, weekend_dict

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


def create_and_add_datetime_features(df):
    # Save profile columns for later
    profileCols = list(df.columns[1:])

    # Create and add new columns/features
    df[newFeatures] = pd.DataFrame([
        df['timestamp'].dt.date,   #date
        df['timestamp'].dt.year,   #year
        df['timestamp'].dt.month.replace(metSeason_dict),  #meteorological season
        df['timestamp'].dt.month_name(),   #month
        df['timestamp'].dt.month,  #month of the year
        df['timestamp'].dt.isocalendar().week, #week of the year
        df['timestamp'].dt.day,    #day of the month
        df['timestamp'].dt.day_name(), #weekday
        df['timestamp'].dt.hour + 1,   #hour of the day
        np.empty(len(df)), #day off
        df['timestamp'].dt.dayofweek.replace(weekend_dict),    #weekend
        df['timestamp'].dt.date.isin(holidays) #holiday
    ]).T.values
    
    # Fill column 'day off' (day off = weekend and/or holiday)
    df['day off'] = (df['weekend'] + df['holiday']).astype(bool)

    # Convert Boolean values into integers
    df[['weekend', 'holiday', 'day off']] = df[['weekend', 'holiday', 'day off']].astype(int)

    # Rearrange columns
    df = df[['timestamp'] + newFeatures + profileCols]

    # Return extended dataframe
    return df