import pandas as pd
import numpy as np


def add_missing_rows(df):  #adds missing timestamps of incomplete days
    dateRange = pd.date_range(start = '2021-01-01', end = '2022-01-02', freq = '15T')[:-1]
    df = df.set_index('Date/Time')
    df = df.reindex(dateRange)
    df.index.name = 'Date/Time'
    df = df.fillna(0)
    df = df.reset_index()
    return df


def keep_only_full_days(df):
    df = df.iloc[
        df[(df['Date/Time'].dt.hour == 0) & (df['Date/Time'].dt.minute == 0)].index[0]: \
        df[(df['Date/Time'].dt.hour == 23) & (df['Date/Time'].dt.minute == 45)].index[-1] + 1
    ]
    return df


def split_datetime_col(df):
    df['Date'] = pd.to_datetime(df['Date/Time']).dt.date
    df['Time'] = pd.to_datetime(df['Date/Time']).dt.time
    df = df.drop(columns = 'Date/Time')
    df = df.set_index(['Date', 'Time'])
    return df


def adjust_label_cols(df):
    col_dict = {item: idx for idx, item in enumerate(df.columns) if 'Electricity:Facility [J](TimeStep)' in item}
    df = df.rename(columns = col_dict)
    return df


def df_to_arr(df, dayCount = 366):
    arr = np.stack([df[col].to_numpy().reshape(dayCount, -1, 1) for col in df.columns], axis = 3).T
    return arr


def data_preparation_wrapper(path, folder, filename):
    df = pd.read_csv(path / folder / filename, parse_dates = ['Date/Time'])
    df = add_missing_rows(df)
    df = keep_only_full_days(df)
    df = split_datetime_col(df)
    df = adjust_label_cols(df)
    X_train = df_to_arr(df)
    return X_train


def arr_to_df(arr):
    df = pd.DataFrame(arr.T.reshape(-1, 100))
    return df