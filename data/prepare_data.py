""" This script will contain functions that help prepare the data"""

import pandas as pd



def create_timestamp(df, date_column) -> pd.DataFrame:
    """creates a timestamp in the 'date' column so we can cluster the data more easily"""
    df.loc[:, date_column] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h')
    return df


# function to convert categorical variables in numerical. Necessary ???
def from_cat_to_num(df) -> pd.DataFrame:
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    return df


def normalize_load(load: pd.Series) -> pd.Series:
    """normalize the load by the peak load so load will be between 0 and 1"""
    # convert load to float
    load_values = load.astype(float)
    max_value = load.max()
    min_value = load.min()
    normalized = (load_values - min_value) / (max_value - min_value)
    return normalized


