import pandas as pd
from pathlib import Path
import glob
import os

class clean_dataframe:

    def __init__(self, path_to_csv):
        path = path_to_csv

    def concatenate_csv(path):

        files = Path(path).glob("*.csv")
        dfs = [pd.read_csv(f, delimiter=";", low_memory=False) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(path+".csv")
        return  df.info()

    def columns_name ():

    def create_timestamp(df, date_column) -> pd.DataFrame:
        """creates a timestamp in the 'date' column so we can cluster the data more easily"""
        df.loc[:, date_column] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h')
        return df

    def drop_missing_values(col_with_missing_val):
        data_no_missing_val = data.drop(columns=col_with_missing_val, axis=1)
        data_no_missing_val.to_csv('data_clean.csv', index=False)
        return data_no_missing_val

    def from_cat_to_num(df) -> pd.DataFrame:
        cat_columns = df.select_dtypes(['object']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
        return df

if __name__ == "__main__":
    path = path_file + "/csv"
    df = import_csv(path_csv=path_to_csv)









