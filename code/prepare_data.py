import pandas as pd
from pathlib import Path
import glob
import os

class CleanDataframe:


    def create_timestamp(df, date_column) -> pd.DataFrame:
        """creates a timestamp in the 'date' column so we can cluster the data more easily"""
        df.loc[:, date_column] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h')
        return df

    def drop_missing_values(col_with_missing_val):
        data_no_missing_val = df.drop(columns=col_with_missing_val, axis=1)
        data_no_missing_val.to_csv('data_clean.csv', index=False)
        return data_no_missing_val

    def from_cat_to_num(df) -> pd.DataFrame:
        cat_columns = df.select_dtypes(['object']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
        return df

    def normalize_all_loads(df: pd.DataFrame) -> pd.DataFrame:
        """
        @rtype: return the dataframe with all loads normalized between 0 and 1
        @param df: dataframe with load profiles
        """
        columns2drop = ["Date"]  # define columns that are not normalized
        date = df.loc[:, "Date"]  # save date column
        normalized = MinMaxScaler().fit_transform(df.drop(columns=columns2drop))  # normalize data
        normalized_df = pd.DataFrame(normalized)  # to pd dataframe
        normalized_df.insert(loc=0, column="Date", value=date)  # insert the date column at first position
        normalized_df.columns = df.columns  # get the column names back
        return normalized_df

if __name__ == "__main__":
    path = path_file + "/csv"
    df = import_csv(path_csv=path_to_csv)









