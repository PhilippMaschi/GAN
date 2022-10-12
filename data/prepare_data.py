""" This script will contain functions that help prepare the data"""

import pandas as pd


class DataPrep:
    """ this class will consist of static methods so it will be easy to import in other scripts"""

    @staticmethod
    def create_timestamp(df, date_column) -> pd.DataFrame:
        """creates a timestamp in the 'date' column so we can cluster the data more easily"""
        df.loc[:, date_column] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h')
        return df

    @staticmethod
    def from_cat_to_num(df) -> pd.DataFrame:
        cat_columns = df.select_dtypes(['object']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
        return df

    @staticmethod
    def normalize_single_load(load: pd.Series) -> pd.Series:
        """normalize the load by the peak load so load will be between 0 and 1"""
        # convert load to float
        load_values = load.astype(float)
        max_value = load.max()
        min_value = load.min()
        normalized = (load_values - min_value) / (max_value - min_value)
        return normalized

    @staticmethod
    def normalize_all_loads(df: pd.DataFrame) -> pd.DataFrame:
        """

        @rtype: return the dataframe with all loads normalized between 0 and 1
        @param df: dataframe with load profiles

        """
        columns2normalize = [name for name in df.columns if "Wh" in name]
        for column_name in columns2normalize:
            load_values = df.loc[:, column_name].astype(float)
            max_value = load_values.max()
            min_value = load_values.min()
            # replace the column with normalized values
            df.loc[:, column_name] = (load_values - min_value) / (max_value - min_value)

        return df

    @staticmethod
    def extract_month_name_from_datetime(df: pd.DataFrame) -> list:
        """
        @param df: dataframe with a datetime index called "date"
        @return: list of month names based on the datetime index
        """
        month_dict = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December"
        }
        names = [month_dict[i] for i in df.loc[:, "date"].dt.month]
        return names

    @staticmethod
    def differentiate_positive_negative_loads(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        splits the dataframe into two frames, one having only the positive loads and one the negative
        @rtype: return 2 dataframes. First dataframe is the consumption, Second is the prosuming data
        """
        columns_positive = [name for name in df.columns if "+" in name]
        columns_negative = [name for name in df.columns if "-" in name]

        return df.drop(columns=columns_negative), df.drop(columns=columns_positive)

    @staticmethod
    def add_hour_of_the_day_to_df(df: pd.DataFrame) -> pd.DataFrame:
        """ this function adds the hour of the day based on the timestamp column "date" """
        df.loc[:, "hour"] = df.loc[:, "date"].dt.hour
        return df

    @staticmethod
    def add_day_of_the_month_to_df(df: pd.DataFrame) -> pd.DataFrame:
        """ this function adds the hour of the day based on the timestamp column "date" """
        df.loc[:, "day"] = df.loc[:, "date"].dt.day
        return df

    @staticmethod
    def sort_columns_months(df: pd.DataFrame) -> pd.DataFrame:
        """ sorts the columns of a dataframe with month names as columns"""
        column_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ]
        df = df[column_names]
        return df


