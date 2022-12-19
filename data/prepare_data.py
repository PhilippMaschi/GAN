""" This script will contain functions that help prepare the data"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def create_timestamp(df, date_column) -> pd.DataFrame:
    """creates a timestamp in the 'date' column so we can cluster the data more easily"""
    df.loc[:, date_column] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
    return df


def from_cat_to_num(df) -> pd.DataFrame:
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    return df


def normalize_single_load(load: pd.Series) -> pd.Series:
    """normalize the load by the peak load so load will be between 0 and 1"""
    # convert load to float
    load_values = load.astype(float)
    max_value = load.max()
    min_value = load.min()
    normalized = (load_values - min_value) / (max_value - min_value)
    return normalized


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


def extract_month_name_from_datetime(date: pd.Series) -> list:
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
    names = [month_dict[i] for i in date.dt.month]
    return names


def add_hour_of_the_day_to_df(df: pd.DataFrame, date: pd.Series) -> pd.DataFrame:
    """
    This function adds the hour of the day based on the timestamp column "Date"
    and stores it in a new column called "Hour of the Day".
    """
    df.loc[:, "Hour"] = date.dt.hour
    return df


def add_day_of_the_month_to_df(df: pd.DataFrame, date: pd.Series) -> pd.DataFrame:
    """ this function adds the hour of the day based on the timestamp column "date" """
    df.loc[:, "Day"] = date.dt.day
    return df


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


def define_float_type(df: pd.DataFrame) -> pd.DataFrame:
    """ the data is converted to flaot32 in order to save memory and increase computational speed"""
    df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
    return df


def determine_season(df: pd.DataFrame) -> list:
    season_names = {1: "winter",
                    2: "spring",
                    3: "summer",
                    4: "autumn"}

    season_list = list(df.Date.dt.month % 12 // 3 + 1)
    return [season_names[season] for season in season_list]


def crop_data_to_one_year(df: pd.DataFrame) -> pd.DataFrame:
    """ returns the data for exactly 8760 hours"""
    # determine start of data (first day where all hours are available:
    day_of_month = df.Date.dt.day.to_numpy()
    previous_cut_index = 0
    for i, day in enumerate(day_of_month):
        if i == 0:
            continue
        if day - day_of_month[i - 1] != 0:
            cut_index = i
            if cut_index - previous_cut_index != 24:
                break
            previous_cut_index = i

    # cut index represents the start of the dataframe, add 8760 hours:
    return df.iloc[cut_index:cut_index+8760, :].reset_index(drop=True)


def split_profiles_to_days(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """ returns 4 pandas dataframes with daily loads for each season (summer, winter, spring, autumn)"""
    daily_df_summer = pd.DataFrame(index=np.arange(24))
    daily_df_winter = pd.DataFrame(index=np.arange(24))
    daily_df_spring = pd.DataFrame(index=np.arange(24))
    daily_df_autumn = pd.DataFrame(index=np.arange(24))
    df.loc[:, "season"] = determine_season(df)
    day_of_month = df.Date.dt.day.to_numpy()
    previous_cut_index = 0
    for i, day in enumerate(day_of_month):
        if i == 0:
            continue
        if day - day_of_month[i - 1] != 0:
            # new day
            cut_index = i
            cutted_df = df.iloc[previous_cut_index:cut_index, :].reset_index(drop=True)
            if len(cutted_df) < 24:
                previous_cut_index = i
                continue  # dont save days where an hour is missing at the beginning or the end of the profile
            season = cutted_df.loc[0, "season"]
            if season == "summer":
                daily_df_summer = pd.concat([daily_df_summer, cutted_df.drop(columns=["Date", "season"])],
                                                 axis=1)
            elif season == "winter":
                daily_df_winter = pd.concat([daily_df_winter, cutted_df.drop(columns=["Date", "season"])],
                                                 axis=1)
            elif season == "spring":
                daily_df_spring = pd.concat([daily_df_spring, cutted_df.drop(columns=["Date", "season"])],
                                                 axis=1)
            elif season == "autumn":
                daily_df_autumn = pd.concat([daily_df_autumn, cutted_df.drop(columns=["Date", "season"])],
                                                 axis=1)
            else:
                assert False, f"season {season} does not exist"
            # update previous cut index
            previous_cut_index = i

    return daily_df_summer, daily_df_winter, daily_df_spring, daily_df_autumn


def split_profiles_to_seasons(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """ returns 4 pandas dataframes with seasonal loads for each season (summer, winter, spring, autumn)"""
    df.loc[:, "season"] = determine_season(df)
    winter = df.loc[df.loc[:, "season"] == "winter", :].reset_index(drop=True).drop(columns=["season"])
    spring = df.loc[df.loc[:, "season"] == "spring", :].reset_index(drop=True).drop(columns=["season"])
    summer = df.loc[df.loc[:, "season"] == "summer", :].reset_index(drop=True).drop(columns=["season"])
    autumn = df.loc[df.loc[:, "season"] == "autumn", :].reset_index(drop=True).drop(columns=["season"])
    return summer, winter, spring, autumn