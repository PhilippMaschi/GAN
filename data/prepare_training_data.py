import numpy
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

from config import Config
from prepare_data import define_float_type, extract_month_name_from_datetime, determine_season, crop_data_to_one_year


@dataclass
class DailyProfile:
    vector: numpy.array
    id: int
    season: str
    number_of_local_peaks: int = field(init=False)
    normalized_vector: np.array = field(init=False)

    def __post_init__(self) -> None:
        self.normalized_vector = MinMaxScaler().fit_transform(self.vector.reshape(-1, 1)).flatten()
        self.number_of_local_peaks = self.count_local_maxima()

    def count_local_maxima(self) -> int:
        hours_of_peaks = find_peaks(self.normalized_vector, distance=3)[0]
        number_of_peaks = len(hours_of_peaks)
        return number_of_peaks







class CreateDailyData:
    """
    profiles will be split into daily load (vectors with 24 entries)
    every vector will be labeled with the season, the ID of the profile and...
    """
    def __init__(self):
        self.data_input = Config().path_2_data
        self.daily_df_summer = pd.DataFrame(index=np.arange(24))
        self.daily_df_winter = pd.DataFrame(index=np.arange(24))
        self.daily_df_spring = pd.DataFrame(index=np.arange(24))
        self.daily_df_autumn = pd.DataFrame(index=np.arange(24))


    def split_profiles_to_days(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """ returns 4 pandas dataframes with daily loads for each season (summer, winter, spring, autumn)"""
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
                    self.daily_df_summer = pd.concat([self.daily_df_summer, cutted_df.drop(columns=["Date", "season"])],
                                                     axis=1)
                elif season == "winter":
                    self.daily_df_winter = pd.concat([self.daily_df_winter, cutted_df.drop(columns=["Date", "season"])],
                                                     axis=1)
                elif season == "spring":
                    self.daily_df_spring = pd.concat([self.daily_df_spring, cutted_df.drop(columns=["Date", "season"])],
                                                     axis=1)
                elif season == "autumn":
                    self.daily_df_autumn = pd.concat([self.daily_df_autumn, cutted_df.drop(columns=["Date", "season"])],
                                                     axis=1)
                else:
                    assert False, f"season {season} does not exist"
                # update previous cut index
                previous_cut_index = i

        return self.daily_df_summer, self.daily_df_winter, self.daily_df_spring, self.daily_df_autumn


    def split_profiles_to_seasons(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """ returns 4 pandas dataframes with seasonal loads for each season (summer, winter, spring, autumn)"""
        df.loc[:, "season"] = determine_season(df)
        winter = df.loc[df.loc[:, "season"] == "winter", :].reset_index(drop=True).drop(columns=["season"])
        spring = df.loc[df.loc[:, "season"] == "spring", :].reset_index(drop=True).drop(columns=["season"])
        summer = df.loc[df.loc[:, "season"] == "summer", :].reset_index(drop=True).drop(columns=["season"])
        autumn = df.loc[df.loc[:, "season"] == "autumn", :].reset_index(drop=True).drop(columns=["season"])
        return summer, winter, spring, autumn






    def main(self):
        df = define_float_type(pd.read_json(self.data_input / "all_load_profiles.json"))
        df = crop_data_to_one_year(df)
        self.split_profiles_to_seasons(df)





if __name__ == "__main__":
    CreateDailyData().main()

    vec = np.sin(np.arange(24))
    ID = 10
    season = "winter"
    number_peaks = 2

    label_class = DailyProfile(vector=vec, id=ID, season=season)
    print(label_class)
