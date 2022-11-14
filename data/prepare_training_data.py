import numpy
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

from config import Config
from prepare_data import define_float_type, extract_month_name_from_datetime, determine_season


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


    def split_profiles_to_days(self, df: pd.DataFrame):
        df.loc[:, "season"] = determine_season(df)
        day_of_month = df.Date.dt.day.to_numpy()

        for i, day in enumerate(day_of_month):
            if i == 0:
                continue
            if day - day_of_month(i - 1) != 0:


        pass

    def determine_season_of_day(self):


        pass

    def main(self):
        df = define_float_type(pd.read_json(self.data_input / "all_load_profiles.json"))

        self.split_profiles_to_days(df)



if __name__ == "__main__":
    CreateDailyData().main()

    vec = np.sin(np.arange(24))
    ID = 10
    season = "winter"
    number_peaks = 2

    label_class = Label(vector=vec, id=ID, season=season)
    print(label_class)
