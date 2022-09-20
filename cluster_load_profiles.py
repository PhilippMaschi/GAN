from typing import List

import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns


class LoadProfileLoader:
    def __init__(self):
        self.path_2_files = r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\MODERATE\data\ENERCOOP"

    def get_csv_names(self) -> List[str]:
        csv_names = os.listdir(self.path_2_files)
        return csv_names

    def normalize_load(self, load: pd.Series) -> pd.Series:
        """normalize the load by the peak load so load will be between 0 and 1"""
        # convert load to float
        load_values = load.astype(float)
        max_value = load.max()
        normalized = load_values / max_value
        return normalized

    # TODO check for outliers
    def check_for_outliers(self):
        pass

    def read_load_profiles(self, csv_names: List[str]) -> pd.DataFrame:
        big_table = pd.DataFrame(columns=["date"])
        for i, name in enumerate(csv_names):
            file = Path(self.path_2_files) / Path(name)
            load = pd.read_csv(file, sep=";", decimal=",").loc[:, ["FECHA(YYYY-MM-DD o DD/MM/YYY)", "HORA(h)", "A+(Wh)"]]
            load.columns = ["date", "hour", "load"]  # rename the columns
            # normalize the load
            big_table.loc[:, f"load_{name}"] = self.normalize_load(load.load)
            time_series = self.create_timestamp(load.loc[:, "date"], load.loc[:, "hour"])
            if i == 0:
                big_table.loc[:, "date"] = time_series
            else:  # check if the data has the same dates
                assert big_table.loc[:, "date"].all() == time_series.all(), "time data does not match"

        return big_table

    def create_timestamp(self, date: pd.Series, hour: pd.Series) -> pd.DataFrame:
        """creates a timestamp in the 'date' column so we can cluster the data more easily"""
        return pd.to_datetime(date) + pd.to_timedelta(hour, unit='h')


    def main(self) -> pd.DataFrame:
        all_profiles = self.read_load_profiles(self.get_csv_names())
        return all_profiles


class Visualization:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_daily_loads(self):
        """plits the dataframe data into daly load pieces"""

        pass

    def lineplot_all_loads(self):
        melted_df = self.df.melt(id_vars="date")
        sns.lineplot(data=melted_df, x="date", y="value", hue="variable")
        plt.show()
        pass


    def main(self):
        self.lineplot_all_loads()





if __name__ == "__main__":
    profiles = LoadProfileLoader().main()
    Visualization(profiles).main()
