from typing import List

import pandas as pd
import numpy as np
from pathlib import Path
import os


class LoadProfileLoader:
    def __init__(self):
        self.path_2_files = r'/Users/francesca/Desktop/e-think/MODERATE/datasets/Enercoop'

    def get_csv_names(self) -> List[str]:
        csv_names = os.listdir(self.path_2_files)
        return csv_names

    def read_load_profiles(self, csv_names: List[str]) -> pd.DataFrame:
        big_table = pd.DataFrame(columns=["date", "hour", "load"])
        for name in csv_names:
            file = Path(self.path_2_files) / Path(name)
            load = pd.read_csv(file, sep=";", decimal=",").loc[:, ["FECHA(YYYY-MM-DD o DD/MM/YYY)", "HORA(h)", "A+(Wh)"]]
            load.columns = ["date", "hour", "load"]  # rename the columns
            # normalize the load
            load.loc[:, "load"] = self.normalize_load(load.load)
            # add all the tables to one big dataframe
            big_table = pd.concat([big_table, load])

        return big_table

    def create_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """creates a timestamp in the 'date' column so we can cluster the data more easily"""
        df.loc[:, "date"] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h')
        return df

    def normalize_load(self, load: pd.Series) -> pd.Series:
        """normalize the load by the peak load so load will be between 0 and 1"""
        # convert load to float
        load_values = load.astype(float)
        max_value = load.max()
        normalized = load_values / max_value
        return normalized






    def main(self):
        all_profiles = self.read_load_profiles(self.get_csv_names())
        df = self.create_timestamp(all_profiles)



if __name__ == "__main__":
    LoadProfileLoader().main()
