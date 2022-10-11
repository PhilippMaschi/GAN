from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

    def __init__(self, path):
        self.path_2_files = path  # I added the argument path so everyone can use it

    def get_csv_names(self) -> List[str]:
        csv_names = os.listdir(self.path_2_files)
        return csv_names

    def read_load_profiles(self, csv_names: List[str], columns_name=[]) -> pd.DataFrame:
        big_table = pd.DataFrame(columns=columns_name)
        for name in csv_names:
            file = Path(self.path_2_files) / Path(name)
            load = pd.read_csv(file, sep=";", decimal=",")  # we should import the whole dataset and then  select
            # what we want after the EDA
            load.columns = columns_name

            # normalize the load > I would not do it here for two reasons:
            # 1. the function should do only one action
            # 2. problem of the outliers
            # load.loc[:, "load"] = self.normalize_load(load.load)

            # add all the tables to one big dataframe
            big_table = pd.concat([big_table, load])

        return big_table

    def create_timestamp(self, date: pd.Series, hour: pd.Series) -> pd.DataFrame:
        """creates a timestamp in the 'date' column so we can cluster the data more easily"""
        df.loc[:, "date"] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h')
        return df

    # function to convert categorical variables in numerical. Necessary??? It is necessary in the EDA
    def from_cat_to_num(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_columns = df.select_dtypes(['object']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
        return df

    # def normalize_load(self, load: pd.Series) -> pd.Series:
    #    """normalize the load by the peak load so load will be between 0 and 1"""
    #    # convert load to float
    #    load_values = load.astype(float)
    #    max_value = load.max()
    #    normalized = load_values / max_value
    #    return normalized
        return pd.to_datetime(date) + pd.to_timedelta(hour, unit='h')

    # I would add it later, after the EDA because we have the problem of the outliers

    def main(self):
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
