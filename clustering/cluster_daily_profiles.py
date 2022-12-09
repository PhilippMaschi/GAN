import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from data.prepare_data import split_profiles_to_days, define_float_type
from config import Config


class CreateDailyData:
    """
    profiles will be split into daily load (vectors with 24 entries)
    every vector will be labeled with the season, the ID of the profile and...
    """

    def __init__(self):
        self.data_input = Config().path_2_data

    def plot_daily_loads(self, frame: pd.DataFrame):
        plt.style.use('seaborn')
        fig = plt.figure()
        for column in frame.columns:
            plt.plot(frame.index, frame.loc[:, column], color="blue", alpha=0.02)
        fig.show()




    def main(self):
        df = define_float_type(pd.read_json(self.data_input / "all_load_profiles.json"))
        summer, winter, spring, autumn = split_profiles_to_days(df)
        self.plot_daily_loads(summer)



if __name__ == "__main__":
    CreateDailyData().main()
