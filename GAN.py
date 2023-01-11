import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch as t
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType, Normalization
#from import_data import data
class Doppelgan:

    def __init__(self, filename, outputdir, load_profile_carrier=None):
        self.df = pd.read_parquet(filename)
        self.df = self.df[["Date", "Consumed energy [Wh]"]]

        self.report_path = f"{outputdir}/results/EDA reports"
        self.figure_path = f"{outputdir}/results/figures"
        self.title = load_profile_carrier
        print(self.df.describe())

    def plot_distribution (self):
        fig = plt.figure(figsize=(8, 4))
        plt.plot(self.df["Date"], self.df["Consumed energy [Wh]"])
        plt.xticks(rotation=90)
        plt.ylabel("Consumed energy")
        plt.xlabel("Date")
        plt.title(self.title)
        plt.savefig(self.figure_path+"/data_GANs_distribution.jpg")
        return plt.show()


if __name__ == "__main__":

    data = Doppelgan(filename = r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop//ENERCOOP_load_profiles.parquet.gzip",
                     outputdir=r"C:/Users/FrancescaConselvan/Documents/MODERATE_GAN",
                     load_profile_carrier="Enercoop")

    data.plot_distribution()