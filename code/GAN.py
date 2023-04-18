import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch as t
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType
import multiprocessing as mp


class Doppelgan:

    def __init__(self, filename, outputdir, output_filename, load_profile_carrier=None):
        self.df = pd.read_parquet(filename)
        self.df = self.df[["Date", "Consumed energy [Wh]"]]
        self.df.Date = pd.to_datetime(self.df.Date)
        self.features = self.features_GAN()

        self.report_path = f"{outputdir}/synthetic_data"
        self.output_file = output_filename
        self.figure_path = f"{outputdir}/DoppelGANger"
        self.title = load_profile_carrier

        print(self.df.describe())

    def plot_distribution(self):
        fig = plt.figure(figsize=(8, 4))
        plt.plot(self.df["Date"], self.df["Consumed energy [Wh]"])
        plt.xticks(rotation=45)
        plt.ylabel("Consumed energy")
        plt.xlabel("Date")
        plt.title(self.title)
        plt.savefig(self.figure_path + "/data_GANs_distribution.jpg")
        return plt.show()

    def features_GAN(self):
        self.features = self.df.drop(columns=["Date"]).to_numpy()
        n = self.features.shape[0] // 24
        self.features = self.features[:(n * 24), :].reshape(-1, 24, self.features.shape[1])

        """saving training data to csv"""
        train_df = self.df.loc[0:len(self.features),]
        train_df.to_csv("train_data.csv")

        print(self.features.shape)
        return self.features.shape

    def train_DGAN(self):
        """train DGAN model"""
        model = DGAN(DGANConfig(
            max_sequence_len=24,
            sample_len=12,
            batch_size=min(self.features.shape[0], 1000),
            apply_feature_scaling=True,
            apply_example_scaling=False,
            use_attribute_discriminator=False,
            attribute_loss_coef=1.0,
            generator_learning_rate=3e-5,
            discriminator_learning_rate=3e-5,
            epochs=50,

        ))

        model.train_numpy(features=self.features)

        """generate synthetic data"""
        _, synthetic_features = model.generate_numpy(1000)

        synthetic_df = pd.DataFrame(synthetic_features.reshape(-1, synthetic_features.shape[2]),
                                    columns=self.df.columns[1:])

        synthetic_df.to_csv(self.report_path + self.output_file + ".csv")
        print(synthetic_df.describe())

        return synthetic_df


if __name__ == "__main__":
    data = Doppelgan(
        filename=r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop//ENERCOOP_load_profiles.parquet.gzip",
        outputdir=r"C:/Users/FrancescaConselvan/Documents/MODERATE/results",
        output_filename="/synthetic_data_5",
        load_profile_carrier="Enercoop")

    # data.plot_distribution()
    data.features_GAN()
    data.train_DGAN()
