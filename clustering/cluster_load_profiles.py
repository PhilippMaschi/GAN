from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data.import_data import DataImporter
from data.prepare_data import DataPrep
from config import Config

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cophenet, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans


class Cluster:
    def __init__(self):
        self.figure_path = Config().fig_cluster

    def hierarchical_cluster(self, df: pd.DataFrame):
        # the clustering clusters after the index so we are transposing the df
        cluster_df = df.drop(columns=["hour", "day", "month", "date"]).transpose()
        # Calculate the distance between each sample

        # possible linkages are: ward, average
        linkage_method = "ward"
        Z = hierarchy.linkage(cluster_df, linkage_method)

        # create figure to visualize the cluster
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=18)
        # Plot with Custom leaves
        dendrogram(Z, leaf_rotation=90, show_contracted=True)#, annotate_above=0.1)  # , truncate_mode="lastp")

        # set x-ticks to csv numbers:
        ts = pd.Series(cluster_df.index)
        x_tick_labels = [item.get_text() for item in ax.get_xticklabels()]
        new_x_labels = []
        for label in x_tick_labels:
            # exclude the A+(Wh)...
            new_x_labels.append(ts[int(label)].replace("_A+(Wh)", "").replace("_A-(Wh)", ""))
        # set new x_ticks
        xticks = ax.get_xticks()
        plt.xticks(xticks, new_x_labels, rotation=-90)

        # draw horizontal line to determine number of clusters:
        # plt.axhline(y=hight, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(self.figure_path / f"Hierarchical_cluster.png")
        plt.show()
        plt.close()


    def heat_map(self, df: pd.DataFrame):
        """ creates a heat map with the hours of the day on the y-axis and the months on the x-axis
         works for a dataframe with a single load profile and for data with multiply profiles
         if multiple profiles are in the dataset the mean value for each hour of the day of all profiles is used
         in the heat map
        """
        # add hour of the day
        df = DataPrep().add_hour_of_the_day_to_df(df)
        df = DataPrep().add_day_of_the_month_to_df(df)
        # prepare the dataframe so it can be plottet as heat map
        melted_df = df.melt(id_vars=["date", "hour", "day", "month"])
        pivot_df = pd.pivot_table(data=melted_df, index="hour", columns=["month", "day"], values="value")
        # sort the columns so the months are not in random order:
        heat_map_table = DataPrep().sort_columns_months(pivot_df)

        # create heat map
        sns.heatmap(heat_map_table)
        x_tick_labels = heat_map_table.columns.get_level_values(level=0).unique()
        ax = plt.gca()
        ax.set_xticks(np.arange(15, 365, 30))
        ax.set_xticklabels(x_tick_labels)
        plt.xlabel("month")
        plt.tight_layout()
        # save figure
        plt.savefig(self.figure_path / f"heat_map_loads.png")
        plt.show()



    def split_cluster(df, title, number_of_cluster, year):
        # define model
        hc = AgglomerativeClustering(n_clusters=number_of_cluster, affinity='euclidean', linkage='ward')
        # fit model
        y_hc = hc.fit_predict(df)

        # define the model
        model = KMeans(n_clusters=number_of_cluster)
        # fit the model
        y_kmeans = model.fit_predict(df)

        excel = {}
        for i in range(number_of_cluster):
            sns.heatmap(df[y_hc == i], vmin=0, vmax=1)
            anzahl_laender = len(df[y_hc == i])
            plt.title("Cluster: " + str(i + 1) + " Number of countries: " + str(anzahl_laender))
            plt.savefig("output/" + title + "/" + str(year) + "/Cluster_Nr_" + str(i + 1) + ".png", bbox_inches='tight')
            plt.close()

            # create excel with the countries of each cluster:
            excel["Agglo " + str(i + 1)] = df[y_hc == i].index.tolist()

            # Vergleich mit KMeans:
            sns.heatmap(df[y_kmeans == i], vmin=0, vmax=1)
            anzahl_laender = len(df[y_kmeans == i])
            plt.title("Cluster_KMeans: " + str(i + 1) + " Number of countries: " + str(anzahl_laender))
            plt.savefig("output/" + title + "/" + str(year) + "/Cluster_KMeans_Nr_" + str(i + 1) + ".png",
                        bbox_inches='tight')
            plt.close()

            # create excel with the countries of each cluster:
            excel["KMeans " + str(i + 1)] = df[y_kmeans == i].index.tolist()

        df_excel = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in excel.items()]))
        df_excel.to_excel("output/" + title + "/" + str(year) + "/cluster overview.xlsx")

        # create overall HEATMAP:
        clustermap = sns.clustermap(df, method="ward")  # , cmap="vlag")
        plt.savefig("output/" + title + "/" + str(year) + "/Clustermap.png")
        plt.close()



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
    profiles = DataImporter().main(create_json=False)
    profiles.loc[:, "month"] = DataPrep().extract_month_name_from_datetime(profiles)
    positive_profiles, _ = DataPrep().differentiate_positive_negative_loads(profiles)
    normalized_df = DataPrep().normalize_all_loads(positive_profiles)

    Cluster().heat_map(normalized_df)

    # hierachical cluster
    Cluster().hierarchical_cluster(normalized_df)
