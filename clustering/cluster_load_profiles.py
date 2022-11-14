import time
from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data.import_data import DataImporter
from config import Config
import data.prepare_data as dataprep

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cophenet, dendrogram
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import hdbscan


class Cluster:
    def __init__(self):
        self.figure_path = Config().fig_cluster

    def check_figure_path(self, path: Path) -> None:
        """ checks if dictionary exists and if not creates it"""
        if not os.path.exists(path):
            os.makedirs(path)

    def drop_date_related_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ drops all the date related columns leaving only load columns"""
        return df.drop(columns=["Hour", "Day", "Month", "Date"])

    def hierarchical_cluster(self, df: pd.DataFrame):
        # the clustering clusters after the index so we are transposing the df
        cluster_df = self.drop_date_related_columns(df).transpose()
        # Calculate the distance between each sample

        # possible linkages are: ward, average
        linkage_method = "ward"
        Z = hierarchy.linkage(cluster_df, linkage_method)

        # create figure to visualize the cluster
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=18)
        # Plot with Custom leaves
        dendrogram(Z, leaf_rotation=90, show_contracted=True)  # , annotate_above=0.1)  # , truncate_mode="lastp")

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

    def elbow_method(self, df: pd.DataFrame) -> int:
        """
        creates a elbow method graph done k-means
        @param df: normalized dataframe
        @return: returns the optimal number of cluster

        CAREFUL, DEPENDING ON HOW KMEANS IS INITIALIZED THE CLUSTERS ARE NOT ALWAYS THE EXACT SAME, thus also
        the optimal number of cluster can change when running it multiple times, which is idk...
        """
        number_of_cluster = np.arange(1, 30)
        # the clustering clusters after the index so we are transposing the df
        cluster_df = self.drop_date_related_columns(df).transpose()
        inertia = []
        distortions = []
        for number in number_of_cluster:
            kmeans_model = KMeans(n_clusters=number)
            kmeans_model.fit_predict(cluster_df)
            inertia.append(kmeans_model.inertia_)
            distortions.append(sum(np.min(cdist(cluster_df, kmeans_model.cluster_centers_,
                                                'euclidean'), axis=1)) / cluster_df.shape[0])
            print(f"fiting kmeans with {number} cluster")

        def calculate_optimal_number(criterion: list) -> int:
            # calculate the delta1, delta2 and the strength of each cluster after
            # https://www.datasciencecentral.com/how-to-automatically-determine-the-number-of-clusters-in-your-dat/
            delta1 = [criterion[i] - criterion[i + 1] for i in range(len(criterion)) if i < len(criterion) - 1]
            delta2 = [delta1[i] - delta1[i + 1] for i in range(len(delta1)) if i < len(delta1) - 1]
            delta1.insert(0, np.nan)
            delta2.insert(0, np.nan)
            delta2.insert(0, np.nan)

            strength = np.array(delta2) - np.array(delta1)
            # optimal cluster number is where the strength has its maximum
            # since python starts counting at 0 number of cluster is +1 and the strength is defined as the difference
            # of the delta2-delta1 at number of cluster +1 we don't need to add anything to the index:
            strength = strength[~np.isnan(strength)]
            # remove first two clusters because they are nan, add + 2 to optimal number
            number_optimal = int(np.argmax(strength)) + 2
            return number_optimal

        optimal_number = calculate_optimal_number(distortions)
        optimal_number_2 = calculate_optimal_number(inertia)

        print(f"optimal number of clusters was found to be: {optimal_number}")
        # plot the distortions so we can visualy check if the numbers are correct
        fig = plt.figure()
        ax = fig.gca()
        plt.plot(number_of_cluster, inertia)
        ymin, ymax = ax.get_ylim()
        plt.vlines(x=optimal_number, ymin=ymin, ymax=ymax, colors="red", label="distortion")
        plt.vlines(x=optimal_number_2, ymin=ymin, ymax=ymax, colors="green", label="inertia", linestyles="--")
        plt.ylabel("inertia")
        plt.legend()
        plt.title("Elbow method")
        plt.savefig(self.figure_path / "Elbow_method.png")
        plt.show()

        # check if the two methods come to the same result
        if optimal_number != optimal_number_2:
            print(f"optimal number is dependent on methode (inertia, distortion) \n "
                  f"using the higher number which is {max([optimal_number, optimal_number_2])}")
            optimal_number = max([optimal_number, optimal_number_2])

        return optimal_number

    def heat_map(self, df: pd.DataFrame) -> plt.figure:
        """ creates a heat map with the hours of the day on the y-axis and the months on the x-axis
         works for a dataframe with a single load profile and for data with multiply profiles
         if multiple profiles are in the dataset the mean value for each hour of the day of all profiles is used
         in the heat map

         @return: returns a matplotlib figure with the heat map
        """
        # add hour, day and month
        df = dataprep.add_hour_of_the_day_to_df(df)
        df = dataprep.add_day_of_the_month_to_df(df)
        df.loc[:, "Month"] = dataprep.extract_month_name_from_datetime(profiles)
        # prepare the dataframe so it can be plottet as heat map
        melted_df = df.melt(id_vars=["Date", "Hour", "Day", "Month"])
        pivot_df = pd.pivot_table(data=melted_df, index="Hour", columns=["Month", "Day"], values="value")
        # sort the columns so the months are not in random order:
        heat_map_table = dataprep.sort_columns_months(pivot_df)

        # create heat map
        fig = plt.figure()
        sns.heatmap(heat_map_table)
        x_tick_labels = heat_map_table.columns.get_level_values(level=0).unique()
        ax = plt.gca()
        ax.set_xticks(np.arange(15, 365, 30))
        ax.set_xticklabels(x_tick_labels)
        plt.xlabel("month")
        # save figure
        # plt.savefig(self.figure_path / f"heat_map_loads.png")
        # fig.show()
        return fig

    def agglomerative_cluster(self, df: pd.DataFrame, number_of_cluster: int):
        # the clustering clusters after the index so we are transposing the df
        date = df.loc[:, "Date"]  # save it to merge it back for heat map
        cluster_df = self.drop_date_related_columns(df).transpose()
        # define model
        agglo_model = AgglomerativeClustering(n_clusters=number_of_cluster, affinity='euclidean', linkage='ward')
        # fit model
        y_agglo = agglo_model.fit_predict(cluster_df)
        total_number_of_profiles = len(y_agglo)

        # plot the heat map for each cluster:
        for i in range(number_of_cluster):
            column_names = list(cluster_df.transpose().columns)
            column_names.insert(0, "Date")
            heat_map_df = pd.concat([date, cluster_df.loc[y_agglo == i, :].transpose()], axis=1, ignore_index=True)
            heat_map_df = heat_map_df.rename(columns={
                old_name: column_names[i] for i, old_name in enumerate(heat_map_df.columns)
            })
            fig = self.heat_map(heat_map_df)
            ax = fig.gca()
            percentage_number_of_profiles = round(len(cluster_df[y_agglo == i]) / total_number_of_profiles * 100, 2)
            ax.set_title(f"Agglo cluster: {i + 1}; {percentage_number_of_profiles}% of all profiles")
            plt.tight_layout()
            figure_path = self.figure_path / f"Agglomerative{number_of_cluster}" / f"Agglo_cluster_Nr_{i + 1}.png"
            self.check_figure_path(figure_path.parent)
            plt.savefig(figure_path, bbox_inches='tight')
            plt.close()
        print(f"created {number_of_cluster} agglomerative cluster")

    def kmeans_cluster(self, df: pd.DataFrame, number_of_cluster: int):
        # the clustering clusters after the index so we are transposing the df
        date = df.loc[:, "Date"]  # save it to merge it back for heat map
        cluster_df = self.drop_date_related_columns(df).transpose()
        # define the model
        kmeans_model = KMeans(n_clusters=number_of_cluster)
        # fit the model
        y_kmeans = kmeans_model.fit_predict(cluster_df)
        total_number_of_profiles = len(y_kmeans)
        # plot the heat map for each cluster:
        for i in range(number_of_cluster):
            column_names = list(cluster_df.transpose().columns)
            column_names.insert(0, "Date")
            heat_map_df = pd.concat([date, cluster_df.loc[y_kmeans == i, :].transpose()], axis=1, ignore_index=True)
            heat_map_df = heat_map_df.rename(columns={
                old_name: column_names[i] for i, old_name in enumerate(heat_map_df.columns)
            })
            fig = self.heat_map(heat_map_df)
            ax = fig.gca()
            percentage_number_of_profiles = round(len(cluster_df[y_kmeans == i]) / total_number_of_profiles * 100, 2)
            ax.set_title(f"Kmeans cluster: {i + 1}; {percentage_number_of_profiles}% of all profiles")
            plt.tight_layout()
            figure_path = self.figure_path / f"KMeans{number_of_cluster}" / f"KMeans_cluster_Nr_{i + 1}.png"
            self.check_figure_path(figure_path.parent)
            plt.savefig(figure_path, bbox_inches='tight')
            plt.close()
        print(f"created {number_of_cluster} kmeans cluster")

    def db_scan_cluster(self, df: pd.DataFrame):  # doesnt need the number of cluster!
        # the clustering clusters after the index so we are transposing the df
        date = df.loc[:, "Date"]  # save it to merge it back for heat map
        cluster_df = self.drop_date_related_columns(df).transpose()
        # TODO find the right method to cluster loads with db scan
        db_model = DBSCAN(eps=10, min_samples=3, metric="euclidean", metric_params=None, algorithm="auto",
                          leaf_size=30, p=None, n_jobs=4)
        y_db = db_model.fit(cluster_df)
        total_number_of_profiles = len(y_db)
        number_of_outliers = len(y_db.labels_[y_db.labels_ == -1])  # outliers are marked with -1
        # plot the heat map for each cluster:
        for i in range(y_db):
            column_names = list(cluster_df.transpose().columns)
            column_names.insert(0, "Date")
            heat_map_df = pd.concat([date, cluster_df.loc[y_db == i, :].transpose()], axis=1, ignore_index=True)
            heat_map_df = heat_map_df.rename(columns={
                old_name: column_names[i] for i, old_name in enumerate(heat_map_df.columns)
            })
            fig = self.heat_map(heat_map_df)
            ax = fig.gca()
            percentage_number_of_profiles = round(len(cluster_df[y_db == i]) / total_number_of_profiles * 100, 2)
            ax.set_title(f"DBSCAN cluster: {i + 1}; {percentage_number_of_profiles}% of all profiles")
            plt.tight_layout()
            figure_path = self.figure_path / f"DBSCAN" / f"DBSCAN_cluster_Nr_{i + 1}.png"
            self.check_figure_path(figure_path.parent)
            plt.savefig(figure_path, bbox_inches='tight')
            plt.close()

    def cluster_hdb_scan(self, df: pd.DataFrame):
        # the clustering clusters after the index so we are transposing the df
        date = df.loc[:, "Date"]  # save it to merge it back for heat map
        cluster_df = self.drop_date_related_columns(df).transpose()
        model_hdb_scan = hdbscan.HDBSCAN(algorithm='best',
                                         approx_min_span_tree=True,
                                         gen_min_span_tree=False,
                                         metric='euclidean',
                                         min_cluster_size=2,
                                         min_samples=1,
                                         p=None)
        model_hdb_scan.fit(cluster_df)

        labels = model_hdb_scan.labels_
        number_of_outliers = len(labels[labels == -1])
        number_of_cluster = model_hdb_scan.labels_.max()
        total_number_of_profiles = len(labels)


        # plot the heat map for each cluster:
        for i in range(number_of_cluster):
            column_names = list(cluster_df.transpose().columns)
            column_names.insert(0, "Date")

            heat_map_df = pd.concat([date, cluster_df.loc[labels == i+1, :].transpose()], axis=1, ignore_index=True)
            heat_map_df = heat_map_df.rename(columns={
                old_name: column_names[i] for i, old_name in enumerate(heat_map_df.columns)
            })
            fig = self.heat_map(heat_map_df)
            ax = fig.gca()
            percentage_number_of_profiles = round(len(cluster_df[labels == i]) / total_number_of_profiles * 100, 2)
            ax.set_title(f"HDBSCAN cluster: {i + 1}; {percentage_number_of_profiles}% of all profiles")
            plt.tight_layout()
            figure_path = self.figure_path / f"HDBSCAN" / f"HDBSCAN_cluster_Nr_{i + 1}.png"
            self.check_figure_path(figure_path.parent)
            plt.savefig(figure_path, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    profiles = DataImporter().main(create_json=False)

    normalized_df = dataprep.normalize_all_loads(profiles)
    # convert to float32:
    normalized_df = dataprep.define_float_type(normalized_df)

    Cluster().heat_map(normalized_df)

    # determine optimal clusters:
    number_of_cluster = Cluster().elbow_method(normalized_df)
    # number_of_cluster = 12

    # hierachical cluster to see how many clusters:
    Cluster().hierarchical_cluster(normalized_df)  # creates a figure

    # cluster with agglomerative:
    Cluster().agglomerative_cluster(normalized_df, number_of_cluster=number_of_cluster)
    # kmeans cluster
    Cluster().kmeans_cluster(normalized_df, number_of_cluster=number_of_cluster)

    Cluster().cluster_hdb_scan(normalized_df)
    # cluster with DBSCAN
    Cluster().db_scan_cluster(normalized_df)


    # TODO try dtw!! (should be better for timeseries) -> Update: dtw is way too slow and resource intensive

    #
    #  and SAX (maybe for nicer heat maps - noise reduction)
