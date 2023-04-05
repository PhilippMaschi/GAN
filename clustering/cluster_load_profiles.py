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
import inspect

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cophenet, dendrogram
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.utils import shuffle
from sklearn.metrics import davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
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
        cluster_df = df.transpose()
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
            try:
                new_x_labels.append(ts[int(label)].replace("_A+(Wh)", "").replace("_A-(Wh)", ""))
            except:
                pass
        # set new x_ticks
        xticks = ax.get_xticks()
        plt.xticks(xticks, new_x_labels, rotation=-90)

        # draw horizontal line to determine number of clusters:
        # plt.axhline(y=hight, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(self.figure_path / f"Hierarchical_cluster.png")
        plt.show()
        plt.close()

    def gap_statistic(self, X, clusterer, k_range) -> list:
        """
        Calculate the gap statistic for a given clustering algorithm and dataset.

        Parameters:
        - X: a 2D array of shape (n_samples, n_features) containing the dataset
        - clusterer: a clustering algorithm (e.g. KMeans, AgglomerativeClustering)
        - k_range: a range of values for the number of clusters

        Returns:
        - gap: the gap statistic
        """
        # calculate the within-cluster sum of squares (WCSS) for each value of k
        WCSS = []
        davies_bouldin = []
        for k in k_range:
            clusterer.set_params(n_clusters=k)
            clusterer.fit(X)
            WCSS.append(clusterer.inertia_)

        # calculate the reference dispersion values
        reference_dispersion = []
        for k in k_range:
            reference_dispersion.append(self._reference_dispersion(X, k))

        # calculate the gap statistic
        gap = np.log(reference_dispersion) - np.log(WCSS)

        return gap

    def _reference_dispersion(self, X, k):
        """
        Calculate the reference dispersion for a given dataset and number of clusters.

        Parameters:
        - X: a 2D array of shape (n_samples, n_features) containing the dataset
        - k: the number of clusters

        Returns:
        - reference_dispersion: the reference dispersion value
        """
        n_samples, _ = X.shape
        dispersion = []
        for _ in range(10):
            X_sample = shuffle(X, random_state=None)
            clusterer = KMeans(n_clusters=k, random_state=None)
            clusterer.fit(X_sample)
            dispersion.append(clusterer.inertia_)
        return np.mean(dispersion)

    def plot_gap_statistics(self, gap_list: list, k_range: np.array, max_gap: int) -> None:
        plt.plot(k_range, gap_list, label="gap", marker="D")
        plt.xlabel("k")
        plt.ylabel("gap value")
        ax = plt.gca()
        low, high = ax.get_ylim()
        plt.vlines(x=max_gap, ymin=low, ymax=high,
                   label=f"maximum at k={max_gap}, score={round(np.max(gap_list), 4)}",
                   linestyles="--", colors="black")
        plt.legend()
        plt.grid()
        plt.title("Gap analysis for KMeans Clustering")
        plt.savefig(self.figure_path / "Gap_analysis.png")
        plt.show()

    def plot_davies_bouldin_index(self, bouldin_list: list, k_range: np.array, min_davies: int, algorithm: str) -> None:
        plt.plot(k_range, bouldin_list, label="davies bouldin index", marker="D")
        plt.xlabel("k")
        plt.ylabel("davies bouldin value")
        ax = plt.gca()
        low, high = ax.get_ylim()
        plt.vlines(x=min_davies, ymin=low, ymax=high,
                   label=f"minimum at k={min_davies}, score={round(np.min(min_davies), 4)}",
                   linestyles="--", colors="black")
        plt.legend()
        plt.grid()
        plt.title(f"Davies Bouldin score for {algorithm} Clustering")
        plt.savefig(self.figure_path / f"Davies_Bouldin_analysis_{algorithm}.png")
        plt.show()

    def davies_bouldin_analysis(self,  X, k_range: np.array) -> (list, list):
        """
        Calculate the davies bouldin statistic for a given clustering algorithm and dataset.

        Parameters:
        - X: a 2D array of shape (n_samples, n_features) containing the dataset
        - k_range: a range of values for the number of clusters

        Returns:
        - davies bouldin (list): the davies bouldin statistic
        """
        cluster_kmeans = KMeans()
        cluster_agglo = AgglomerativeClustering()
        davies_bouldin_kmeans = []
        davies_bouldin_agglo = []
        for k in k_range:
            cluster_kmeans.set_params(n_clusters=k)
            cluster_agglo.set_params(n_clusters=k)

            model_kmeans = cluster_kmeans.fit_predict(X)
            model_agglo = cluster_agglo.fit_predict(X)
            davies_bouldin_kmeans.append(davies_bouldin_score(X, model_kmeans))
            davies_bouldin_agglo.append(davies_bouldin_score(X, model_agglo))

        return davies_bouldin_kmeans, davies_bouldin_agglo

    def find_number_of_cluster(self, df: pd.DataFrame, k_range: np.array) -> None:
        """
        creates a elbow method graph done k-means
        @param df: normalized dataframe
        @return: returns the optimal number of cluster

        CAREFUL, DEPENDING ON HOW KMEANS IS INITIALIZED THE CLUSTERS ARE NOT ALWAYS THE EXACT SAME, thus also
        the optimal number of cluster can change when running it multiple times, which is idk...
        """
        cluster_df = df.transpose()

        # Elbow distortion method for Agglomerative clustering
        visualizer_distortion = KElbowVisualizer(AgglomerativeClustering(), k=k_range, timings=False)
        visualizer_distortion.fit(cluster_df)
        visualizer_distortion.show(outpath=self.figure_path / "Elbow_distortion_Agglo.png", clear_figure=True)
        print(f"saved optimal number of clusters using the elbow-distortion Agglo method under: \n "
              f"{self.figure_path / 'Elbow_distortion_Agglo.png'}")

        # Elbow distortion method for KMeans clustering
        visualizer_distortion = KElbowVisualizer(KMeans(), k=k_range, timings=False)
        visualizer_distortion.fit(cluster_df)
        visualizer_distortion.show(outpath=self.figure_path / "Elbow_distortion_Kmeans.png", clear_figure=True)
        print(f"saved optimal number of clusters using the elbow-distortion Kmeans method under: \n "
              f"{self.figure_path / 'Elbow_distortion_Kmeans.png'}")

        # Silhouette method for Agglomerative clustering
        visualizer_silhouette = KElbowVisualizer(AgglomerativeClustering(), k=k_range, timings=False, metric="silhouette")
        visualizer_silhouette.fit(cluster_df)
        visualizer_silhouette.show(outpath=self.figure_path / "Silhouette_Agglo.png", clear_figure=True)
        print(f"saved optimal number of clusters using Agglo and the silhouette method under: \n "
              f"{self.figure_path / 'Silhouette_Agglo.png'}")

        # Silhouette method for KMeans clustering
        visualizer_silhouette = KElbowVisualizer(KMeans(), k=k_range, timings=False, metric="silhouette")
        visualizer_silhouette.fit(cluster_df)
        visualizer_silhouette.show(outpath=self.figure_path / "Silhouette_KMeans.png", clear_figure=True)
        print(f"saved optimal number of clusters using KMeans and the silhouette method under: \n "
              f"{self.figure_path / 'Silhouette_KMeans.png'}")

        # Calinski method for Agglomerative clustering
        calinski_agglo = KElbowVisualizer(AgglomerativeClustering(), k=k_range, timings=False, metric="calinski_harabasz")
        calinski_agglo.fit(cluster_df)
        calinski_agglo.show(outpath=self.figure_path / "Calinski_Agglo.png", clear_figure=True)
        print(f"saved optimal number of clusters using the elbow-calinski Agglo method under: \n "
              f"{self.figure_path / 'Calinski_Agglo.png'}")

        # Calinski method for KMeans clustering
        calinski_kmeans = KElbowVisualizer(KMeans(), k=k_range, timings=False, metric="calinski_harabasz")
        calinski_kmeans.fit(cluster_df)
        calinski_kmeans.show(outpath=self.figure_path / "Calinski_KMeans.png", clear_figure=True)
        print(f"saved optimal number of clusters using the elbow-calinski KMeans method under: \n "
              f"{self.figure_path / 'Calinski_KMeans.png'}")

        # calculate the number of clusters with the GAP statistic:
        gap = self.gap_statistic(X=cluster_df, clusterer=KMeans(), k_range=k_range)
        # optimal number of clusters is the cluster with the highest gap
        highest_gap = np.argmax(gap) + min(k_range)
        self.plot_gap_statistics(gap_list=gap, k_range=k_range, max_gap=highest_gap)
        print(f"optimal number of cluster using GAP: {highest_gap}")


        # calculate the number of clusters with the davies bouldin statistic
        davies_kmeans, davies_agglo = self.davies_bouldin_analysis(X=cluster_df, k_range=k_range)
        # optimal number of clusters is the cluster with the lowest boulding index:
        lowest_bouldin_kmeans = np.argmin(davies_kmeans) + min(k_range)
        self.plot_davies_bouldin_index(bouldin_list=davies_kmeans,
                                       k_range=k_range,
                                       min_davies=lowest_bouldin_kmeans,
                                       algorithm="KMeans")
        lowest_bouldin_agglo = np.argmin(davies_agglo) + min(k_range)
        self.plot_davies_bouldin_index(bouldin_list=davies_agglo,
                                       k_range=k_range,
                                       min_davies=lowest_bouldin_agglo,
                                       algorithm="Agglomerative")
        print(f"optimal number of cluster using davies bouldin and KMeans: {lowest_bouldin_kmeans}")
        print(f"optimal number of cluster using davies bouldin and Agglomerative clustering: {lowest_bouldin_agglo}")


    def heat_map(self, df: pd.DataFrame) -> plt.figure:
        """ creates a heat map with the hours of the day on the y-axis and the months on the x-axis
         works for a dataframe with a single load profile and for data with multiply profiles
         if multiple profiles are in the dataset the mean value for each hour of the day of all profiles is used
         in the heat map

         @return: returns a matplotlib figure with the heat map
        """
        # add hour, day and month
        df = dataprep.add_hour_of_the_day_to_df(df, DATE)
        df = dataprep.add_day_of_the_month_to_df(df, DATE)
        df.loc[:, "Month"] = dataprep.extract_month_name_from_datetime(DATE)
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
        cluster_df = df.transpose()
        # define model
        agglo_model = AgglomerativeClustering(n_clusters=number_of_cluster, affinity='euclidean', linkage='ward')
        # fit model
        y_agglo = agglo_model.fit_predict(cluster_df)
        total_number_of_profiles = len(y_agglo)

        # plot the heat map for each cluster:
        for i in range(number_of_cluster):
            column_names = list(cluster_df.transpose().columns)
            column_names.insert(0, "Date")
            heat_map_df = pd.concat([DATE, cluster_df.loc[y_agglo == i, :].transpose()], axis=1, ignore_index=True)
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
        cluster_df = df.transpose()
        # define the model
        kmeans_model = KMeans(n_clusters=number_of_cluster)
        # fit the model
        y_kmeans = kmeans_model.fit_predict(cluster_df)
        total_number_of_profiles = len(y_kmeans)
        # plot the heat map for each cluster:
        for i in range(number_of_cluster):
            column_names = list(cluster_df.transpose().columns)
            column_names.insert(0, "Date")
            heat_map_df = pd.concat([DATE, cluster_df.loc[y_kmeans == i, :].transpose()], axis=1, ignore_index=True)
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
            heat_map_df = pd.concat([DATE, cluster_df.loc[y_db == i, :].transpose()], axis=1, ignore_index=True)
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
        cluster_df = df.transpose()
        model_hdb_scan = hdbscan.HDBSCAN(min_cluster_size=2,  # The minimum number of samples in a group for that group to be considered a cluster
                                         min_samples=None,  # defaults to min_cluster_size
                                         cluster_selection_epsilon=10.0,  # A distance threshold. Clusters below this value will be merged.
                                         max_cluster_size=0,  # A limit to the size of clusters returned by the eom algorithm, does not work with "leaf"
                                         metric="euclidean",
                                         alpha=1.0,  # A distance scaling parameter as used in robust single linkage.
                                         p=None,
                                         algorithm="best",
                                         leaf_size=40,
                                         approx_min_span_tree=True,  # an provide a significant speedup, but the resulting clustering may be of marginally lower quality.
                                         gen_min_span_tree=False,  # for later analysis (useless now)
                                         core_dist_n_jobs=12,
                                         cluster_selection_method="eom",  # options are eom and leaf
                                         allow_single_cluster=True,
                                         prediction_data=False,  # Whether to generate extra cached data for predicting labels or membership vectors few new unseen points later.
                                         match_reference_implementation=False,  # useless
                                         )

        model_hdb_scan.fit(cluster_df)

        labels = model_hdb_scan.labels_
        number_of_outliers = len(labels[labels == -1])
        number_of_cluster = model_hdb_scan.labels_.max()
        total_number_of_profiles = len(labels)

        # plot the heat map for each cluster:
        for i in range(number_of_cluster):
            column_names = list(cluster_df.transpose().columns)
            column_names.insert(0, "Date")

            heat_map_df = pd.concat([DATE, cluster_df.loc[labels == i + 1, :].transpose()], axis=1, ignore_index=True)
            heat_map_df = heat_map_df.rename(columns={
                old_name: column_names[i] for i, old_name in enumerate(heat_map_df.columns)
            })
            fig = self.heat_map(heat_map_df)
            ax = fig.gca()
            percentage_number_of_profiles = round(len(cluster_df[labels == i]) / total_number_of_profiles * 100, 2)
            ax.set_title(f"HDBSCAN cluster: {i + 1}; {percentage_number_of_profiles}% of all profiles")
            plt.tight_layout()
            figure_path = self.figure_path / f"HDBSCAN{number_of_cluster}" / f"HDBSCAN_cluster_Nr_{i + 1}.png"
            self.check_figure_path(figure_path.parent)
            plt.savefig(figure_path, bbox_inches='tight')
            plt.close()

        print(f"created {number_of_cluster} HDBSCAN cluster")


if __name__ == "__main__":
    profiles = DataImporter().main(create_json=False)

    normalized_df = dataprep.normalize_all_loads(profiles)
    # convert to float32:
    normalized_df = dataprep.define_float_type(normalized_df)
    DATE = normalized_df["Date"]
    normalized_df = normalized_df.drop(columns=["Date"])

    df = pd.read_excel(r"C:\Users\mascherbauer\PycharmProjects\GAN\data\Mean_scaled_consumed_energy.xlsx", index_col=0, engine="openpyxl")
    df_cluster = df.T.reset_index(drop=True)
    # determine optimal clusters:
    Cluster().find_number_of_cluster(df.T, k_range=np.arange(2, 15))
    number_of_cluster = 11

    # hierachical cluster to see how many clusters:
    Cluster().hierarchical_cluster(df_cluster)  # creates a figure
    # #
    # # # cluster with agglomerative:
    Cluster().agglomerative_cluster(df_cluster, number_of_cluster=number_of_cluster)
    # # # kmeans cluster
    Cluster().kmeans_cluster(df_cluster, number_of_cluster=number_of_cluster)
    # cluster with HDBSCAN
    # Cluster().cluster_hdb_scan(normalized_df)

    # # cluster with DBSCAN
    # Cluster().db_scan_cluster(normalized_df)

    # TODO try dtw!! (should be better for timeseries) -> Update: dtw is way too slow and resource intensive

    #
    #  and SAX (maybe for nicer heat maps - noise reduction)
