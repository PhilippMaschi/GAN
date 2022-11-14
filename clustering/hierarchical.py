from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_score
import pandas as pd
from dtaidistance import clustering, dtw
from scipy.cluster.hierarchy import fcluster

from plotting import plot_HCA, plot_silhouette_method, plot_HCA_w_DTW


def HCA(df, method = 'ward'):   #hierarchical cluster analysis
    Z = linkage(df, method = method)
    fig = plot_HCA(Z, df)
    return fig, Z


def silhouette_method(X, clusterings):
    k_list = clusterings.keys()
    sil_scores = []
    for k in k_list:
        sil_scores.append(silhouette_score(X, clusterings[k]))
    
    df_results = pd.DataFrame(
        data = list(zip(k_list, sil_scores)),
        columns = ['k', 'silhouette score']
    )
    fig = plot_silhouette_method(df_results)
    return df_results, fig


def HCA_w_DTW(df):  #hierarchical cluster analysis with dynamic time warping
    X = df.to_numpy()
    model = clustering.HierarchicalTree(dists_fun = dtw.distance_matrix_fast, dists_options = {})
    labels = model.fit(X)
    k_list = list(range(2, len(X) + 1))
    clusterings = {}
    for k in k_list:
        clustering_temp = fcluster(model.linkage, t = k, criterion = 'maxclust').tolist()
        if len(set(clustering_temp)) == k:
            clusterings[k] = clustering_temp
    
    df_results = pd.DataFrame(clusterings.items(), columns = ['k', 'labels'])
    df_sil_scores, fig_sil_scores = silhouette_method(X, clusterings)
    df_results = df_results.merge(df_sil_scores)
    fig_results = plot_HCA_w_DTW(model)
    return df_results, fig_results, fig_sil_scores