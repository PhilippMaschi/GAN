from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.metrics import silhouette_score

from plotting import plot_ellbow_method, plot_silhouette_method


def ellbow_method(X):
    k_list = list(range(1, len(X) + 1))
    distortions, inertias = [], []
    for k in k_list:
        kmeans = KMeans(n_clusters = k, random_state = 42)
        kmeans.fit_predict(X)
        distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis = 1))/len(X))
        inertias.append(kmeans.inertia_)

    def calculate_k_opt(criterion):
        delta1 = - np.diff(criterion)
        delta2 = - np.diff(delta1, prepend = np.nan)
        strength = delta2 - delta1
        k_opt = np.argwhere(strength == np.nanmax(strength)).item() + 1
        return k_opt, strength

    k_opt_dict, strength_dict = {}, {}
    k_opt_dict['distortion'], strength_dict['distortion'] = calculate_k_opt(distortions)
    k_opt_dict['inertia'], strength_dict['inertia'] = calculate_k_opt(inertias)
    df_results = pd.DataFrame(
        data = list(zip(k_list, distortions, inertias, strength_dict['distortion'], strength_dict['inertia'])),
        columns = ['k', 'distortion', 'inertia', 'strength_distortion', 'strength_inertia']
    )
    fig = plot_ellbow_method(df_results, k_opt_dict)
    return k_opt_dict, df_results, fig


def silhouette_method(X):
    k_list = list(range(2, len(X)))
    sil_scores = []
    for k in k_list:
        kmeans = KMeans(n_clusters = k, random_state = 42)
        labels = kmeans.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
    
    df_results = pd.DataFrame(
        data = list(zip(k_list, sil_scores)),
        columns = ['k', 'silhouette score']
    )
    fig = plot_silhouette_method(df_results)
    return df_results, fig