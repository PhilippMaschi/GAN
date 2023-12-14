import cryptpandas as crp
import os, sys, signal
import getpass
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import torch

print(f'torch {torch.__version__}')
from sklearn.preprocessing import minmax_scale, MinMaxScaler
import json
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from datetime import date

from data_manip import remove_incomplete_days
from preproc import import_and_preprocess_data, create_and_add_datetime_features
from GAN_Philipp import GAN, generate_data_from_saved_model
from visualization_script import plot_seasonal_daily_means, compare_peak_and_mean, plot_pca_analysis
from plot import plot_losses
from plot import plot_synthetic_vs_real_samples
import argparse

import gc


def conv_index_to_bins(index):
    """Calculate bins to contain the index values.
    The start and end bin boundaries are linearly extrapolated from
    the two first and last values. The middle bin boundaries are
    midpoints.

    Example 1: [0, 1] -> [-0.5, 0.5, 1.5]
    Example 2: [0, 1, 4] -> [-0.5, 0.5, 2.5, 5.5]
    Example 3: [4, 1, 0] -> [5.5, 2.5, 0.5, -0.5]"""
    assert index.is_monotonic_increasing or index.is_monotonic_decreasing

    # the beginning and end values are guessed from first and last two
    start = index[0] - (index[1] - index[0]) / 2
    end = index[-1] + (index[-1] - index[-2]) / 2

    # the middle values are the midpoints
    middle = pd.DataFrame({'m1': index[:-1], 'p1': index[1:]})
    middle = middle['m1'] + (middle['p1'] - middle['m1']) / 2

    if isinstance(index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(middle).union([start, end])
    elif isinstance(index, (pd.Float64Index, pd.RangeIndex, pd.Int64Index)):
        idx = pd.Float64Index(middle).union([start, end])
    else:
        print('Warning: guessing what to do with index type %s' %
              type(index))
        idx = pd.Float64Index(middle).union([start, end])

    return idx.sort_values(ascending=index.is_monotonic_increasing)


def calc_df_mesh(df):
    """Calculate the two-dimensional bins to hold the index and
    column values."""
    return np.meshgrid(conv_index_to_bins(df.index),
                       conv_index_to_bins(df.columns))


def get_labels_for_cluster(cluster_number: int, filepath: Path) -> pd.DataFrame:
    df_labels = pd.read_csv(os.path.join(filepath), sep=';')
    df_labels['name'] = df_labels['name'].str.split('_', expand=True)[1]
    return df_labels


def load_labels_for_cluster(clusterLabel: int, filepath: Path, number_of_profiles: int):
    df_labels = get_labels_for_cluster(cluster_number=clusterLabel, filepath=filepath)
    label_list = df_labels.loc[df_labels['labels'] == clusterLabel, 'name'].to_list()
    if number_of_profiles is None:
        labels = label_list
    else:
        labels = label_list[:number_of_profiles]
    return labels


def is_number(s):
    """
    Check if the input string s is a number.

    Parameters:
    s (str): The string to check.

    Returns:
    bool: True if s is a number, False otherwise.
    """
    try:
        float(s)  # for int, long and float
    except ValueError:
        return False
    return True


def create_training_data(all_profiles: pd.DataFrame, labels: list):
    # filter the total amount of profiles:
    meta_data_cols = [col for col in all_profiles.columns if not is_number(col)]
    df_profiles = all_profiles[meta_data_cols + labels]
    return df_profiles


def create_numpy_matrix_for_gan(df_train: pd.DataFrame) -> (np.array, np.array, pd.DataFrame):
    """

    Args:
        df_train:

    Returns: target and features and df_hull which is a df which contains the orig index for reshaping the generated
    data later

    """
    m_unique = df_train["month of the year"].unique()
    df_train["month sin"] = df_train["month of the year"].apply(lambda x: np.sin(x * (2 * np.pi / len(m_unique))))
    df_train["month cos"] = df_train["month of the year"].apply(lambda x: np.cos(x * (2 * np.pi / len(m_unique))))
    non_numeric_cols = [col for col in df_train.columns if not is_number(col)]
    numeric_cols = [col for col in df_train.columns if is_number(col)]
    df_shape = df_train.melt(id_vars=df_train[non_numeric_cols],
                             value_vars=df_train[numeric_cols],
                             var_name='profile')

    df_pivot = df_shape.pivot_table(values='value',
                                    index=['date', 'profile', "month sin", "month cos", "day off"],
                                    columns='hour of the day')
    # create a shape of the df_pivot that is needed to reshape the generated data from the GAN later
    df_hull = df_pivot.copy()
    df_hull[::] = np.zeros(df_hull.shape)

    target = df_pivot.values
    features = np.vstack([df_pivot.index.get_level_values("month sin").to_numpy(),
                          df_pivot.index.get_level_values("month cos").to_numpy(),
                          df_pivot.index.get_level_values("day off").to_numpy()]).T
    del df_train
    return target, features, df_hull


def numpy_matrix_to_pandas_table_with_metadata(hull: pd.DataFrame, synthetic_data: np.array, original_meta_data):
    hull[::] = synthetic_data
    synthetic = hull.reset_index()
    # todo the month sin etc. as list to this function dependent on the model so this is automated for other variables
    df_synthetic = synthetic.melt(
        id_vars=['date', 'profile', "month sin", "month cos", "day off"],
        var_name="hour of the day",
        value_name="value")
    df_pivot = df_synthetic.pivot_table(values='value',
                                        index=['date', "month sin", "month cos", "day off", "hour of the day"],
                                        columns='profile').reset_index()
    final = pd.concat([original_meta_data.reset_index(), df_pivot[[col for col in df_pivot.columns if is_number(col)]]],
                      axis=1)

    return final


def create_training_dataframe(password_,
                              clusterLabel: int,
                              number_of_profiles: int,
                              label_csv_filename: str = 'DBSCAN_15_clusters_labels.csv',
                              ) -> pd.DataFrame:
    # Data import
    GAN_data_path = Path().absolute().parent / 'GAN_data'
    # all load profiles:
    df_loadProfiles = crp.read_encrypted(path=os.path.join(GAN_data_path, 'all_profiles.crypt'), password=password_)

    # filter the total amount of profiles:
    labels = load_labels_for_cluster(clusterLabel=clusterLabel,
                                     filepath=GAN_data_path / label_csv_filename,
                                     number_of_profiles=number_of_profiles)
    print(f"number of profiles: {len(labels)}")
    training_df = create_training_data(all_profiles=df_loadProfiles, labels=labels).set_index("timestamp")

    return training_df


def train_gan(
        batchSize: int,
        dimNoise: int,
        training_df: pd.DataFrame,
        epochCount=500,
        lr=1e-5,
        maxNorm=1e6,
        ):

    # create np array with target and features
    target, features, df_hull = create_numpy_matrix_for_gan(training_df.copy())

    # Configure GAN
    if 1 and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('GPU is used.')
    else:
        device = torch.device('cpu')
        print('CPU is used.')

    featureCount = features.shape[1]  # stunden pro tag (pro label hat das model 24 werte)
    # testLabel = 0
    model_name = 'model_test_philipp'
    model = GAN(
        name=model_name,
        device=device,
        batchSize=batchSize,
        target=target,
        features=features,
        dimNoise=dimNoise,
        featureCount=featureCount,
        lr=lr,
        maxNorm=maxNorm,
        epochCount=epochCount,
        # testLabel = testLabel,
        n_transformed_features=2,
        n_number_features=1,
    )

    model.train()
    # save df_hull to the model folder so the generated data can be easily reshaped:
    df_hull.to_parquet(Path(model.folder_name) / "hull.parquet.gzip")
    # save the original features to a npz file so it can be used for generating data later:
    np.save(file=Path(model.folder_name) / "original_features.npy", arr=features)
    # save the original metadata:
    orig_metadata = training_df[[col for col in training_df.columns if not is_number(col)]]
    orig_metadata.to_parquet(Path(model.folder_name) / "meta_data.parquet.gzip")
    print(f"Training {model.folder_name} done""")
    return model.folder_name


def visualize_results_from_model_folder(folder_path, noise_dimension, device):
    # visualize the training results:
    file_names = [file.name for file in Path(folder_path).glob("*.pt")]
    file_names.sort()

    orig_features = np.load(Path(folder_path) / "original_features.npy")
    hull = pd.read_parquet(Path(folder_path) / "hull.parquet.gzip")
    orig_meta_data = pd.read_parquet(Path(folder_path) / "meta_data.parquet.gzip")
    for model in file_names:
        epoch = int(model.replace("epoch_", "").replace(".pt", ""))
        synthetic_data = generate_data_from_saved_model(
            model_path=f"{folder_path}/{model}",
            noise_dim=noise_dimension,
            featureCount=3,  # depends on the features selected in train_gan -> automate
            targetCount=24,
            original_features=orig_features,
            device=device,
        )

        df_synthetic = numpy_matrix_to_pandas_table_with_metadata(hull=hull,
                                                                  synthetic_data=synthetic_data,
                                                                  original_meta_data=orig_meta_data)
        folder_name = Path(folder_path).stem
        output_path = Path(folder_path).parent.parent / "plots" / folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        plot_seasonal_daily_means(df_real=train_df,
                                  df_synthetic=df_synthetic,
                                  output_path=output_path,
                                  epoch_number=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='launches test torch')
    parser.add_argument("--password", default="Ene123Elec#4")
    args = parser.parse_args()

    password = args.password
    if password == "":
        password = getpass.getpass('Password: ')
        print("No password given. Exit now.")
        sys.exit()

    pid = (os.getpid())
    print(pid)
    noise_dimension = 50
    n_profiles = None
    cluster_label = 0
    batchSize = int(1_000)
    train_df = create_training_dataframe(
        password_=password,
        clusterLabel=cluster_label,
        number_of_profiles=n_profiles,
        label_csv_filename="DBSCAN_15_clusters_labels.csv",
    )

    model_folder = train_gan(
         batchSize=batchSize,
         dimNoise=noise_dimension,
         training_df=train_df,
         epochCount=2000,
         lr=1e-5,
         maxNorm=1e6,
    )
    torch.cuda.empty_cache()

    visualize_results_from_model_folder(folder_path=model_folder,
                                        noise_dimension=noise_dimension,
                                        device="cuda:0")

os.kill(pid, signal.SIGTERM)
sys.exit()
