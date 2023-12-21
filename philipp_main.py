import cryptpandas as crp
import os, sys, signal
import getpass
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from GAN_Philipp import GAN
import argparse

print(f'torch {torch.__version__}')


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
    df_train["month sin"] = np.sin(df_train.index.month * (2 * np.pi / len(df_train.index.month.unique())))
    df_train["month cos"] = np.cos(df_train.index.month * (2 * np.pi / len(df_train.index.month.unique())))
    df_train["weekday sin"] = np.sin(df_train.index.weekday * (2 * np.pi / len(df_train.index.weekday.unique())))
    df_train["weekday cos"] = np.cos(df_train.index.weekday * (2 * np.pi / len(df_train.index.weekday.unique())))

    non_numeric_cols = [col for col in df_train.columns if not is_number(col)]
    numeric_cols = [col for col in df_train.columns if is_number(col)]
    df_shape = df_train.melt(id_vars=df_train[non_numeric_cols],
                             value_vars=df_train[numeric_cols],
                             var_name='profile')

    df_pivot = df_shape.pivot_table(values='value',
                                    index=[
                                        'date', 'profile', "month sin", "month cos", "weekday sin", "weekday cos",
                                           "day off"
                                    ],
                                    columns='hour of the day')
    # create a shape of the df_pivot that is needed to reshape the generated data from the GAN later
    df_hull = df_pivot.copy()
    df_hull[::] = np.zeros(df_hull.shape)

    target = df_pivot.values
    features = np.vstack([df_pivot.index.get_level_values("month sin").to_numpy(),
                          df_pivot.index.get_level_values("month cos").to_numpy(),
                          df_pivot.index.get_level_values("weekday sin").to_numpy(),
                          df_pivot.index.get_level_values("weekday cos").to_numpy(),
                          df_pivot.index.get_level_values("day off").to_numpy()]).T
    del df_train
    return target, features, df_hull


def create_training_dataframe(password_,
                              clusterLabel: int,
                              number_of_profiles: int,
                              path_to_orig_file: Path = Path().absolute().parent / 'GAN_data',
                              label_csv_filename: str = 'DBSCAN_15_clusters_labels.csv',
                              ) -> pd.DataFrame:
    # Data import
    # all load profiles:
    df_loadProfiles = crp.read_encrypted(path=path_to_orig_file / 'all_profiles.crypt', password=password_)

    # filter the total amount of profiles:
    labels = load_labels_for_cluster(clusterLabel=clusterLabel,
                                     filepath=path_to_orig_file / label_csv_filename,
                                     number_of_profiles=number_of_profiles)
    print(f"number of profiles: {len(labels)}")
    training_df = create_training_data(all_profiles=df_loadProfiles, labels=labels).set_index("timestamp")

    return training_df


def train_gan(
        batchSize: int,
        dimNoise: int,
        training_df: pd.DataFrame,
        cluster_label,
        cluster_algorithm,
        loss: str,
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
    model_name = 'ModelTestPhilipp'
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
        n_transformed_features=4,
        n_number_features=1,
        cluster_label=cluster_label,
        cluster_algorithm=cluster_algorithm,
        n_profiles_trained_on=len([col for col in training_df.columns if is_number(col)]),
        LossFct=loss
    )
    # save df_hull to the model folder so the generated data can be easily reshaped:
    df_hull.to_parquet(Path(model.folder_name) / "hull.parquet.gzip")
    # save the original features to a npz file so it can be used for generating data later:
    np.save(file=Path(model.folder_name) / "original_features.npy", arr=features)
    # save the original metadata:
    orig_metadata = training_df[[col for col in training_df.columns if not is_number(col)]]
    orig_metadata.to_parquet(Path(model.folder_name) / "meta_data.parquet.gzip")

    model.train()
    print(f"Training {model.folder_name} done""")


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
    n_profiles = 100  # kann None sein, dann werden alle Profile genommen
    cluster_label = 0
    batchSize = 1000
    epochs = 5000
    Loss = "BCE"  # BCE, MSE, KLDiv, MAE

    train_df = create_training_dataframe(
        password_=password,
        clusterLabel=cluster_label,
        number_of_profiles=n_profiles,
        label_csv_filename="DBSCAN_15_clusters_labels.csv",
    )

    train_gan(
        batchSize=batchSize,
        dimNoise=noise_dimension,
        training_df=train_df,
        epochCount=epochs,
        cluster_algorithm="DBSCAN",
        cluster_label=cluster_label,
        lr=1e-5,
        maxNorm=1e6,
        loss=Loss
    )
    torch.cuda.empty_cache()

    os.kill(pid, signal.SIGTERM)
    sys.exit()
