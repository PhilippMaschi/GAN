import pandas as pd
import numpy as np


def import_data(filePath):
    df = pd.read_parquet(filePath)
    return df


def keep_only_one_year(df, startDate, endDate):
    df = df[(df['timestamp'] >= startDate) & (df['timestamp'] < endDate)]
    return df


def set_index(df):
    df = df.set_index('timestamp')
    return df


def keep_only_profile_columns(df):
    colsToKeep = [col for col in df if col.isdigit()]
    df = df[colsToKeep]
    return df


def import_labels(filePath):
    df = pd.read_csv(filePath, sep = ';')
    df['name'] = df['name'].str.split('_', expand = True)[1]
    return df


def get_labels_for_cluster(filePath, clusterLabels):
    df = import_labels(filePath)
    if len(clusterLabels) == 0:
        labels = df.loc[:, 'labels'].to_list()
    else:
        labels = df.loc[df['labels'].isin(clusterLabels), 'name'].to_list()
    return labels


def keep_only_specific_labels(df, labels, maxProfileCount):
    df = df[labels[:maxProfileCount]]
    return df


def limit_load_sums(series, alpha):
    colsToRemove = set(series[(series < np.quantile(series, alpha/2)) | (series > np.quantile(series, 1 - alpha/2))].index)
    return colsToRemove


def find_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = q3 - q1
    colsToRemove = set(series[(series < q1 - 1.5*IQR) | (series > q3 + 1.5*IQR)].index)
    return colsToRemove


def outlier_removal_wrapper(df, alpha):
    loadSums = df.sum()
    initialColCount = df.shape[1]
    colsToRemove = limit_load_sums(loadSums, alpha) | find_outliers(df.min()) | find_outliers(df.max())
    df = df.drop(columns = colsToRemove)
    print(f'Outlier detection: {len(colsToRemove)} profiles were removed \
        ({initialColCount} → {initialColCount - len(colsToRemove)}).')
    return df


def preproc_wrapper(inputFilePath, startDate, endDate, labelsFilePath, clusterLabels, maxProfileCount, alpha):
    df = import_data(inputFilePath)
    df = keep_only_one_year(df, startDate, endDate)
    df = set_index(df)
    df = keep_only_profile_columns(df)
    labels = get_labels_for_cluster(labelsFilePath, clusterLabels)
    df = keep_only_specific_labels(df, labels, maxProfileCount)
    df = outlier_removal_wrapper(df, alpha)
    return df

####################################################################################################

from pathlib import Path

DATA_PATH = Path().absolute() / 'data' / 'ENERCOOP'
INPUT_FILENAME = 'all_load_profiles.parquet.gzip'
LABELS_FILENAME = 'DBSCAN_15_clusters_labels.csv'
INPUT_PATH = DATA_PATH / INPUT_FILENAME
LABELS_PATH = DATA_PATH / LABELS_FILENAME
START_DATE = '2021-06-01 00:00:00'
END_DATE = '2022-06-01 00:00:00'
CLUSTER_LABELS = [1]
MAX_PROFILE_COUNT = None
ALPHA = 0.2

X_TRAIN = preproc_wrapper(
    inputFilePath = INPUT_PATH,
    startDate = START_DATE,
    endDate = END_DATE,
    labelsFilePath = LABELS_PATH,
    clusterLabels = CLUSTER_LABELS,
    maxProfileCount = MAX_PROFILE_COUNT,
    alpha = ALPHA
)