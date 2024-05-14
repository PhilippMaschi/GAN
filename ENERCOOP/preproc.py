import pandas as pd
import numpy as np


ALPHA = 0.2
FEATURE_RANGE = (-1, 1)


def import_data(filePath):
    df = pd.read_parquet(filePath)
    return df


def keep_only_full_weeks(df):
    df = df.iloc[df[df.weekday == 'Monday'].index[0]:df[df.weekday == 'Sunday'].index[-1] + 1]
    return df


def import_labels(filePath):
    df = pd.read_csv(filePath, sep = ';')
    df['name'] = df['name'].str.split('_', expand = True)[1]
    return df


def get_labels_for_cluster(filePath, clusterLabels):
    df = import_labels(filePath)
    if len(clusterLabels) == 0:
        labels = df.loc[:, "labels"].to_list()
    else:
        labels = df.loc[df['labels'].isin(clusterLabels), 'name'].to_list()

    return labels


def create_training_data(df, labels, maxProfileCount):
    metaDataCols = [col for col in df.columns if not col.isdigit()]
    df_result = df[metaDataCols + labels[:maxProfileCount]]
    return df_result


def limit_load_sums(series, alpha = ALPHA):
    colsToRemove = set(series[(series < np.quantile(series, alpha/2)) | (series > np.quantile(series, 1 - alpha/2))].index)
    return colsToRemove


def find_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = q3 - q1
    colsToRemove = set(series[(series < q1 - 1.5*IQR) | (series > q3 + 1.5*IQR)].index)
    return colsToRemove


def outlier_removal_wrapper(df):
    numericCols = [col for col in df.columns if col.isdigit()]
    loadSums = df[numericCols].sum()
    colsToRemove = \
        limit_load_sums(loadSums) | find_outliers(df[numericCols].min()) | find_outliers(df[numericCols].max())
    df = df.drop(columns = colsToRemove)
    print(f'Outlier detection: {len(colsToRemove)} profiles were removed \
({len(numericCols)} → {len(numericCols) - len(colsToRemove)}).')
    return df


def df_to_arr(df):
    numericCols = [col for col in df.columns if col.isdigit()]
    df = df[numericCols]
    arr = df.to_numpy()
    return arr


def reshape_arr(arr, dim = 3):
    if int(dim) == 2:
        arr = np.split(arr.T, arr.shape[0]/168, axis = 1)
        arr = np.stack(arr, axis = 2)
        arr = np.expand_dims(arr, axis = 1)
    elif int(dim) == 3:
        arr = np.split(arr.T, arr.shape[0]/24, axis = 1)
        arr = np.stack(arr, axis = 2)
        arr = np.split(arr, arr.shape[2]/7, axis = 2)
        arr = np.stack(arr, axis = 3)
    else:
        raise ValueError(f'{dim} is not a valid dimension {{2, 3\}}!')
    return arr


def revert_reshape_arr(arr, dim = 3):
    if int(dim) == 2:
        arr = np.squeeze(arr, axis = 1)
        arr = np.split(arr, arr.shape[2], axis = 2)
        arr = np.concatenate(arr, axis = 1).squeeze().T
    elif int(dim) == 3:
        arr = np.split(arr, arr.shape[3], axis = 3)
        arr = np.concatenate(arr, axis = 2).squeeze()
        arr = np.split(arr, arr.shape[2], axis = 2)
        arr = np.concatenate(arr, axis = 1).squeeze().T
    return arr


def data_preparation_wrapper(dataFilePath, labelsFilePath, clusterLabels, maxProfileCount):
    df = import_data(dataFilePath)
    df = keep_only_full_weeks(df)
    labels = get_labels_for_cluster(labelsFilePath, clusterLabels)
    df_train = create_training_data(df, labels, maxProfileCount)
    df_train = outlier_removal_wrapper(df_train)
    X_train = df_to_arr(df)
    X_trainResh = reshape_arr(X_train)
    return X_trainResh, X_train