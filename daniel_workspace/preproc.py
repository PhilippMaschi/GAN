import cryptpandas as crp
import pandas as pd
import numpy as np


ALPHA = 0.2
FEATURE_RANGE = (-1, 1)


def import_data(filePath, password):
    df = crp.read_encrypted(path = filePath, password = password)
    return df


def keep_only_full_weeks(df):
    df = df.iloc[df[df.weekday == 'Monday'].index[0]:df[df.weekday == 'Sunday'].index[-1] + 1]
    return df


def import_labels(filePath):
    df = pd.read_csv(filePath, sep = ';')
    df['name'] = df['name'].str.split('_', expand = True)[1]
    return df


def get_labels_for_cluster(filePath, clusterLabel):
    df = import_labels(filePath)
    labels = df.loc[df['labels'] == clusterLabel, 'name'].to_list()
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


def data_preparation_wrapper(dataFilePath, password, labelsFilePath, clusterLabel, maxProfileCount):
    df = import_data(dataFilePath, password)
    df = keep_only_full_weeks(df)
    labels = get_labels_for_cluster(labelsFilePath, clusterLabel)
    df_train = create_training_data(df, labels, maxProfileCount)
    df_train = outlier_removal_wrapper(df_train)
    return df_train


def get_categorical_columns(df):
    categoricalCols = [col for col in df.columns if not col.isdigit()]
    df_result = df[categoricalCols].copy()
    df_result.drop(columns = 'timestamp', inplace = True)
    df_result[df_result.columns] = df[df_result.columns].astype('category')
    df_result.reset_index(drop = True, inplace = True)
    return df_result


def df_to_arr(df):
    numericCols = [col for col in df.columns if col.isdigit()]
    df = df[numericCols]
    arr = df.to_numpy()
    return arr


def min_max_scaler(arr, featureRange = FEATURE_RANGE):
    valMin, valMax = np.min(arr), np.max(arr)
    arr_scaled = (arr - valMin)/(valMax - valMin)*(featureRange[1] - featureRange[0]) + featureRange[0]
    return arr_scaled, valMin, valMax


def reshape_arr(arr):
    arr = np.split(arr.T, arr.shape[0]/24, axis = 1)
    arr = np.stack(arr, axis = 2)
    arr = np.split(arr, arr.shape[2]/7, axis = 2)
    arr = np.stack(arr, axis = 3)
    return arr


def gan_input_wrapper(df_train, outputPath):
    X_train = df_to_arr(df_train)
    X_trainProcd = X_train.copy()
    X_trainProcd = reshape_arr(X_trainProcd)
    X_trainProcd, valMin, valMax = min_max_scaler(X_trainProcd) #scale data
    minMax = np.array([valMin, valMax])
    np.save(file = outputPath / 'min_max.npy', arr = minMax)
    return X_trainProcd, X_train, minMax


def revert_reshape_arr(arr):
    arr = np.split(arr, arr.shape[3], axis = 3)
    arr = np.concatenate(arr, axis = 2).squeeze()
    arr = np.split(arr, arr.shape[2], axis = 2)
    arr = np.concatenate(arr, axis = 1).squeeze().T
    return arr


def invert_min_max_scaler(arr_scaled, minMax, featureRange = FEATURE_RANGE):
    valMin, valMax = minMax[0], minMax[1]
    arr = (arr_scaled - featureRange[0])*(valMax - valMin)/(featureRange[1] - featureRange[0]) + valMin #!rounding problem?
    return arr