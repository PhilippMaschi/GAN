import pandas as pd
import numpy as np
import csv


def add_zeros_rows(df: pd.DataFrame, dayCount: int) -> pd.DataFrame:
    """Appends additional rows to the input dataframe in the case that the number\n
    of days (which equals the number of rows of `df` divided by 24) is lower than\n
    `dayCount`.

    Args:
        df (pd.DataFrame): A dataframe containing multiple consumption profiles.\n
            Each column should correspond to one profile and contain hourly values\n
            for one year.\n
        dayCount (int): The maximum number of days to be processed by the GAN.\n
            If this value is changed, the layer structure of the Generator and\n
            the Discriminator need to be adapted accordingly. 

    Raises:
        ValueError: An error is produced if the number of days exceeds `dayCount`.

    Returns:
        pd.DataFrame: An updated dataframe with additional rows containing zeros.
    """
    addRowCount = 24*dayCount - df.shape[0]
    if addRowCount < 0:
        raise ValueError(f'The maximum amount of days allowed is {dayCount}!')
    df_zeros = pd.DataFrame(np.zeros((addRowCount, df.shape[1])), columns = df.columns)
    df = pd.concat([df, df_zeros])
    return df


def df_to_arr(df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    """Converts a Pandas DataFrame to a NumPy array.\n
    Preserves the index.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        np.ndarray, pd.Index: The corresponding array and the index of the dataframe.
    """
    dfIdx = df.index
    arr = df.to_numpy()
    return arr, dfIdx


def reshape_arr(arr: np.ndarray, dayCount: int) -> np.ndarray:
    """Reshapes the input array. The resulting array can be viewed as a collection\n
    of heatmaps and serves as the input of the GAN.

    Args:
        arr (np.ndarray): The input array.\n
        dayCount (int): The maximum number of days to be processed by the GAN.

    Returns:
        np.ndarray: The reshaped array. Its shape is\n
        (number of profiles, 1, number of hours of a day, number of days).
    """
    arr = np.stack([col.reshape(dayCount, -1, 1) for col in arr.T], axis = 3).T
    return arr


def revert_reshape_arr(arr: np.ndarray) -> np.ndarray:
    """Reverts the operation of `reshape_arr`.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        np.ndarray: The array with its original shape.
    """
    arr = arr.T.reshape(-1, arr.shape[0])
    return arr


def min_max_scaler(arr: np.ndarray, featureRange: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Scales the values of the input array between a minimum and a maximum given by\n
    `featureRange`.

    Args:
        arr (np.ndarray): The input array.\n
        featureRange (tuple[int, int]): The range used for scaling the data.

    Returns:
        tuple[np.ndarray, np.ndarray]: The resulting array as well as an array containing\n
        the minimum and the maximum value.
    """
    valMin, valMax = np.min(arr), np.max(arr)
    arr_minMax = np.array([valMin, valMax])
    arr_scaled = (arr - valMin)/(valMax - valMin)*(featureRange[1] - featureRange[0]) + featureRange[0]
    return arr_scaled, arr_minMax


def invert_min_max_scaler(arr_scaled: np.ndarray, arr_minMax: np.ndarray, featureRange: tuple[int, int]) -> np.ndarray:
    """Reverts the operation of `min_max_scaler`.

    Args:
        arr_scaled (np.ndarray): The input array.\n
        arr_minMax (np.ndarray): Array containing the minimum and maximum value.\n
        featureRange (tuple[int, int]): The range used for scaling the data.

    Returns:
        np.ndarray: The resulting array.
    """
    valMin, valMax = arr_minMax[0], arr_minMax[1]
    arr = (arr_scaled - featureRange[0])*(valMax - valMin)/(featureRange[1] - featureRange[0]) + valMin
    return arr


def data_prep_wrapper(df: pd.DataFrame, dayCount: int, featureRange: tuple[int, int]) -> tuple:
    """A wrapper calling the following functions in the specified order:
    * `add_zeros_rows`
    * `df_to_arr`
    * `reshape_arr`
    * `min_max_scaler`

    Args:
        df (pd.DataFrame): A dataframe containing multiple consumption profiles.\n
            Each column should correspond to one profile and contain hourly values\n
            for one year.\n
        dayCount (int): The maximum number of days to be processed by the GAN.\n
            If this value is changed, the layer structure of the Generator and\n
            the Discriminator need to be adapted accordingly.\n
        featureRange (tuple[int, int]): The range used for scaling the data.

    Returns:
        tuple[np.ndarray, pd.Index, np.ndarray]: Resulting array, index of the dataframe,\n
        array containing the minimum and the maximum value.
    """
    df = add_zeros_rows(df, dayCount)
    arr, dfIdx = df_to_arr(df)
    arr = reshape_arr(arr, dayCount)
    arr, arr_minMax = min_max_scaler(arr, featureRange)
    return arr, dfIdx, arr_minMax


def get_sep(path):
    """Determines the separator used in a CSV file.

    Args:
        path (str): Path to the file.

    Returns:
        str: The separator.
    """
    with open(path, newline = '') as file:
        sep = csv.Sniffer().sniff(file.read()).delimiter
        return sep


def get_sep_marimo(data):
    """Determines the separator used in a CSV file for the marimo notebook.

    Args:
        data (_io.StringI): Data object from marimo file uploader.

    Returns:
        str: The separator.
    """
    sep = csv.Sniffer().sniff(data.getvalue()).delimiter
    return sep