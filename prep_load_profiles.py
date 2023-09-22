import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

def handle_single_entry_array(val):
    if isinstance(val, (list, np.ndarray)) and len(val) == 1:
        return float(str(val[0]).replace(',', '.'))
    elif isinstance(val, str):
        return float(val.replace(',', '.'))
    return val


def handle_dst(df: pd.DataFrame) -> pd.DataFrame:
    date_column = [col for col in df.columns if "FECHA" in col][0]
    # add hour to timestamp
    df["date"] = pd.to_datetime(df[date_column]).dt.date
    df["hour"] = df.groupby("date").cumcount()
    df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')

    # find day with 25 hours:
    days_with_25_hours = df[df["hour"] == 24]["date"].values
    days_with_23_hours = df.groupby("date").size()[df.groupby("date").size() == 23].index.values

    for day in days_with_25_hours:
        time_3am = pd.Timestamp(day) + pd.to_timedelta('03:00:00')
        time_4am = pd.Timestamp(day) + pd.to_timedelta('04:00:00')
        mean = df.loc[
            df['timestamp'].isin([time_3am, time_4am]), ['A+(Wh)', 'A-(Wh)']
        ].applymap(handle_single_entry_array).applymap(pd.to_numeric).mean()
        df.loc[df['timestamp'].isin([time_3am]), ['A+(Wh)', 'A-(Wh)']] = mean
        # Drop the 4 o'clock hour
        df = df[df['timestamp'] != time_4am]

    # Handle days with 23 hours
    for day in days_with_23_hours:
        # Create an additional hour at 3 o'clock in the morning with NaN value (you can adjust this)
        additional_row = pd.DataFrame.from_dict({
            'timestamp': pd.Timestamp(day) + pd.to_timedelta('03:00:00'),
            'A+(Wh)': np.nan,
            'A-(Wh)': np.nan
        }, orient="index").T
        # add new row
        df = pd.concat([df, additional_row], ignore_index=True)
    # Sort dataframe
    df = df.sort_values('timestamp').reset_index(drop=True)
    # bfill the nan values and recreate the hours column, rewrite the timestamp and check if it works now:
    df = df.fillna(method="bfill").drop(columns=["hour"])
    df["hour"] = df.groupby("date").cumcount()
    assert not (df["hour"] == 24).any(), "double dates still exist"
    # overwrite timestamp
    df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    return df


def read_df(file_path: Path) -> pd.DataFrame:
    number = file_path.name.replace(".csv", "")
    dataframe = pd.read_csv(file_path, sep=";")

    corrected_df = handle_dst(dataframe)
    corrected_df = corrected_df.rename(columns={
            "A+(Wh)": f"A+(Wh)_{number}",
            "A-(Wh)": f"A-(Wh)_{number}"
        })
    corrected_df.set_index("timestamp", inplace=True)
    df_return = corrected_df[[f"A+(Wh)_{number}", f"A-(Wh)_{number}"]]
    return df_return


file_path = Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\MODERATE\data\Load Profiles 5000")
# dataframes = []
diff_timestamp = []

csv_files = file_path.glob('*.csv')
# Use all available cores
dataframes = Parallel(n_jobs=-1)(delayed(read_df)(file) for file in tqdm(csv_files))

# for i, csv_file in enumerate(tqdm(csv_files)):
#     df = read_df(csv_file)
#     dataframes.append(df)

big_frame = pd.concat(dataframes, axis=1, join="outer").sort_values(by="timestamp")
# replace , with . to be able to change to numeric
big_frame = big_frame.applymap(
    lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x
).applymap(pd.to_numeric)

# save dataframe
big_frame.to_csv(file_path.parent / "all_profiles.csv")
big_frame.to_parquet(file_path.parent / "all_profiles.gzip.parquet")
