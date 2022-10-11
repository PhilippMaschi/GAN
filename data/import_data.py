from config import Config
import pandas as pd
import os
from typing import List
from pathlib import Path


class DataImporter:
    def __init__(self):
        self.path2data = Config().path_2_data / "raw_data"
        self.save_results = Config().path_2_data

    def get_csv_names(self) -> List[str]:
        csv_names = os.listdir(self.path2data.absolute())
        return csv_names

    def rename_columns_with_filename(self, df: pd.DataFrame, filename: str):
        return df.rename(columns={"A+(Wh)": f"{filename.replace('.csv', '')}_A+(Wh)",
                                  "A-(Wh)": f"{filename.replace('.csv', '')}_A-(Wh)"})

    def read_all_load_profiles(self, csv_names: List[str]) -> pd.DataFrame:
        """ reads all load profiles in the raw data folder and returns them in one big table """
        for i, name in enumerate(csv_names):
            file = Path(self.path2data) / Path(name)
            load = self.read_load_profile(file)
            if i == 0:  # in the first loop the dataframe is not merged
                # rename columns
                base_table = self.rename_columns_with_filename(load, name)
                base_time = base_table.loc[:, "date"]
            else:  # dataframe is merged with the old base_table
                # rename columns
                df = self.rename_columns_with_filename(load, name)
                # check if datetime is identical
                assert (base_time == df.loc[:, "date"]).any(), f"datetime does not match in file {name}"
                # merge the frames (pandas merge is not possible because of memory timeout when frames are large...)
                base_table.loc[:, f"{name.replace('.csv', '')}_A+(Wh)"] = df.loc[
                                                                          :, f"{name.replace('.csv', '')}_A+(Wh)"]
                base_table.loc[:, f"{name.replace('.csv', '')}_A-(Wh)"] = df.loc[:,
                                                                          f"{name.replace('.csv', '')}_A-(Wh)"]

                print(f"added {name} to df")

        # check if nan are in the base table if the datetimes of the dataframes are not the same
        if base_table.isna().all().all():
            print("NAN in dataframe!")
        return base_table

    @staticmethod
    def read_load_profile(path2file: "Path") -> pd.DataFrame:
        """ loads an ENERCOOP profile and changes the date to a datetime object and drops useless columns"""
        load_df = pd.read_csv(path2file, sep=";", decimal=",").dropna(how="all", axis=1)
        load = DataImporter().drop_useless_columns(load_df)
        df = DataImporter().rename_date_columns(load)
        df_1 = DataImporter().create_timestamp(df)
        return df_1

    @staticmethod
    def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
        """ drop tje columns which are irrelevant to us (R1, R2, R3...)"""
        columns2drop = ["R1(VAh)", "R2(VAh)", "R3(VAh)", "R4(VAh)", "P1", "P2"]
        return df.drop(columns=columns2drop)

    @staticmethod
    def rename_date_columns(df: pd.DataFrame) -> pd.DataFrame:
        """ rename the date columns """
        to_rename = {"FECHA(YYYY-MM-DD o DD/MM/YYY)": "date",
                     "HORA(h)": "hour"}
        return df.rename(columns=to_rename)

    @staticmethod
    def create_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """creates a timestamp in the 'date' column so we can cluster the data more easily"""
        df.loc[:, "date"] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h')
        df = df.drop(columns="hour")
        return df

    def main(self, load_json: bool = False) -> pd.DataFrame:
        """ provides a pandas table with all the profiles from ENERCOOP"""
        if not load_json:
            all_profiles = self.read_all_load_profiles(self.get_csv_names())
            # save the big "all_profiles" table to a json so it can be used much faster in other scripts:
            all_profiles.to_json(self.save_results / "all_load_profiles.json")
            print("saved json file")
        else:
            # else read the json which will be much faster than reading from csv files

            # check if file is there:
            if (self.save_results / "all_load_profiles.json").exists():
                all_profiles = pd.read_json(self.save_results / "all_load_profiles.json")
            else:  # if its not there proceed to generate it
                all_profiles = self.read_all_load_profiles(self.get_csv_names())
                # save the big "all_profiles" table to a json so it can be used much faster in other scripts:
                all_profiles.to_json(self.save_results / "all_load_profiles.json")
                print("saved json file")

        return all_profiles


if __name__ == "__main__":
    DataImporter().main(load_json=False)

