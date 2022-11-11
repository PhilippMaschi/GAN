import numpy as np

from config import Config
import pandas as pd
import os
from typing import List
from pathlib import Path


class DataImporter:
    def __init__(self):
        self.path2data = Config().path_2_data / "raw_data"
        self.save_results = Config().path_2_data
        self.global_user_id = []

    def get_csv_names(self) -> List[str]:
        csv_names = os.listdir(self.path2data.absolute())
        return csv_names

    def rearange_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """ brings the electricity data in column format with the ID in the column name"""
        ids = np.unique(table.loc[:, "User ID"].to_numpy())
        first_id = ids[0]
        base_table = table.loc[table.loc[:, "User ID"] == first_id, ["Date"]].reset_index(drop=True)
        for id in ids:
            base_table.loc[:, f"{id}"] = table.loc[table.loc[:, "User ID"] == id, ["Consumed Energy (Wh)"]].reset_index(drop=True)
            # base_table.loc[:, f"Exported Energy (Wh) {id}"] = table.loc[table.loc[:, "User ID"] == id, ["Exported Energy (Wh)"]].reset_index(drop=True)
            # TODO create a table for Exported Energy and drop the households that have no PV from that
        return base_table

    def read_all_load_profiles(self, csv_names: List[str]) -> pd.DataFrame:
        """ reads all load profiles in the raw data folder and returns them in one big table """
        for i, name in enumerate(csv_names):
            file = Path(self.path2data) / Path(name)
            load = self.read_load_profile(file)
            # bring the table from long format into wide:
            load_in_columns_df = self.rearange_table(load)
            if i == 0:  # in the first loop the dataframe is not merged
                # rename columns
                base_table = load_in_columns_df
                base_time = base_table.loc[:, "Date"]
            else:  # dataframe is merged with the old base_table
                # rename columns
                df = load_in_columns_df.drop(columns=["Date"])
                # check if datetime is identical
                assert (base_time == load_in_columns_df.loc[:, "Date"]).any(), f"datetime does not match in file {name}"
                # merge the frames (pandas merge is not possible because of memory timeout when frames are large...)
                base_table = pd.concat([base_table, df], axis=1)

                print(f"added {name} to df")

        # check if nan are in the base table if the datetimes of the dataframes are not the same
        if base_table.isna().all().all():
            print("NAN in dataframe!")
        return base_table

    def check_user_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """ checks if the ID is already in one of the prior dataframes and if yes removes the corresponding rows from
        the current one"""
        unique_ids = np.unique(df.loc[:, "User ID"].to_numpy())
        # compare the new ids with the IDs already imported:
        for id in unique_ids:
            if id in self.global_user_id:
                # id should not be appended
                df = df.drop(df.loc[df.loc[:, "User ID"] == id, :].index)
            else:
                self.global_user_id.append(id)
        return df

    def read_load_profile(self, path2file: "Path") -> pd.DataFrame:
        """ loads an ENERCOOP profile and changes the date to a datetime object and drops useless columns"""
        load_df = pd.read_csv(path2file, decimal=",").dropna(how="all", axis=1)
        df1 = DataImporter().drop_useless_columns(load_df)
        df2 = self.check_user_ids(df1)
        load = DataImporter().create_timestamp(df2)
        return load

    @staticmethod
    def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
        """ drop tje columns which are irrelevant to us (R1, R2, R3...)"""
        columns2keep = ["Date", "Hour", "Consumed Energy (Wh)", "Exported Energy (Wh)", "User ID"]
        return df.loc[:, columns2keep]

    @staticmethod
    def create_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """creates a timestamp in the 'date' column so we can cluster the data more easily"""
        df.loc[:, "Date"] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
        df = df.drop(columns="Hour")
        return df

    def main(self, create_json: bool = False) -> pd.DataFrame:
        """ provides a pandas table with all the profiles from ENERCOOP"""
        if create_json:
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
    DataImporter().main(create_json=True)

