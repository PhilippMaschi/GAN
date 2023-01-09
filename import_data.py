import pandas as pd
import pyarrow
import pathlib


path_files = r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop"

data_all = pd.read_csv(path_files + "/data_reduced.csv")
data = pd.read_parquet(path_files + "/all_data.gzip")

data_all = pd.read_parquet(path_files+"/ENERCOOP_load_profiles.parquet.gzip")
data_json_all = pd.read_json(r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop/all_load_profiles.json")

data_reduced = data_all.iloc[:100000]
data_reduced.to_csv(path_files + "/data_reduced.csv")

