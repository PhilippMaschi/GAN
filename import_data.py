import pandas as pd
import pyarrow
import pathlib


path_files = r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop"

data = pd.read_parquet(path_files+"/ENERCOOP_load_profiles.parquet.gzip")

df = pd.read_json(r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop/all_load_profiles.json")

data_reduced = data.iloc[:100000]
data_reduced.to_csv(path_files + "/data_reduced.csv")