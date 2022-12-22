import pandas as pd
import pyarrow
import pathlib

path_file = r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop"
data = pd.read_parquet(path_file+"/ENERCOOP_load_profiles.parquet.gzip")

df = pd.read_json(r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop/all_load_profiles.json")

