import pandas as pd
import pyarrow
import pathlib


path_files = r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop"

data = pd.read_parquet(path_files+"/ENERCOOP_load_profiles.parquet.gzip")


