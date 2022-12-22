import pandas as pd
import pyarrow
import pathlib

path_file = r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop"
data = pd.read_parquet(path_file+"/ENERCOOP_load_profiles.parquet.gzip")

