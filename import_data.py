print ("Start")

import pandas as pd
import pyarrow
import pathlib

path_file = r"C:/Users/FrancescaConselvan/Dropbox/MODERATE/Enercoop"
data = pd.read_parquet(path_file+"/ENERCOOP_load_profiles.parquet.gzip")

data_raw = pd.read_json(path_file+"/all_load_profiles.json")

print ("Stop")