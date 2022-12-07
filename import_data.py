print ("Start")

import pandas as pd

path_file = r"C:/Users/FrancescaConselvan/OneDrive - e-think energy research/MODERATE/datasets/Enercoop"

data = pd.read_json(path_file+"/all_load_profiles.json")
