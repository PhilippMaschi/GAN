import pandas as pd
from import_data import data
import matplotlib.pyplot as plt

class Univariate_analysis:

    def __init__(self, filename, load_profile_carrier):
        self.df = pd.read_parquet(filename)
        self.figure_path = r"C:/Users/FrancescaConselvan/Documents/MODERATE/results/figures"



g = sns.Facet

# specific columns
def count_elements(seq) -> dict:
    """Tally elements from `seq`."""
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist





# variable correlation
f = plt.figure(figsize=(19, 15))
corrMatrix = data.corr()
ax = sns.heatmap(corrMatrix, annot=True)
plt.show()
