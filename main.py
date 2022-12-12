import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import sweetviz as sv

from import_data import data
import data_exploratory as de

path_2_file = r"C:/Users/FrancescaConselvan/Documents/MODERATE/results"
de.pandas_profiling(data=data,
                    report_title="Enercoop",
                    path = path_2_file,
                    report_name="Enercoop_pandas")


de.sweet_viz(data=data,
             path=path_2_file,
             report_title="Enercoop_sweetviz")


de.outliers_boxplot(data=data,
                    path = path_2_file,
                    fig_name="Enercoop_outliers")




