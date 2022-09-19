import pandas as pd
from pandas_profiling import ProfileReport
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class

path = r'C:\Users\FrancescaConselvan\Desktop\MODERATE\GAN'
def open_csv (path, name):
    data = pd.read_csv(path+name)
    cat_columns = data.select_dtypes(['object']).columns  # 'date', 'consumed energy'
    data[cat_columns] = data[cat_columns].apply(lambda x: pd.factorize(x)[0])
    return data


#pandas profiling
profile = ProfileReport(data, title="Report")
profile.to_file(output_file='reports\pandas_profiling.html')

#sweetViz
report = sv.analyze(data)
report.show_html('reports\sweetViz.html')

#autoViz
AV = AutoViz_Class()
df_av = AV.AutoViz('reports\miss_val.csv')
# not possible to export in HTML????