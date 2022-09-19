import pandas as pd
import os
import glob

from pandas import DataFrame
from pandas_profiling import ProfileReport
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class


class LoadFiles:
    def __init__(self, path):
        self.path = path

    def read_csv(self) -> pd.DataFrame:
        csv_files = glob.glob(os.path.join(self.path, '*.csv'))
        data = pd.concat([pd.read_csv(f, delimiter=';', index_col=None, ) for f in csv_files])
        data.columns = ['date', 'hour', 'consumed energy', 'exported energy',
                        'reactive energy Q1', 'reactive energy Q2', 'reactive energy Q3',
                        'reactive energy Q4', 'contacted power P1', 'contacted power P2',
                        'contacted power P3', 'contacted power P4', 'contacted power P5',
                        'contacted power P6', 'no name']
        data.to_csv('concatenated_csv.csv', index=False)

        cat_columns = data.select_dtypes(['object']).columns  # 'date', 'consumed energy'
        data[cat_columns] = data[cat_columns].apply(lambda x: pd.factorize(x)[0])
        return data


path = r'C:\Users\FrancescaConselvan\Desktop\MODERATE\datasets\Enercoop'

df = LoadFiles(path)
data: DataFrame = df.read_csv()





