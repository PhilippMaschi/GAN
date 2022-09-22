import pandas as pd
from pandas import DataFrame
import os
import glob


class FileLoader:
    def __init__(self, path):
        self.path = path
        self.data = None

    def read_csv(self) -> pd.DataFrame:
        self.load_data()
        self.categorical()
        return self.data

    def load_data(self, file_name, column_list=[]) -> pd.DataFrame:
        csv_files = glob.glob(os.path.join(self.path, '*.csv'))
        self.data = pd.concat([pd.read_csv(f, delimiter=';', index_col=None, ) for f in csv_files])
        self.data.columns = column_list
        self.data.to_csv(file_name, index=False)

    def categorical(self):
        cat_columns = self.data.select_dtypes(['object']).columns  # 'date', 'consumed energy'
        self.data[cat_columns] = self.data[cat_columns].apply(lambda x: pd.factorize(x)[0])


if __name__ == "__main__":
    path = r'C:\Users\FrancescaConselvan\Desktop\MODERATE\datasets\Enercoop'
    df = FileLoader(path)
    data: DataFrame = df.refine_data(column_list=df)
    print(df.data)

csv_file = FileLoader(path)

path_FC = r'C:\Users\FrancescaConselvan\Desktop\MODERATE\datasets\Enercoop'
columns_name = ['date', 'hour', 'consumed energy', 'exported energy',
                'reactive energy Q1', 'reactive energy Q2', 'reactive energy Q3',
                'reactive energy Q4', 'contacted power P1', 'contacted power P2',
                'contacted power P3', 'contacted power P4', 'contacted power P5',
                'contacted power P6', 'no name']
