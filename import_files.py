import pandas as pd
import os
import glob

class LoadFiles:
    def __init__(self, path):
        self.path = path

    def read_csv(self) -> pd.DataFrame:
        csv_files = glob.glob (os.path.join (self.path,'*.csv'))
        data = pd.concat ([pd.read_csv (f,delimiter=';',index_col=None,) for f in csv_files])
        data.columns = ['date', 'hour', 'consumed energy', 'exported energy',
                    'reactive energy Q1', 'reactive energy Q2', 'reactive energy Q3',
                    'reactive energy Q4', 'contacted power P1', 'contacted power P2',
                    'contacted power P3', 'contacted power P4', 'contacted power P5',
                    'contacted power P6', 'no name']
        data.to_csv ('concatenated_csv.csv',index=False)
        return data


r'/Users/francesca/Desktop/e-think/MODERATE/datasets/Enercoop'