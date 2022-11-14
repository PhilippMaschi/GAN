import glob
from pathlib import Path

import pandas as pd


class Data:
    def __init__(self):
        # ENERCOOP
        self.ENERCOOP_path = r'C:\Users\Daniel\Projekte\MODERATE\Data\ENERCOOP'
        self.ENERCOOP_filenames = sorted(glob.glob(self.ENERCOOP_path + '/*.csv'), key = len)   #file paths in a natural sort order
        self.df_ENERCOOP = self.import_and_preprocess_ENERCOOP_files()
    
    def import_and_preprocess_ENERCOOP_files(self):
        dfs = []
        for item in self.ENERCOOP_filenames:
            df_temp = pd.read_csv(
                item,
                sep = ';',
                usecols = list(range(14)),
                decimal = ','
            )
            df_temp.columns = [
                'Date',
                'Hour',
                'Consumed energy [Wh]',
                'Exported energy [Wh]',
                'Reactive energy Q1 [VAh]',
                'Reactive energy Q2 [VAh]',
                'Reactive energy Q3 [VAh]',
                'Reactive energy Q4 [VAh]',
                'Contracted power P1 [kW]',
                'Contracted power P2 [kW]',
                'Contracted power P3 [kW]',
                'Contracted power P4 [kW]',
                'Contracted power P5 [kW]',
                'Contracted power P6 [kW]'
            ]
            df_temp['Date'] = pd.to_datetime(df_temp['Date']) + pd.to_timedelta(df_temp['Hour'], unit = 'h')
            df_temp[['Year', 'Month', 'Day']] = pd.DataFrame([df_temp['Date'].dt.year, df_temp['Date'].dt.month, df_temp['Date'].dt.day]).T.values
            df_temp['Profile'] = int(Path(item).name[:-4])
            cols = df_temp.columns.tolist()
            df_temp = df_temp[cols[-1:] + cols[:1] + cols[-4:-1] + cols[1:-4]]    #rearranges the columns
            dfs.append(df_temp)
        df_ENERCOOP = pd.concat(dfs).reset_index(drop = True)
        return df_ENERCOOP