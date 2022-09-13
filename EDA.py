import pandas as pd
import os
import glob

path = r'/Users/francesca/Desktop/e-think/MODERATE/datasets/Enercoop'
column_names = ['date', 'hour', 'consumed energy', 'exported energy',
                    'reactive energy Q1', 'reactive energy Q2', 'reactive energy Q3',
                    'reactive energy Q4', 'contacted power P1', 'contacted power P2',
                    'contacted power P3', 'contacted power P4', 'contacted power P5',
                    'contacted power P6']

csv_files =glob.glob(os.path.join(path, '*.csv'))

data = pd.concat([pd.read_csv(f, delimiter=';', skiprows=1, names=column_names)
                  for f in csv_files])

data.to_csv("combined_csv.csv")




#TODO: missing vlaues

#TODO: null values

#TODO: outlyers

