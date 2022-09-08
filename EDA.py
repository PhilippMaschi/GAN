import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt


#TODO: CREATE A CLASS TO IMPORT and CONC DATA
path = r'/Users/francesca/Desktop/e-think/MODERATE/datasets/Enercoop'

column_names = ['date', 'hour', 'consumed energy', 'exported energy',
                    'reactive energy Q1', 'reactive energy Q2', 'reactive energy Q3',
                    'reactive energy Q4', 'contacted power P1', 'contacted power P2',
                    'contacted power P3', 'contacted power P4', 'contacted power P5',
                    'contacted power P6', 'no name']

def combine_csv(path, delim):
    csv_files =glob.glob(os.path.join(path, '*.csv'))
    data = pd.concat([pd.read_csv(f, delimiter=delim, index_col=None, ) for f in csv_files])
    data.columns = column_names
    return data

data.to_csv("combined_csv.csv")
data.to_excel('combined_csv.xlsx')

#TODO: convert 'imported energy' from object > int + delete the uname column

## DATA EXPLORATION
#TODO: CREATE A CLASS TO DATA EXPLORATORY

# Missing values
# count the number or missing values and the percentage
def missing_values (df):
    tot_missing = df.isnull().sum().sort_values(ascending=False)
    percent_missing = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data  = pd.concat([tot_missing, percent_missing], axis=1, keys=['Total', 'Percent'])
    return missing_data

missing_data.to_excel('missing_data.xlsx')

# dealing with missing data.
# Since there are only 4 columns with missing data (100%) > dropped them
data_ohne_mv = data.drop(columns = ['contacted power P2', 'contacted power P3',
                            'contacted power P4', 'contacted power P5',
                            'contacted power P6'], axis=1)

# 0 values
def null_values (df):
    for column_name in df.columns:
        column = df[column_name]
        count = (column==0).sum()
        print('Count of zer0s', column_name, ':', count)

#TODO: dataframe and export to excel

# Outlayers

# boxpots:
fig = plt.figure()
bp = data.boxplot()
plt.show()

fig = plt.figure()
bp = data.plot (kind='box',subplots=True,layout=(5,3),figsize=(10,10))
plt.savefig('test.png')
plt.show()

# TODO: scatterplots




## DATA ANALYSIS
#TODO: univariate analysis - variable distribution

#TODO: bivariate analysis (????) - variable correlation

#TODO: multivariate analysis (????)


## DATA NORMALIZATION


#TODO: normalize data
