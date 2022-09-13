import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from  lets_plot import *
LetsPlot.setup_html()


#TODO: CREATE A CLASS TO IMPORT and CONC DATA
path = r'/Users/francesca/Desktop/e-think/MODERATE/datasets/Enercoop'

column_names = ['date', 'hour', 'consumed energy', 'exported energy',
                    'reactive energy Q1', 'reactive energy Q2', 'reactive energy Q3',
                    'reactive energy Q4', 'contacted power P1', 'contacted power P2',
                    'contacted power P3', 'contacted power P4', 'contacted power P5',
                    'contacted power P6', 'no name']

def combine_csv(path, delim, file_name):
    csv_files =glob.glob(os.path.join(path, '*.csv'))
    data = pd.concat([pd.read_csv(f, delimiter=delim, index_col=None, ) for f in csv_files])
    data.columns = column_names
    data.to_csv (file_name,index=False)
    return data

# categorical variables and conversion


def import_file (file_name):
    data = pd.read_csv(file_name, index_col=0, low_memory=False)
    cat_columns = data.select_dtypes (['object']).columns  # 'date', 'consumed energy'
    data[cat_columns] = data[cat_columns].apply(lambda x: pd.factorize(x)[0])



## DATA EXPLORATION

data_all = pd.read_csv('combined_csv.csv', index_col=0, low_memory=False)

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
data_no_missing_val = data.drop(columns = ['contacted power P2', 'contacted power P3',
                            'contacted power P4', 'contacted power P5',
                            'contacted power P6', 'no name'], axis=1)
data_no_missing_val.to_csv('data_no_miss_val.csv', index=False)

# 0 values
def null_values (df):
    for column_name in df.columns:
        column = df[column_name]
        count = (column==0).sum()
        print('Count of zer0s', column_name, ':', count)

#TODO: dataframe and export to excel


#* OUTLIERS
data.describe()
data = pd.read_csv('data_no_miss_val.csv', index_col=0, low_memory=False)

# boxpots:
def outliers_boxplot(dataset, fig_name):
    fig = plt.figure()
    bp = dataset.plot (kind='box',subplots=True,layout=(5,3),figsize=(10,10))
    plt.savefig(fig_name)
    return plt.show()


data.hist(column='reactive energy Q1')
plt.show()
Q1 = data_clean['reactive energy Q1'].value_counts()
Q2 = data_clean['reactive energy Q2'].value_counts() #two unique values: 0 and 1
ee = data_clean['exported energy'].value_counts()   #two unique values: 0 and 1
unique = pd.concat([Q1, Q2, ee], axis=1)

# unique values frequency:
def count_elements (seq) -> dict:
    """Tally elements from `seq`."""
    hist = {}
    for i in seq:
        hist[i] = hist.get (i,0) + 1
    return hist

## DATA ANALYSIS
#TODO: barplot date and hour
# TODO: snd vs histogram
def univariate_analysis (data, fig_name):
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle('univariate distribution')
    sns.kdeplot(ax=axes[0, 0], data=data['date'])
    sns.kdeplot(ax=axes[0, 1], data=data['hour'])
    sns.kdeplot(ax=axes[0, 2], data=data['consumed energy'])
    sns.kdeplot(ax=axes[1, 0], data=data['exported energy'])
    sns.kdeplot(ax=axes[1, 1], data=data['reactive energy Q1'])
    sns.kdeplot(ax=axes[1, 2], data=data['reactive energy Q2'])
    sns.kdeplot(ax=axes[2, 0], data=data['reactive energy Q3'])
    sns.kdeplot(ax=axes[2, 1], data=data['reactive energy Q4'])
    sns.kdeplot(ax=axes[2, 2], data=data['contacted power P1'])
    plt.savefig(fig_name)
    return plt.show()

#sweetViz
import sweetviz as sv
report = sv.analyze(data)
report.show_html('report.html')

#pandas profiling
from pandas_profiling import ProfileReport
profile = ProfileReport(data, title="Report")
profile.to_file(output_file='profile.html')
profile

#autoViz
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
df_av = AV.AutoViz('data_no_miss_val.csv')


#TODO: bivariate analysis (?? - variable correlation

#TODO: multivariate analysis (????)


## DATA NORMALIZATION

#TODO: normalize data
