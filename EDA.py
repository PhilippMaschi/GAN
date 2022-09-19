import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# MISSING VALUES
# Missing values
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
data_no_missing_val.to_csv('data_clean.csv', index=False)


#* 0 values
def null_values (df):
    for column_name in df.columns:
        column = df[column_name]
        count = (column==0).sum()
        print('Count of zer0s', column_name, ':', count)


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
# all columns
unique_values = pd.DataFrame(data_clean.nunique())

# specific columns
def count_elements (seq) -> dict:
    """Tally elements from `seq`."""
    hist = {}
    for i in seq:
        hist[i] = hist.get (i,0) + 1
    return hist

#* VARIABLE DISTRIBUTION
#TODO: barplot date and hour

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


# variable correlation
f = plt.figure(figsize=(19, 15))
corrMatrix = data.corr()
ax = sns.heatmap (corrMatrix, annot=True)
plt.show()

#TODO: multivariate analysis (????)


## DATA NORMALIZATION

#TODO: normalize data
