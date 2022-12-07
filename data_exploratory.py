import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import prepare_data

data = prepare_data.create_timestamp(df=df, date_column="date")


# MISSING VALUES

def missing_values(df):
    tot_missing = df.isnull().sum().sort_values(ascending=False)
    percent_missing = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([tot_missing, percent_missing], axis=1, keys=['Total', 'Percent'])
    return missing_data


def drop_missing_values(col_with_missing_val):
    data_no_missing_val = data.drop(columns=col_with_missing_val, axis=1)
    data_no_missing_val.to_csv('data_clean.csv', index=False)
    return data_no_missing_val


col_miss_val = ['contacted power P2', 'contacted power P3',
                'contacted power P4', 'contacted power P5',
                'contacted power P6', 'no name']


# 0 VALUESs
def null_values(df):
    for column_name in df.columns:
        column = df[column_name]
        count = (column == 0).sum()
        print('Count of zer0s', column_name, ':', count)


# OUTLIERS

# boxpots:
def outliers_boxplot(dataset, fig_name):
    fig = plt.figure()
    bp = dataset.plot(kind='box', subplots=True, layout=(5, 3), figsize=(10, 10))
    plt.savefig(fig_name)
    return plt.show()


# UNIQUE VALUES
# all columns
unique_values = pd.DataFrame(data_clean.nunique())

# specific columns
def count_elements(seq) -> dict:
    """Tally elements from `seq`."""
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist


# * VARIABLE DISTRIBUTION
# TODO: barplot date and hour

def univariate_analysis(data, fig_name):
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
ax = sns.heatmap(corrMatrix, annot=True)
plt.show()


