import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import sweetviz as sv


########## PROFILES ##########
####################################
def pandas_profiling(data, report_title, path, report_name):
    profile = ProfileReport(data, title=report_title)
    profile.to_file(output_file=path + "/" + report_name + ".html")


def sweet_viz(data, path, report_title):
    report = sv.analyze(data)
    report.show_html(path + "/" + report_title + ".html")


########## MISSING VALUES ##########
####################################
def missing_values(df):
    tot_missing = df.isnull().sum().sort_values(ascending=False)
    percent_missing = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([tot_missing, percent_missing], axis=1, keys=['Total', 'Percent'])
    return missing_data


def drop_missing_values(col_with_missing_val):
    data_no_missing_val = data.drop(columns=col_with_missing_val, axis=1)
    data_no_missing_val.to_csv('data_clean.csv', index=False)
    return data_no_missing_val


def null_values(df):
    for column_name in df.columns:
        column = df[column_name]
        count = (column == 0).sum()
        print('Count of zer0s', column_name, ':', count)


########## OUTLIERS ##########
####################################
def outliers_boxplot(data, path, fig_name):
    fig = plt.figure()
    bp = data.plot(kind='box', subplots=True, layout=(5, 3), figsize=(10, 10))
    plt.savefig(path + "/" + fig_name + ".jpg")
    return plt.show()


if __name__ == "__main__":
  pandas_profiling()
  sweet_viz()
  missing_values()
  null_values()
  outliers_boxplot()