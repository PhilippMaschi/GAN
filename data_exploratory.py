import pandas as pd
import pyarrow
import pathlib

from pandas_profiling import ProfileReport
import sweetviz as sv

import matplotlib.pyplot as plt


class Profiles:

    def __init__(self, filename, load_profile_carrier):
        self.df = pd.read_parquet(filename)
        self.report_path = r"C:/Users/FrancescaConselvan/Documents/MODERATE/results/EDA reports"
        self.figure_path = r"C:/Users/FrancescaConselvan/Documents/MODERATE/results/figures"
        self.title = load_profile_carrier
        print(self.df.describe())

    def pandas_profiling(self):
        pandas_profile = ProfileReport(self.df, title=self.title)
        pandas_profile.to_file(output_file=self.report_path + "/" + self.title + ".html")
        return pandas_profile

    def sweet_viz_report(self):
        sweetviz_report = sv.analyze(self.df)
        sweetviz_report.show_html(self.report_path + "/" + self.title + ".html")
        return sweetviz_report

    def outliers_boxplot(self):
        fig = plt.figure()
        box_plot = self.df.plot(kind="box", subplots=True, layout=(5, 3), figsize=(10, 10))
        plt.savefig(self.figure_path + "/" + self.title + ".jpg")
        return plt.show()

    def missing_values(self):
        tot_missing = self.df.isnull().sum().sort_values(ascending=False)
        percent_missing = (self.df.isnull().sum() / self.df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([tot_missing, percent_missing], axis=1, keys=['Total', 'Percent'])
        missing_data.to_excel(self.report_path + "/" + self.title +".xlsx" )
        return missing_data


if __name__ == "__main__":
    Profiles()

