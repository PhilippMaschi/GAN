import pandas as pd
from pandas_profiling import ProfileReport
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class
from clustering.cluster_load_profiles import ProfileLoader

path = r'C:\Users\FrancescaConselvan\Desktop\MODERATE\GAN'


def data_preparation(path, name):
    data = pd.read_csv(path+name)
    pl = ProfileLoader(path=path)
    data = pl.create_timestamp(df=data)
    data = pl.from_cat_to_num(df=data)
    """" converted obbject columns into numeric otherwise does not work """
    return data


def pandas_profiling(data, report_title, report_name_html):
    profile = ProfileReport(data, title=report_title)
    profile.to_file(output_file=report_name_html)


def sweet_viz(data, report_name_html):
    report = sv.analyze(data)
    report.show_html(report_name_html)


def auto_viz(report_name_csv):
    AV = AutoViz_Class()
    df_av = AV.AutoViz('report_name_csv')
# not possible to export in HTML????



