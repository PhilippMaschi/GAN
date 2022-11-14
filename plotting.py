import plotly.express as px
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np


def plot_ENERCOOP_data(df_ENERCOOP, param, title, width = 1366, height = 768):
    figs = {}
    for profile in df_ENERCOOP['Profile'].unique():
        df_temp = df_ENERCOOP[df_ENERCOOP['Profile'] == profile].copy()
        df_temp['Date'] = df_temp['Date'].dt.date.astype(str)
        df_temp = df_temp.pivot(index = 'Hour', columns = 'Date', values = param)
        figs[f'Profile {str(profile)}'] = px.imshow(df_temp, labels = {'color': param})
    
    updatemenus = [{
        'buttons': [],
        'direction': 'down',
        'showactive': True
    }]
    for key in figs:
        updatemenus[0]['buttons'].append({
            'method': 'restyle',
            'label': key,
            'args': [{
                'x': [dat.x for dat in figs[key].data],
                'y': [dat.y for dat in figs[key].data],
                'z': [dat.z for dat in figs[key].data]
            }]
        })

    fig = figs[list(figs.keys())[0]].update_layout(
        updatemenus = updatemenus,
        title = title
    )
    fig.update_layout({
        'width': width,
        'height': height
    })
    return fig


def plot_HCA(Z, df, width = 9, height = 6):
    fig = plt.figure(figsize = (width, height))
    dendrogram(Z, labels = df.index, leaf_rotation = 0)
    plt.title('Hierarchical clustering - dendrogram')
    plt.xlabel('cluster')
    plt.ylabel('distance')
    plt.close()
    return fig


def plot_ellbow_method(df, k_opt_dict, width = 1280, height = 720):
    figs = {}
    for criterion in k_opt_dict.keys():
        fig_temp = px.line(df, x = 'k', y = criterion, markers = True)
        fig_temp.add_vline(x = k_opt_dict[criterion], line_dash = 'dash', annotation_text = f'k_opt = {k_opt_dict[criterion]}')
        fig_temp.update_layout({
            'xaxis': {
                'range': [0.5, max(df['k']) + 0.5],
                'tick0': 1,
                'dtick': 1
            }
        })
        figs[criterion.capitalize()] = fig_temp

    updatemenus = [{
        'buttons': [],
        'direction': 'down',
        'showactive': True
        }]
    for key in figs:
        updatemenus[0]['buttons'].append({
            'method': 'update',
            'label': key,
            'args': [{
                'hovertemplate': figs[key].data[0].hovertemplate,
                'x': [dat.x for dat in figs[key].data],
                'y': [dat.y for dat in figs[key].data],
            }, {
                'annotations': figs[key].layout.annotations,
                'shapes': figs[key].layout.shapes,
                'yaxis': figs[key].layout.yaxis,
            }],
        })

    fig = figs['Distortion'].update_layout(
        updatemenus = updatemenus,
        title = 'Ellbow method'
    )
    fig.update_layout({
        'width': width,
        'height': height
    })
    return fig


def plot_silhouette_method(df, width = 1280, height = 720):
    fig = px.line(df, x = 'k', y = 'silhouette score', title = 'Silhouette method', markers = True)
    fig.update_layout({
        'xaxis': {
            'range': [1.5, max(df['k']) + 0.5],
            'tick0': 1,
            'dtick': 1
        },
        'width': width,
        'height': height
    })
    return fig


def plot_HCA_w_DTW(model):
    f, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 10))
    ax[0].set_title('Dendrogram')
    ax[1].set_title('Time series')
    fig = model.plot(
        axes = ax,
        ts_label_margin = -500,
        show_ts_label = True,
        show_tr_label = True
    )
    return fig


def plot_cumulated_explained_variance(pca_fit, width = 720, height = 576):
    exp_var_cum = np.cumsum(pca_fit.explained_variance_ratio_)
    fig = px.area(
        x = range(1, exp_var_cum.shape[0] + 1),
        y = exp_var_cum,
        labels = {'x': 'Number of components', 'y': 'Explained variance'},
        title = 'Cumulated explained variance'
    )
    fig.update_layout({
        'width': width,
        'height': height
    })
    return fig