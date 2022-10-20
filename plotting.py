import plotly.express as px
import numpy as np

def plot_ENERCOOP_data(df_ENERCOOP):
    figs = {}
    for profile in df_ENERCOOP['Profile'].unique():
        df_temp = df_ENERCOOP[df_ENERCOOP['Profile'] == profile].copy()
        df_temp['Date'] = df_temp['Date'].dt.date.astype(str)
        df_temp = df_temp.pivot(index = 'Hour', columns = 'Date', values = 'Load [Wh]')
        figs[f'Profile {str(profile)}'] = px.imshow(df_temp, labels = {'color': 'Load [Wh]'})
    
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

    fig = figs['Profile 1'].update_layout(
        updatemenus = updatemenus,
        title = 'Load profiles'
    )
    fig.update_layout({
        'width': 1366,
        'height': 768
    })
    return fig


def plot_cumulated_explained_variance(pca_fit):
    exp_var_cum = np.cumsum(pca_fit.explained_variance_ratio_)
    fig = px.area(
        x = range(1, exp_var_cum.shape[0] + 1),
        y = exp_var_cum,
        labels = {'x': 'Number of components', 'y': 'Explained variance'},
        title = 'Cumulated explained variance'
    )
    fig.update_layout({
        'width': 720,
        'height': 576
    })
    return fig