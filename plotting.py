import plotly.express as px

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
    fig.write_html('ENERCOOP_load_profiles.html')
    return fig