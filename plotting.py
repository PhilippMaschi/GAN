import matplotlib.pyplot as plt
from datetime import datetime as dt
from math import ceil
from matplotlib.lines import Line2D


def plot_profile(df, profile, x = 'date', custom_title = None):
    fig = plt.figure(figsize = (12, 4), facecolor = 'w')
    plt.plot(df[x], df[profile])
    plt.title(f'Profile {profile}' if not custom_title else custom_title, fontsize = 16)
    plt.xlabel(x.capitalize(), fontsize = 14)
    plt.xticks(fontsize = 12.5)
    plt.ylabel('Consumed energy [Wh]', fontsize = 14)
    plt.yticks(fontsize = 12.5)
    plt.show()  #optional
    plt.close()
    #return fig


def plot_subsequence(df, profile, date):
    temp_df = df[df['date'] == dt.strptime(date, '%Y-%m-%d').date()]
    fig = plot_profile(temp_df, profile, 'hour of the day', f"Profile {profile}\n{temp_df['date'].unique()[0]}")
    #return fig


def visualize_and_analyse_synthetic_profiles(synth_df, real_df, profile, attributes, ncols = 4):
    plotCount = len(synth_df.groupby(attributes).count())
    nrows = ceil(plotCount/ncols)
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (5*ncols, 4*nrows), facecolor = 'w')
    axes_list = axes.reshape(-1)
    axesCount = 0
    for month in synth_df['month of the year'].unique():
        for day_off in synth_df['day off'].unique():
            tempSynth_df = synth_df.query("`month of the year` == @month & `day off` == @day_off").copy()
            if len(tempSynth_df) > 0:
                tempSynth_df['hour of the day'] = tempSynth_df['timestamp'].dt.hour + 1
                tempSynth_df = tempSynth_df.pivot_table(values = profile, index = 'date index', columns = 'hour of the day')
                tempReal_df = real_df.query("`month of the year` == @month & `day off` == @day_off").copy()
                tempReal_df = tempReal_df.pivot_table(values = profile, index = 'date', columns = 'hour of the day')
                title = f'month: {month} | day off: {day_off} | count: {len(tempSynth_df)}'
                tempSynthPlot = tempSynth_df.T.plot(color = 'red', title = title, alpha = 0.5, legend = False, ax = axes_list[axesCount])
                tempRealPlot = tempReal_df.T.plot(legend = False,  color = 'grey', alpha = 0.5, ax = axes_list[axesCount])
                tempRealPlot.set(xlabel = None, ylabel = None)
                axesCount += 1
            else:
                print(f'Missing: month: {month} | day off: {day_off}')
    plt.tight_layout()
    for idx in range(axesCount, nrows*ncols):
        axes_list[idx].axis('off')
    fig.text(0.5, -0.01, 'Hour of the day', ha = 'center', fontsize = 18)
    fig.text(-0.01, 0.5, 'Consumed energy [Wh]', va = 'center', rotation = 'vertical', fontsize = 18)
    legendElements = [
        Line2D([0], [0], marker = 'o', color = 'w', label = 'Synthetic', markerfacecolor = 'red', markersize = 15),
        Line2D([0], [0], marker = 'o', color = 'w', label = 'Real', markerfacecolor = 'grey', markersize = 15)
    ]
    axes_list[0].legend(handles = legendElements, loc = 2);