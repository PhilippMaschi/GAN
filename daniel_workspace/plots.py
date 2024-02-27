import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')   #for seaborn
import os


def plot_losses(df, plotPath):
    df.reset_index().plot(x = 'index', y = ['loss_discriminator_real', 'loss_discriminator_fake', 'loss_generator'], alpha = 0.5, figsize = (12, 5))
    plt.title('Training losses')
    plt.xlabel('epoch × batch_index')
    plt.ylabel('value')
    plt.ylim(0, 10)
    plt.savefig(plotPath / 'training_losses.png')
    plt.close();


def compare_distributions(X_real, X_synth, plotPath):
    plt.figure(figsize = (8, 5), facecolor = 'w')
    plt.hist(X_real.flatten(), bins = 100, alpha = 0.5, label = 'Real', color = 'aqua')
    plt.hist(X_synth.flatten(), bins = 100, alpha = 0.5, label = 'Synthetic', color = 'hotpink')
    plt.title('Comparison of the distributions of the values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotPath / 'load_distrib.png')
    plt.close();


def plot_seasonal_daily_means(X_real, X_synth, df_hull, plotPath):
    meansReal, stdsReal = get_seasonal_hourly_means_and_stds(X_real, df_hull)
    meansSynth, stdsSynth = get_seasonal_hourly_means_and_stds(X_synth, df_hull)
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']    #alternative: `df_hull['meteorological season'].cat.categories`
    hours = range(1, 25)

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 6), sharey = True)
    axes = axes.flatten()
    for idx, season in enumerate(seasons):
        meansRealTemp = meansReal[meansReal.index.get_level_values('meteorological season') == season]
        stdsRealTemp = stdsReal[stdsReal.index.get_level_values('meteorological season') == season]
        meansSynthTemp = meansSynth[meansSynth.index.get_level_values('meteorological season') == season]
        stdsSynthTemp = stdsSynth[stdsSynth.index.get_level_values('meteorological season') == season]

        axes[idx].plot(hours, meansRealTemp, color = 'b', alpha = 0.5, linewidth = 3, label = 'Mean real')
        axes[idx].fill_between(hours, meansRealTemp - stdsRealTemp, meansRealTemp + stdsRealTemp, color = 'aqua', alpha = 0.3, label = 'Std real')
        axes[idx].plot(hours, meansSynthTemp, color = 'r', alpha = 0.5, linewidth = 3, label = 'Mean synth')
        axes[idx].fill_between(hours, meansSynthTemp - stdsSynthTemp, meansSynthTemp + stdsSynthTemp, color = 'hotpink', alpha = 0.3, label = 'Std synth')
        axes[idx].set_title(f'{season}')
        axes[idx].set_xlabel('Hour of the day')
        axes[idx].set_ylabel('Mean value')
        axes[idx].set_xlim(hours.start, hours.stop - 1)
        axes[idx].legend()
    plt.suptitle('Mean day for each season')
    plt.tight_layout()
    plt.savefig(plotPath / 'seasons_mean_day.png')
    plt.close();


def get_seasonal_hourly_means_and_stds(X, df_hull):
    df = pd.DataFrame(X)
    df = pd.concat([df_hull, df], axis = 1)
    df.set_index(list(df_hull.columns), inplace = True)
    df = df.groupby(['meteorological season', 'hour of the day'], observed = False).mean()
    means = df.mean(axis = 1)
    stds = df.std(axis = 1)
    return means, stds


def plot_peaks(X_real, X_synth, plotPath):
    peaksReal = X_real.max(axis = 0)
    peaksSynth = X_synth.max(axis = 0)

    plt.figure(figsize = (8, 5), facecolor = 'w')
    plt.plot(peaksReal, label = 'Real peaks', color = 'blue', alpha = 0.5)
    plt.plot(peaksSynth, label = 'Synthetic peaks', color = 'red', alpha = 0.5)
    plt.title(f'Comparison of peak values')
    plt.xlabel('Profile Index')
    plt.ylabel('Value')
    plt.xticks([])
    plt.legend()
    plt.savefig(plotPath / 'peaks.png')
    plt.close();


def plot_means(X_real, X_synth, plotPath):
    meansReal = X_real.mean(axis = 0)
    meansSynth = X_synth.mean(axis = 0)

    plt.figure(figsize = (8, 5), facecolor = 'w')
    plt.plot(meansReal, label = 'Real means', color = 'blue', alpha = 0.5)
    plt.plot(meansSynth, label = 'Synthetic means', color = 'red', alpha = 0.5)
    plt.title(f'Comparison of mean values')
    plt.xlabel('Profile Index')
    plt.ylabel('Value')
    plt.xticks([])
    plt.legend()
    plt.savefig(plotPath / 'means.png')
    plt.close();


def plot_examples(X, plotPath, Xtype: ['real', 'synth']):
    fig, axes = plt.subplots(nrows = 5, ncols = 3, figsize = (12, 10), sharey = True)
    axes = axes.flatten()
    for i in range(15):
        axes[i].imshow(X[:, i].reshape(-1, 24*7), cmap = 'hot')
        axes[i].set_xlabel('Hour of the week')
        axes[i].set_ylabel('Week')
    plt.suptitle(f'Heatmaps of example profiles ({Xtype})')
    plt.tight_layout()
    plt.savefig(plotPath / f'{Xtype}_example_profiles.png')
    plt.close();


def plot_average_week(X_real, X_synth, df_hull, plotPath):
    df_real = get_seasonal_daily_means(X_real, df_hull)
    df_real['type'] = 'real'
    df_synth = get_seasonal_daily_means(X_synth, df_hull)
    df_synth['type'] = 'synth'
    df_plot = pd.concat([df_real, df_synth])
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']    #alternative: `df_hull['meteorological season'].cat.categories`
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 8), sharey = True)
    axes = axes.flatten()
    for idx, season in enumerate(seasons):
        sns.boxplot(data = df_plot[df_plot['meteorological season'] == season], x = 'weekday', y = 'value', hue = 'type', order = weekdays, ax = axes[idx], showfliers = False)
    plt.suptitle('Mean week for each season')
    plt.tight_layout()
    plt.savefig(plotPath / 'seasons_average_week.png')
    plt.close();


def get_seasonal_daily_means(X, df_hull):
    df = pd.DataFrame(X)
    df = pd.concat([df_hull, df], axis = 1)
    df.set_index(list(df_hull.columns), inplace = True)
    df = df.groupby(['meteorological season', 'weekday'], observed = False).mean()
    df = df.melt(ignore_index = False, var_name = 'profile number').reset_index()
    return df


def plot_wrapper(X_real, X_synth, df_hull, runPath):
    plotPath = runPath / 'plots'
    os.makedirs(plotPath) if not os.path.exists(plotPath) else None
    compare_distributions(X_real, X_synth, plotPath)
    plot_seasonal_daily_means(X_real, X_synth, df_hull, plotPath)
    plot_peaks(X_real, X_synth, plotPath)
    plot_means(X_real, X_synth, plotPath)
    plot_examples(X_real, plotPath, Xtype = 'real')
    plot_examples(X_synth, plotPath, Xtype = 'synth')
    plot_average_week(X_real, X_synth, df_hull, plotPath)