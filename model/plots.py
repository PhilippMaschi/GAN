import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.ioff()


def plot_losses(df, plotPath):
    df = df[~df['epoch'].duplicated(keep = 'last')]
    df = df.reset_index()
    df.plot(x = 'index', y = ['loss_discriminator_real', 'loss_discriminator_fake', 'loss_generator'], alpha = 0.5, figsize = (12, 5))
    plt.title('Training losses')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.ylim(0, df[['loss_discriminator_real', 'loss_discriminator_fake', 'loss_generator']].max().max()*1.1)
    plt.savefig(plotPath / 'losses.png')
    plt.close()


def compare_distributions(X_real, X_synth, plotPath = None):
    fig = plt.figure(figsize = (8, 5), facecolor = 'w')
    plt.hist(np.array(X_real).flatten(), bins = 100, alpha = 0.5, label = 'Real', color = 'aqua')
    plt.hist(np.array(X_synth).flatten(), bins = 100, alpha = 0.5, label = 'Synthetic', color = 'hotpink')
    plt.title('Comparison of the distributions of the values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    if plotPath:
        plt.savefig(plotPath / 'load_distrib.png')
    plt.close()
    return fig


def plot_means(X_real, X_synth, plotPath = None):
    meansReal = X_real.mean(axis = 0)
    meansSynth = X_synth.mean(axis = 0)
    fig, ax = plt.subplots(figsize = (8, 5), facecolor = 'w')
    ax.boxplot([meansReal, meansSynth])
    ax.set_xticklabels(['Real means', 'Synthetic means'])
    plt.title(f'Comparison of mean values')
    plt.ylabel('Value')
    if plotPath:
        plt.savefig(plotPath / 'means.png')
    plt.close()
    return fig


def plot_stds(X_real, X_synth, plotPath = None):
    stdsReal = X_real.std(axis = 0)
    stdsSynth = X_synth.std(axis = 0)
    fig, ax = plt.subplots(figsize = (8, 5), facecolor = 'w')
    ax.boxplot([stdsReal, stdsSynth])
    ax.set_xticklabels(['Real std', 'Synthetic std'])
    plt.title(f'Comparison of standard deviation values')
    plt.ylabel('Value')
    if plotPath:
        plt.savefig(plotPath / 'stds.png')
    plt.close()
    return fig


def plot_medians(X_real, X_synth, plotPath = None):
    mediansReal = np.median(X_real, axis = 0)
    mediansSynth = np.median(X_synth, axis = 0)
    fig, ax = plt.subplots(figsize = (8, 5), facecolor = 'w')
    ax.boxplot([mediansReal, mediansSynth])
    ax.set_xticklabels(['Real medians', 'Synthetic medians'])
    plt.title(f'Comparison of median values')
    plt.ylabel('Value')
    if plotPath:
        plt.savefig(plotPath / 'medians.png')
    plt.close()
    return fig


def plot_skews(X_real, X_synth, plotPath = None):
    skewsReal = 3*(X_real.mean(axis = 0) - np.median(X_real, axis = 0))/(X_real.std(axis = 0) + 10*(-20))
    skewsSynth = 3*(X_synth.mean(axis = 0) - np.median(X_synth, axis = 0))/X_synth.std(axis = 0)
    fig, ax = plt.subplots(figsize = (8, 5), facecolor = 'w')
    ax.boxplot([skewsReal, skewsSynth])
    ax.set_xticklabels(['Real skews', 'Synthetic skews'])
    plt.title(f'Comparison of skewness values')
    plt.ylabel('Value')
    if plotPath:
        plt.savefig(plotPath / 'skews.png')
    plt.close()
    return fig


def plot_mins(X_real, X_synth, plotPath = None):
    minsReal = X_real.min(axis = 0)
    minsSynth = X_synth.min(axis = 0)
    fig, ax = plt.subplots(figsize = (8, 5), facecolor = 'w')
    ax.boxplot([minsReal, minsSynth])
    ax.set_xticklabels(['Real min', 'Synthetic min'])
    plt.title(f'Comparison of minimum values')
    plt.ylabel('Value')
    if plotPath:
        plt.savefig(plotPath / 'mins.png')
    plt.close()
    return fig


def plot_maxs(X_real, X_synth, plotPath = None):
    maxsReal = X_real.max(axis = 0)
    maxsSynth = X_synth.max(axis = 0)
    fig, ax = plt.subplots(figsize = (8, 5), facecolor = 'w')
    ax.boxplot([maxsReal, maxsSynth])
    ax.set_xticklabels(['Real max', 'Synthetic max'])
    plt.title(f'Comparison of maximum values')
    plt.ylabel('Value')
    if plotPath:
        plt.savefig(plotPath / 'maxs.png')
    plt.close()
    return fig


def plot_mean_profiles(X_real, X_synth, plotPath):
    maxCols = min([X_real.shape[1], X_synth.shape[1]])
    fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (14, 10))
    sns.heatmap(X_real.astype(float).mean(axis = 1).reshape(24, -1), ax = axs[0, 0])
    axs[0, 0].set_title('Mean real profile')
    sns.heatmap(X_synth.astype(float).mean(axis = 1).reshape(24, -1), ax = axs[0, 1])
    axs[0, 1].set_title('Mean synth profile')
    sns.heatmap((X_synth.astype(float)[:, :maxCols] - X_real.astype(float)[:, :maxCols]).mean(axis = 1).reshape(24, -1), ax = axs[1, 0])
    axs[1, 0].set_title('Mean diff profile (synth - real)')
    axs[1, 1].axis('off')
    plt.tight_layout()
    if plotPath:
        plt.savefig(plotPath / 'mean_profiles.png')
    plt.close()
    return fig


def array_converter(df):
    arr = df.to_numpy()
    return arr


def model_plot_wrapper(X_real, X_synth, plotPath = None, removeIndex = True):
    if removeIndex:
        X_synth = X_synth[:, 1:]
    X_real = X_real.astype(float)
    X_synth = X_synth.astype(float)
    if type(X_real) != np.ndarray:
        X_real = X_real.to_numpy()
    compare_distributions(X_real, X_synth, plotPath)
    plot_means(X_real, X_synth, plotPath)
    plot_stds(X_real, X_synth, plotPath)
    plot_medians(X_real, X_synth, plotPath)
    plot_skews(X_real, X_synth, plotPath)
    plot_mins(X_real, X_synth, plotPath)
    plot_maxs(X_real, X_synth, plotPath)
    plot_mean_profiles(X_real, X_synth, plotPath)