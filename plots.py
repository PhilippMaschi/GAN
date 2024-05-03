import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')   #for seaborn
import os
import io
import numpy as np
from PIL import Image
import imageio


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


def plot_wrapper(X_real, X_synth, runPath):
    plotPath = runPath / 'plots'
    os.makedirs(plotPath) if not os.path.exists(plotPath) else None
    compare_distributions(X_real, X_synth, plotPath)
    plot_peaks(X_real, X_synth, plotPath)
    plot_means(X_real, X_synth, plotPath)