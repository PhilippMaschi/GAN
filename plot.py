import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


def plot_synthetic_vs_real_samples(model, df_profile, samplesScaled, synthSamples):
    outputPath = f'plots/{model.name}/synth_vs_real'
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for label in tqdm(range(len(df_profile))):
        plt.figure(facecolor = 'w', figsize = (10, 4))
        plt.plot(synthSamples[label], color = 'green', alpha = 0.35)
        plt.plot([], color = 'green', label = 'Synthetic')
        plt.plot(samplesScaled[label].T, color = 'red', label = 'Real', alpha = 0.75)
        plt.legend()
        plt.title(df_profile.index[label])
        plt.savefig(os.path.join(outputPath, f'{df_profile.index[label]}.png'))
        plt.close();


def plot_losses(model):
    outputPath = f'plots/{model.name}/losses'
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for item in tqdm(model.df_loss.columns[-6:]):
        plt.figure(facecolor = 'w')
        model.df_loss[item].astype(float).plot(title = item)
        plt.savefig(os.path.join(outputPath, f"{item.replace(' ', '_')}.png"))
        plt.close()