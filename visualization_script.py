import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cryptpandas as crp
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def plot_pca_analysis(real_data, synthetic_data, output_path: Path):
    """
    Perform PCA and plot the first two principal components for real and synthetic data.
    
    Parameters:
    real_data (DataFrame): DataFrame containing the real load profiles.
    synthetic_data (DataFrame): DataFrame containing the synthetic load profiles.
    """
    # Combining the data
    real = real_data.iloc[:, 13:]
    synthetic = synthetic_data.iloc[:, 13:]
    combined_data = pd.concat([real, synthetic], axis=0)
    labels = np.array(['Real'] * len(real) + ['Synthetic'] * len(synthetic))

    # Standardizing the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(combined_data)

    # Performing PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(standardized_data)

    # Plotting the PCA
    plt.figure(figsize=(10, 7))
    plt.scatter(principal_components[labels == 'Real', 0],
                principal_components[labels == 'Real', 1],
                alpha=0.3,
                label='Real',
                )
    plt.scatter(principal_components[labels == 'Synthetic', 0],
                principal_components[labels == 'Synthetic', 1],
                alpha=0.3,
                label='Synthetic',
                )
    plt.title('PCA of Real vs Synthetic Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig(output_path / f"PCA.png")
    plt.show()


def compare_peak_and_mean(real_data, synthetic_data):
    """
    Compare the peak and mean values between real and synthetic data.
    
    Parameters:
    real_data (DataFrame): DataFrame containing the real load profiles.
    synthetic_data (DataFrame): DataFrame containing the synthetic load profiles.
    """
    # Calculating peak and mean values
    real_peaks = real_data.max()
    synthetic_peaks = synthetic_data.max()
    real_means = real_data.mean()
    synthetic_means = synthetic_data.mean()

    # Plotting peak values
    plt.figure(figsize=(10, 7))
    plt.plot(real_peaks, label='Real Peaks', color='blue')
    plt.plot(synthetic_peaks, label='Synthetic Peaks', color='red')
    plt.title('Comparison of Peak Values')
    plt.xlabel('Profile Index')
    plt.ylabel('Peak Value')
    plt.legend()
    plt.show()

    # Plotting mean values
    plt.figure(figsize=(10, 7))
    plt.plot(real_means, label='Real Means', color='blue')
    plt.plot(synthetic_means, label='Synthetic Means', color='red')
    plt.title('Comparison of Mean Values')
    plt.xlabel('Profile Index')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.show()


def plot_seasonal_daily_means(df_real: pd.DataFrame, df_synthetic: pd.DataFrame):
    # add seasons to df:
    season_groups = df.groupby("meteorological season")

    # Separate plots for each season
    season_colors = {
        'Winter': 'lightblue',
        'Spring': 'lightgreen',
        'Summer': 'lightcoral',
        'Fall': 'wheat'
    }
    seasons = df["meteorological season"].unique()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 6), sharey=True)
    axes = axes.flatten()
    for i, season in enumerate(seasons):
        ax = axes[i]
        season_df = season_groups.get_group(season)
        columns_to_drop = [name for name in season_df.columns if type(name) != int and name != "hour of the day"]
        season_df = season_df.drop(columns=columns_to_drop)
        # Filter the dataframe for the season and aggregate data by hour
        seasonal_hourly_means = season_df.groupby("hour of the day").mean()
        seasonal_hourly_std = season_df.groupby("hour of the day").std()

        # Plot seasonal mean and standard deviation
        ax.plot(np.arange(24),
                seasonal_hourly_means.mean(axis=1),
                color='black',
                linewidth=2,
                label=f'{season.capitalize()} Mean',
                )
        ax.fill_between(np.arange(24),
                        seasonal_hourly_means.mean(axis=1) - seasonal_hourly_std.mean(axis=1),
                        seasonal_hourly_means.mean(axis=1) + seasonal_hourly_std.mean(axis=1),
                        alpha=0.3,
                        label=f'{season.capitalize()} Std Dev',
                        color=season_colors[season])

        # Formatting the seasonal plot
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Profile Value')
        ax.set_title(f'Mean Hourly Profile for {season.capitalize()} and Custer {cluster_label}')
        ax.legend()
        # plt.xticks(range(0, 24))
        # ax.grid(True)
    plt.tight_layout()
    fig.savefig(f"figures/seasonal_means_of_cluster_{cluster_label}")
    plt.close(fig)


def load_all(clusterLabel: int):
    path = Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\MODERATE\model")
    # load synthetic profiles
    df_ = pd.read_csv(path / "synthetic.csv").drop(columns=["date", "hour of the day"])
    label_list = list(df_.columns)

    df_loadProfiles = crp.read_encrypted(
        path=path / 'all_profiles.crypt',
        password="Ene123Elec#4")
    columns_to_keep = list(df_loadProfiles.columns)[:13]
    df_real = df_loadProfiles[columns_to_keep + label_list]
    df_synthetic = pd.concat([df_real[columns_to_keep], df_], axis=1)
    assert df_real.shape == df_synthetic.shape
    return df_real, df_synthetic


if __name__ == "__main__":
    figures_output = Path(r"plots")
    df_real, df_synthetic = load_all(clusterLabel=1)
    plot_pca_analysis(real_data=df_real,
                      synthetic_data=df_synthetic,
                      output_path=figures_output)
