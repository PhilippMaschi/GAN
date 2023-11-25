
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cryptpandas as crp
from pathlib import Path


def plot_pca_analysis(real_data, synthetic_data, output_path: Path):
    """
    Perform PCA and plot the first two principal components for real and synthetic data.
    
    Parameters:
    real_data (DataFrame): DataFrame containing the real load profiles.
    synthetic_data (DataFrame): DataFrame containing the synthetic load profiles.
    """
    # Combining the data
    numeric_cols = [col for col in df_real.columns if is_number(col)]
    real = real_data[numeric_cols]
    synthetic = synthetic_data[numeric_cols]
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


def compare_peak_and_mean(real_data, synthetic_data, output_path: Path):
    """
    Compare the peak and mean values between real and synthetic data.
    
    Parameters:
    real_data (DataFrame): DataFrame containing the real load profiles.
    synthetic_data (DataFrame): DataFrame containing the synthetic load profiles.
    """
    numeric_cols = [col for col in df_real.columns if is_number(col)]
    real = real_data[numeric_cols]
    synthetic = synthetic_data[numeric_cols]
    # Calculating peak and mean values
    real_peaks = real.max()
    synthetic_peaks = synthetic.max()
    real_means = real.mean()
    synthetic_means = synthetic.mean()

    # Plotting peak values
    plt.figure(figsize=(10, 7))
    plt.plot(real_peaks, label='Real Peaks', color='blue')
    plt.plot(synthetic_peaks, label='Synthetic Peaks', color='red')
    plt.title('Comparison of Peak Values')
    plt.xlabel('Profile Index')
    plt.ylabel('Peak Value')
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"total_peak_of_profiles_comparison.png")
    plt.show()

    # Plotting mean values
    plt.figure(figsize=(10, 7))
    plt.plot(real_means, label='Real Means', color='blue')
    plt.plot(synthetic_means, label='Synthetic Means', color='red')
    plt.title('Comparison of Mean Values')
    plt.ylabel('Mean Value')
    plt.legend()
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    plt.xlabel('Profile Index')
    plt.tight_layout()
    plt.savefig(output_path / f"total_mean_of_profiles_comparison.png")
    plt.show()


def is_number(s):
    """
    Check if the input string s is a number.

    Parameters:
    s (str): The string to check.

    Returns:
    bool: True if s is a number, False otherwise.
    """
    try:
        float(s)  # for int, long and float
    except ValueError:
        return False
    return True


def plot_seasonal_daily_means(df_real: pd.DataFrame,
                              df_synthetic: pd.DataFrame,
                              output_path: Path):
    numeric_cols = [col for col in df_real.columns if is_number(col)]
    # add seasons to df:
    season_groups_real = df_real.groupby("meteorological season")
    season_groups_synthetic = df_synthetic.groupby("meteorological season")
    # Separate plots for each season
    seasons = df_real["meteorological season"].unique()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 6), sharey=True)
    axes = axes.flatten()
    for i, season in enumerate(seasons):
        ax = axes[i]
        season_real = season_groups_real.get_group(season)[["hour of the day"] + numeric_cols]
        season_synthetic = season_groups_synthetic.get_group(season)[["hour of the day"] + numeric_cols]
        # Filter the dataframe for the season and aggregate data by hour
        seasonal_hourly_means_real = season_real.groupby("hour of the day").mean()
        seasonal_hourly_std_real = season_real.groupby("hour of the day").std()

        seasonal_hourly_means_synthetic = season_synthetic.groupby("hour of the day").mean()
        seasonal_hourly_std_synthetic = season_synthetic.groupby("hour of the day").std()

        # Plot seasonal mean and standard deviation
        ax.plot(np.arange(24),
                seasonal_hourly_means_real.mean(axis=1),
                color="blue",
                linewidth=2,
                label=f'{season.capitalize()} Mean Real',
                )
        ax.fill_between(np.arange(24),
                        seasonal_hourly_means_real.mean(axis=1) - seasonal_hourly_std_real.mean(axis=1),
                        seasonal_hourly_means_real.mean(axis=1) + seasonal_hourly_std_real.mean(axis=1),
                        alpha=0.3,
                        label=f'{season.capitalize()} Std Dev Real',
                        color="cyan")

        ax.plot(np.arange(24),
                seasonal_hourly_means_synthetic.mean(axis=1),
                color="red",
                linewidth=2,
                label=f'{season.capitalize()} Mean Synthetic',
                )
        ax.fill_between(np.arange(24),
                        seasonal_hourly_means_synthetic.mean(axis=1) - seasonal_hourly_std_synthetic.mean(axis=1),
                        seasonal_hourly_means_synthetic.mean(axis=1) + seasonal_hourly_std_synthetic.mean(axis=1),
                        alpha=0.3,
                        label=f'{season.capitalize()} Std Dev Synthetic',
                        color="lightcoral")

        # Formatting the seasonal plot
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Profile Value')
        ax.set_title(f'Mean Hourly Profile for {season.capitalize()} and Custer')
        ax.legend()
        # plt.xticks(range(0, 24))
        # ax.grid(True)
    plt.tight_layout()
    fig.savefig(output_path / f"Daily_Mean_Comparison.png")
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
    plot_seasonal_daily_means(df_real=df_real,
                              df_synthetic=df_synthetic,
                              output_path=figures_output)
    compare_peak_and_mean(real_data=df_real,
                          synthetic_data=df_synthetic,
                          output_path=figures_output)
    plot_pca_analysis(real_data=df_real,
                      synthetic_data=df_synthetic,
                      output_path=figures_output)
