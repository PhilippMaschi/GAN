
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cryptpandas as crp
from pathlib import Path
from scipy.stats import ks_2samp
import torch


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
        seasonal_hourly_means_real = season_real.groupby("hour of the day").mean().mean(axis=1)
        seasonal_hourly_std_real = season_real.groupby("hour of the day").std().mean(axis=1)

        seasonal_hourly_means_synthetic = season_synthetic.groupby("hour of the day").mean().mean(axis=1)
        seasonal_hourly_std_synthetic = season_synthetic.groupby("hour of the day").std().mean(axis=1)

        # Plot seasonal mean and standard deviation
        ax.plot(np.arange(24),
                seasonal_hourly_means_real,
                color="blue",
                linewidth=2,
                label=f'{season.capitalize()} Mean Real',
                )
        ax.fill_between(np.arange(24),
                        seasonal_hourly_means_real - seasonal_hourly_std_real,
                        seasonal_hourly_means_real + seasonal_hourly_std_real,
                        alpha=0.3,
                        label=f'{season.capitalize()} Std Dev Real',
                        color="cyan")

        ax.plot(np.arange(24),
                seasonal_hourly_means_synthetic,
                color="red",
                linewidth=2,
                label=f'{season.capitalize()} Mean Synthetic',
                )
        ax.fill_between(np.arange(24),
                        seasonal_hourly_means_synthetic - seasonal_hourly_std_synthetic,
                        seasonal_hourly_means_synthetic + seasonal_hourly_std_synthetic,
                        alpha=0.3,
                        label=f'{season.capitalize()} Std Dev Synthetic',
                        color="lightcoral")

        # Formatting the seasonal plot
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Profile Value')
        ax.set_title(f'Mean Hourly Profile for {season.capitalize()}')
        ax.legend()
        # plt.xticks(range(0, 24))
        # ax.grid(True)
    plt.tight_layout()
    fig.savefig(output_path / f"Daily_Mean_Comparison.png")
    plt.show()
    plt.close(fig)


def compare_distributions(real_df,
                          synthetic_df,
                          output_path: Path,
                          bins=50, ):
    """
    Compares the distributions of columns in two dataframes using histogram comparison
    and the Kolmogorov-Smirnov test.

    Parameters:
    real_df (pd.DataFrame): Dataframe with real load profiles.
    synthetic_df (pd.DataFrame): Dataframe with synthetic load profiles.
    bins (int): Number of bins to use for histograms.

    Returns:

    """
    if real_df.shape[1] != synthetic_df.shape[1]:
        raise ValueError("Both dataframes must have the same number of columns.")

    results = {}
    numeric_cols = [col for col in df_real.columns if is_number(col)]
    real = real_df[numeric_cols]
    synthetic = synthetic_df[numeric_cols]
    for col in real.columns:
        if col not in synthetic:
            raise ValueError(f"Column {col} not found in synthetic dataframe.")

        real_data = real[col].dropna()
        synthetic_data = synthetic[col].dropna()

        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = ks_2samp(real_data, synthetic_data)
        results[col] = {'KS Statistic': ks_stat, 'KS p-value': ks_p_value}

        # Histogram comparison
        plt.figure(figsize=(10, 6))
        plt.hist(real_data, bins=bins, alpha=0.5, label='Real')
        plt.hist(synthetic_data, bins=bins, alpha=0.5, label='Synthetic')
        plt.title(f'Histogram Comparison for {col}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


def get_season(date):
    seasons = {'spring': pd.date_range(start='2023-03-21 00:00:00', end='2023-06-20 23:00:00', freq="H"),
               'summer': pd.date_range(start='2023-06-21 00:00:00', end='2023-09-22 23:00:00', freq="H"),
               'autumn': pd.date_range(start='2023-09-23 00:00:00', end='2023-12-20 23:00:00', freq="H")}

    if date in seasons['spring']:
        return 'spring'
    elif date in seasons['summer']:
        return 'summer'
    elif date in seasons['autumn']:
        return 'autumn'
    else:
        return 'winter'

def get_season_and_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > 365:
        datetime_index = pd.date_range(start='2023-01-01 00:00:00', periods=8760, freq='H')
    else:
        datetime_index = pd.date_range(start='2023-01-01 ', periods=365, freq='D')
    df.index = datetime_index
    df['season'] = df.index.map(get_season)
    df["hour"] = df.index.hour
    return df.reset_index(drop=True)


def load_all(clusterLabel: int):
    path = Path(__file__).parent.parent / "GAN_data"
    df_loadProfiles = crp.read_encrypted(
        path=path / 'all_profiles.crypt',
        password="Ene123Elec#4")

    GAN_data_path = Path().absolute().parent / 'GAN_data'
    df_labels = pd.read_csv(GAN_data_path / 'DBSCAN_15_clusters_labels.csv', sep = ';')
    df_labels['name'] = df_labels['name'].str.split('_', expand = True)[1]

    number_of_profiles_gan_was_trained_on = 30
    profiles = df_labels.loc[df_labels['labels'] == clusterLabel, 'name'].to_list()[:number_of_profiles_gan_was_trained_on]
    df_profiles = df_loadProfiles[df_loadProfiles.columns[:13].tolist() + [item for item in profiles if item in df_loadProfiles.columns]].copy()
    

    df_shape= df_profiles.melt(id_vars = df_loadProfiles.columns[:13], value_vars = df_profiles.columns[13:], var_name = 'profile')
    df_shape = df_shape.pivot_table(values = 'value', index = ['date', 'profile'], columns = 'hour of the day')

   
         # load synthetic profiles
    model = torch.load("models/model_test_andi.pt")
    array = model.generate_sample()
    df_synthProfiles = df_shape.copy()
    df_synthProfiles[::] = array
    
    df_synthetic = df_synthProfiles.reset_index().melt(id_vars=["date","profile"]).pivot_table(values="value", columns="profile", index=["date", "hour of the day"])
    df_synthetic = pd.concat([df_loadProfiles[df_loadProfiles.columns[:13]], df_synthetic.reset_index(drop=True)], axis=1)
    
    assert df_profiles.shape == df_synthetic.shape
    return df_profiles, df_synthetic


def run_sdm_lstm_detection_test(real_df, synthetic_df):
    numeric = [col for col in df_real.columns if is_number(col)]

    a = LSTMDetection.compute(real_data=df_real[numeric],
                              synthetic_data=df_synthetic[numeric],
                              metadata=df_real["timestamp"])

if __name__ == "__main__":
    figures_output = Path(r"plots")
    df_real, df_synthetic = load_all(clusterLabel=1)
    #run_sdm_lstm_detection_test(real_df=df_real,
    #                            synthetic_df=df_synthetic)
    # compare_distributions(real_df=df_real,
    #                       synthetic_df=df_synthetic,
    #                       output_path=figures_output,
    #                       bins=50)

    plot_seasonal_daily_means(df_real=df_real,
                              df_synthetic=df_synthetic,
                              output_path=figures_output)
    compare_peak_and_mean(real_data=df_real,
                          synthetic_data=df_synthetic,
                          output_path=figures_output)
    plot_pca_analysis(real_data=df_real,
                      synthetic_data=df_synthetic,
                      output_path=figures_output)
