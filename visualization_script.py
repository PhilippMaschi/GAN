import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cryptpandas as crp
from pathlib import Path
from scipy.stats import ks_2samp
import torch
import plotly.express as px
import random
from GAN_Philipp import generate_data_from_saved_model, MinMaxScaler
from philipp_main import create_training_dataframe, create_numpy_matrix_for_gan
import seaborn as sns



# matplotlib.use('Agg')


def plotly_single_profiles(real_data, synthetic_data, epoch: int):
    numeric_cols = [col for col in real_data.columns if is_number(col)]
    random_profiles = random.sample(numeric_cols, 3)
    real = real_data[random_profiles]
    real.loc[:, "type"] = "real"
    real.loc[:, "hour"] = np.arange(len(real))
    synthetic = synthetic_data[random_profiles]
    synthetic.loc[:, "type"] = "synthetic"
    synthetic.loc[:, "hour"] = np.arange(len(synthetic))

    plot_df = pd.concat([real, synthetic], axis=0)
    plot_df = plot_df.melt(id_vars=["type", "hour"], var_name="profile")

    fig = px.line(
        data_frame=plot_df,
        x="hour",
        y="value",
        line_dash="type",
        color="profile",
        title=f"Epoch: {epoch}"
    )
    fig.show()


def plot_pca_analysis(real_data, synthetic_data, output_path: Path, epoch: int):
    """
    Perform PCA and plot the first two principal components for real and synthetic data.
    
    Parameters:
    real_data (DataFrame): DataFrame containing the real load profiles.
    synthetic_data (DataFrame): DataFrame containing the synthetic load profiles.
    """
    # Combining the data
    numeric_cols = [col for col in real_data.columns if is_number(col)]
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

    plt.title(f'PCA of Real vs Synthetic Data epoch: {epoch}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig(output_path / f"PCA_{epoch}.png")


def plot_average_week(synthetic_df, real_df, output_path, epoch: int):
    numeric_cols = [col for col in real_df.columns if is_number(col)]
    # group by weeks
    season_groups_real = real_df.groupby("meteorological season")
    season_groups_synthetic = synthetic_df.groupby("meteorological season")
    seasons = real_df["meteorological season"].unique()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 6), sharey=True)
    axes = axes.flatten()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, season in enumerate(seasons):
        ax = axes[i]
        season_real = season_groups_real.get_group(season)[["weekday"] + numeric_cols]
        season_synthetic = season_groups_synthetic.get_group(season)[["weekday"] + numeric_cols]
        season_real_melt = season_real.melt(id_vars="weekday", var_name="profiles", value_name="load (Wh)")
        season_real_melt.loc[:, "type"] = "real"
        season_synthetic_melt = season_synthetic.melt(id_vars="weekday", var_name="profiles", value_name="load (Wh)")
        season_synthetic_melt.loc[:, "type"] = "synthetic"
        plot_df = pd.concat([season_real_melt, season_synthetic_melt], axis=0)
        sns.boxplot(
            data=plot_df,
            x="weekday",
            y="load (Wh)",
            ax=ax,
            hue="type",
            order=weekday_order,
            showfliers=False
        )
        ax.set_title(f"{season}")

    plt.suptitle(f"Epoch: {epoch}")
    plt.tight_layout()
    plt.savefig(output_path / f"Weekly_load_{epoch}.png")



def compare_peak_and_mean(real_data, synthetic_data, output_path: Path, epoch: int):
    """
    Compare the peak and mean values between real and synthetic data.
    
    Parameters:
    real_data (DataFrame): DataFrame containing the real load profiles.
    synthetic_data (DataFrame): DataFrame containing the synthetic load profiles.
    """
    numeric_cols = [col for col in real_data.columns if is_number(col)]
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
    plt.title(f'Comparison of Peak Values epoch {epoch}')
    plt.xlabel('Profile Index')
    plt.ylabel('Peak Value')
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"total_peak_of_profiles_comparison_{epoch}.png")

    # Plotting mean values
    plt.figure(figsize=(10, 7))
    plt.plot(real_means, label='Real Means', color='blue')
    plt.plot(synthetic_means, label='Synthetic Means', color='red')
    plt.title(f'Comparison of Mean Values epoch {epoch}')
    plt.ylabel('Mean Value')
    plt.legend()
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    plt.xlabel('Profile Index')
    plt.tight_layout()
    plt.savefig(output_path / f"total_mean_of_profiles_comparison_{epoch}.png")


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
                              output_path: Path,
                              epoch_number: int
                              ):
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
    plt.suptitle(f"epoch: {epoch_number}")
    plt.tight_layout()
    fig.savefig(output_path / f"Daily_Mean_Comparison_{epoch_number}.png")
    plt.close(fig)


def compare_distributions(real_df,
                          synthetic_df,
                          output_path: Path,
                          epoch: int,
                          bins=100, ):
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

    numeric_cols = [col for col in real_df.columns if is_number(col)]
    real_flattend = real_df[numeric_cols].values.flatten()
    synthetic_flattend = synthetic_df[numeric_cols].values.flatten()

    # Kolmogorov-Smirnov test
    # ks_stat, ks_p_value = ks_2samp(real_flattend, synthetic_flattend)

    # Histogram comparison
    plt.figure(figsize=(10, 6))
    plt.hist(real_flattend, bins=bins, alpha=0.5, label='Real')
    plt.hist(synthetic_flattend, bins=bins, alpha=0.5, label='Synthetic')
    plt.title(f'Histogram Comparison for')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"Load_Distribution_{epoch}.png")


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


# def run_sdm_lstm_detection_test(real_df, synthetic_df):
#     numeric = [col for col in df_real.columns if is_number(col)]
#
#     a = LSTMDetection.compute(real_data=df_real[numeric],
#                               synthetic_data=df_synthetic[numeric],
#                               metadata=df_real["timestamp"])


def numpy_matrix_to_pandas_table_with_metadata(hull: pd.DataFrame, synthetic_data: np.array, original_meta_data):
    hull[::] = synthetic_data
    synthetic = hull.reset_index()
    # todo the month sin etc. as list to this function dependent on the model so this is automated for other variables
    df_synthetic = synthetic.melt(
        id_vars=['date', 'profile', "month sin", "month cos", "weekday sin", "weekday cos", "day off"],
        var_name="hour of the day",
        value_name="value")
    df_pivot = df_synthetic.pivot_table(values='value',
                                        index=['date', "month sin", "month cos", "weekday sin", "weekday cos",
                                               "day off", "hour of the day"],
                                        columns='profile').reset_index()
    final = pd.concat([original_meta_data.reset_index(), df_pivot[[col for col in df_pivot.columns if is_number(col)]]],
                      axis=1)

    return final


def visualize_results_from_model_folder(
        folder_path,
        noise_dimension,
        feature_count,  # depends on the features selected in train_gan -> automate
        target_count,  # 24 if we trained on days
        n_profiles_trained_on,
        normalize,
        device,
        loss,
):
    """

    Args:
        folder_path:
        noise_dimension:
        feature_count:
        target_count:
        n_profiles_trained_on:
        normalize: if normalized than the comparison is done on profiles between -1 and 1
        device: cpu or cuda:0
        loss:

    Returns:

    """
    # visualize the training results:
    file_names = [file.name for file in Path(folder_path).glob("*.pt")]
    file_names.sort()

    orig_features = np.load(folder_path / "original_features.npy")
    hull = pd.read_parquet(folder_path / "hull.parquet.gzip")
    orig_meta_data = pd.read_parquet(folder_path / "meta_data.parquet.gzip")
    for model in file_names:
        epoch = int(model.replace("epoch=", "").replace(".pt", ""))
        synthetic_data = generate_data_from_saved_model(
            model_path=f"{folder_path}/{model}",
            noise_dim=noise_dimension,
            featureCount=feature_count,
            targetCount=target_count,
            original_features=orig_features,
            normalized=normalize,
            device=device,
        )

        df_synthetic = numpy_matrix_to_pandas_table_with_metadata(
            hull=hull,
            synthetic_data=synthetic_data,
            original_meta_data=orig_meta_data
        ).set_index("timestamp")

        folder_name = Path(folder_path).stem
        output_path = Path(folder_path).parent.parent / "plots" / folder_name
        output_path.mkdir(parents=True, exist_ok=True)

        train_df = create_training_dataframe(
            password_="Ene123Elec#4",
            clusterLabel=cluster_label,
            number_of_profiles=n_profiles_trained_on,
            label_csv_filename="DBSCAN_15_clusters_labels.csv",
            path_to_orig_file=Path(folder_path).parent.parent.parent / "GAN_data"
        )
        if normalize:
            target, features, df_hull = create_numpy_matrix_for_gan(train_df.copy())
            scaler = MinMaxScaler(feature_range=(-1, 1))
            samplesScaled = scaler.fit_transform(target.T).T
            df_real = numpy_matrix_to_pandas_table_with_metadata(
                hull=hull,
                synthetic_data=samplesScaled,
                original_meta_data=orig_meta_data
            ).set_index("timestamp")
        else:
            df_real = train_df


        plot_average_week(df_synthetic, df_real, output_path, epoch)

        plot_seasonal_daily_means(df_real=df_real,
                                  df_synthetic=df_synthetic,
                                  output_path=output_path,
                                  epoch_number=epoch)

        plot_pca_analysis(real_data=df_real,
                          synthetic_data=df_synthetic,
                          output_path=output_path,
                          epoch=epoch)

        compare_peak_and_mean(real_data=df_real,
                              synthetic_data=df_synthetic,
                              output_path=output_path,
                              epoch=epoch)

        compare_distributions(real_df=df_real,
                              synthetic_df=df_synthetic,
                              output_path=output_path,
                              epoch=epoch,
                              bins=100)


        # plotly_single_profiles(real_data=df_real,
        #                        synthetic_data=df_synthetic,
        #                        epoch=epoch)


if __name__ == "__main__":
    model_nickname = "ModelTestPhilipp"
    batch_size = 1000
    noise_dim = 50
    feature_count = 5
    cluster_algorithm = "DBSCAN"
    cluster_label = 0
    n_profiles_trained_on = 100
    target_count = 24
    device = "cuda:0"
    loss = "BCE"

    folder_name = f"models/{model_nickname}_" \
                  f"Clustered={cluster_algorithm}_" \
                  f"ClusterLabel={cluster_label}_" \
                  f"NProfilesTrainedOn={n_profiles_trained_on}_" \
                  f"BatchSize={batch_size}_" \
                  f"FeatureCount={feature_count}_" \
                  f"NoiseDim={noise_dim}_" \
                  f"Loss={loss}"

    # model_folder = Path(r"X:\projects4\workspace_danielh_pr4\GAN") / folder_name
    model_folder = Path(__file__).absolute().parent / folder_name

    normalize = True
    visualize_results_from_model_folder(
        folder_path=model_folder,
        noise_dimension=noise_dim,
        feature_count=feature_count,
        target_count=target_count,
        n_profiles_trained_on=n_profiles_trained_on,
        normalize=normalize,
        device=device,
        loss=loss
    )
