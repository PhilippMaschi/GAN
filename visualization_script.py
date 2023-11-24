
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca_analysis(real_data, synthetic_data):
    """
    Perform PCA and plot the first two principal components for real and synthetic data.
    
    Parameters:
    real_data (DataFrame): DataFrame containing the real load profiles.
    synthetic_data (DataFrame): DataFrame containing the synthetic load profiles.
    """
    # Combining the data
    combined_data = pd.concat([real_data, synthetic_data])
    labels = np.array(['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data))

    # Standardizing the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(combined_data)

    # Performing PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(standardized_data)

    # Plotting the PCA
    plt.figure(figsize=(10, 7))
    plt.scatter(principal_components[labels == 'Real', 0], principal_components[labels == 'Real', 1], alpha=0.5, label='Real')
    plt.scatter(principal_components[labels == 'Synthetic', 0], principal_components[labels == 'Synthetic', 1], alpha=0.5, label='Synthetic')
    plt.title('PCA of Real vs Synthetic Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
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
