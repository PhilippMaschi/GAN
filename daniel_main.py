from pathlib import Path
from datetime import datetime
import os
import torch
from torch import nn

from daniel_preproc import data_preparation_wrapper, get_categorical_columns, gan_input_wrapper
from daniel_GAN import GAN, generate_data_from_saved_model
from daniel_config import config_wrapper
from daniel_plots import plot_wrapper

####################################################################################################

inputPath = Path().absolute().parent / 'GAN_data'
inputFilename = 'all_profiles.crypt'
inputPassword = 'Ene123Elec#4'
labelsFilename = 'DBSCAN_15_clusters_labels.csv'
clusterLabel = 0
maxProfileCount = None

####################################################################################################

batchSize = 10
lossFct = 'BCE'
lrGen = 1e-4/2
lrDis = 1e-4/2
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
epochCount = 170
labelReal = 1
labelFake = 0
dimNoise = 100

dimHidden = 64
channelCount = 24
modelGen = nn.Sequential(   #https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
    # 1st layer
    nn.ConvTranspose2d(in_channels = dimNoise, out_channels = 8*dimHidden, kernel_size = 3, stride = 1, padding = 0, bias = False),
    nn.BatchNorm2d(num_features = 8*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.1),
    # 2nd layer
    nn.ConvTranspose2d(in_channels = 8*dimHidden, out_channels = 4*dimHidden, kernel_size = 3, stride = 2, padding = 0, bias = False),
    nn.BatchNorm2d(num_features = 4*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.1),
    # 3rd layer
    nn.ConvTranspose2d(in_channels = 4*dimHidden, out_channels = 2*dimHidden, kernel_size = 3, stride = (1, 2), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = 2*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.1),
    # 4th layer
    nn.ConvTranspose2d(in_channels = 2*dimHidden, out_channels = dimHidden, kernel_size = 3, stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.1),
    # Output layer
    nn.ConvTranspose2d(in_channels = dimHidden, out_channels = channelCount, kernel_size = 3, stride = (1, 2), padding = (1, 0), bias = False),
    nn.Tanh()
)
modelDis = nn.Sequential(
    # 1st layer
    nn.Conv2d(in_channels = channelCount, out_channels = dimHidden, kernel_size = 3, stride = (1, 2), padding = (1, 0), bias = False),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # 2nd layer
    nn.Conv2d(in_channels = dimHidden, out_channels = 2*dimHidden, kernel_size = 3, stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = 2*dimHidden),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # 3rd layer
    nn.Conv2d(in_channels = 2*dimHidden, out_channels = 4*dimHidden, kernel_size = 3, stride = (1, 2), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = 4*dimHidden),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # 4th layer
    nn.Conv2d(in_channels = 4*dimHidden, out_channels = 8*dimHidden, kernel_size = 3, stride = 2, padding = 0, bias = False),
    nn.BatchNorm2d(num_features = 8*dimHidden),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # Output layer
    nn.Conv2d(in_channels = 8*dimHidden, out_channels = 1, kernel_size = 3, stride = 1, padding = 0, bias = False),
    nn.Sigmoid()
)

runName = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]
outputPath = Path().absolute() / 'runs' / runName
os.makedirs(outputPath)
modelSaveFreq = 50

####################################################################################################

if __name__ == '__main__':
    df_train = data_preparation_wrapper(
        dataFilePath = inputPath / inputFilename,
        password = inputPassword,
        labelsFilePath = inputPath / labelsFilename,
        clusterLabel = clusterLabel,
        maxProfileCount = maxProfileCount
    )
    X_trainProcd, X_train, minMax = gan_input_wrapper(df_train, outputPath) #Procd... processed, meaning normalized and reshaped
    df_hull = get_categorical_columns(df_train)
    del df_train

    model = GAN(
        dataset = X_trainProcd,
        batchSize = batchSize,
        modelGen = modelGen,
        modelDis = modelDis,
        lossFct = lossFct,
        lrGen = lrGen,
        lrDis = lrDis,
        device = device,
        epochCount = epochCount,
        labelReal = labelReal,
        labelFake = labelFake,
        dimNoise = dimNoise,
        outputPath = outputPath,
        modelSaveFreq = modelSaveFreq
    )
    config_wrapper(model, outputPath)
    model.train()
    X_synth = generate_data_from_saved_model(
        runPath = outputPath,
        modelGen = modelGen,
        device = device,
        profileCount = X_trainProcd.shape[0],
        dimNoise = dimNoise
    )

    plot_wrapper(X_train, X_synth, df_hull, outputPath)