from pathlib import Path
from datetime import datetime
import os
import torch
from torch import nn
import wandb

from preproc import data_preparation_wrapper, get_categorical_columns, save_profile_IDs, gan_input_wrapper
from GAN import GAN, generate_data_from_saved_model
from config import config_wrapper
from plots import plot_wrapper

####################################################################################################

inputPath = Path().absolute().parent.parent / 'GAN_data'
inputFilename = 'all_profiles.crypt'
inputPassword = 'Ene123Elec#4'
labelsFilename = 'DBSCAN_15_clusters_labels.csv'
clusterLabels = [0]
maxProfileCount = None

runName = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]
outputPath = Path().absolute() / 'daniel_workspace' / 'runs' / runName
os.makedirs(outputPath)
dimData = 3
modelSaveFreq = 200

####################################################################################################

batchSize = 15
lossFct = 'BCE'
lrGen = 1e-4/3
lrDis = 1e-4/2
betas = (0.4, 0.999)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
epochCount = 45
labelReal = 0
labelFake = 1
dimNoise = 90

dimHidden = 64
channelCount = 24
betas = (0.5, 0.999)

hyperparams = {
    "batchSize": batchSize,
    "lossFct": lossFct,
    "lrGen": lrGen,
    "lrDis": lrDis,
    "device": device,
    "epochCount": epochCount,
    "labelReal": labelReal,
    "labelFake": labelFake,
    "dimNoise": dimNoise,
    "dimHidden": dimHidden,
    "channelCount": channelCount,
    "betas": betas

}

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
    nn.ConvTranspose2d(in_channels = 2*dimHidden, out_channels = 2*dimHidden, kernel_size = 3, stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = 2*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.1),
    # 5th layer
    nn.ConvTranspose2d(in_channels = 2*dimHidden, out_channels = dimHidden, kernel_size = 3, stride = (1, 1), padding = (1, 1), bias = False),
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

####################################################################################################

if __name__ == '__main__':
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="GAN",

        # mode='offline',

        # track hyperparameters and run metadata
        config=hyperparams
    )
    model_name = wandb.run.name
    outputPath = Path().absolute() / 'daniel_workspace' / 'runs' / f"{model_name}_{runName}"
    os.makedirs(outputPath)
    df_train = data_preparation_wrapper(
        dataFilePath = inputPath / inputFilename,
        password = inputPassword,
        labelsFilePath = inputPath / labelsFilename,
        clusterLabels = clusterLabels,
        maxProfileCount = maxProfileCount
    )
    X_trainProcd, X_train, minMax = gan_input_wrapper(df_train, dimData, outputPath) #Procd... processed, meaning normalized and reshaped
    df_hull = get_categorical_columns(df_train)
    save_profile_IDs(df_train, outputPath)
    del df_train

    model = GAN(
        dataset = X_trainProcd,
        batchSize = batchSize,
        modelGen = modelGen,
        modelDis = modelDis,
        lossFct = lossFct,
        lrGen = lrGen,
        lrDis = lrDis,
        betas = betas,
        device = device,
        epochCount = epochCount,
        labelReal = labelReal,
        labelFake = labelFake,
        dimNoise = dimNoise,
        outputPath = outputPath,
        modelSaveFreq = modelSaveFreq,
        wandb=wandb,
        betas=betas
    )
    config_wrapper(model, outputPath)



    wandb.watch(model)
    model.train()
    wandb.finish()



    X_synth = generate_data_from_saved_model(
        runPath = outputPath,
        modelGen = modelGen,
        device = device,
        profileCount = X_trainProcd.shape[0],
        dimNoise = dimNoise,
        dimData = dimData
    )

    plot_wrapper(X_train, X_synth, df_hull, outputPath)