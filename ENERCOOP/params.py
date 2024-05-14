import torch
from torch import nn
import pandas as pd
import numpy as np

####################################################################################################

trackProgress = False
batchSize = 15
lossFct = 'BCE'
lrGen = 1e-4/3.25
lrDis = 1e-4/2.25
betas = (0.5, 0.999)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
epochCount = 450
modelSaveFreq = 500
loopCountGen = 2
thresh = None
threshEpochMin = 100

####################################################################################################

dimNoise = 90
dimHidden = 64
channelCount = 24

modelGen = nn.Sequential(   #https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
    # 1st layer
    nn.ConvTranspose2d(in_channels = dimNoise, out_channels = 8*dimHidden, kernel_size = 3, stride = 1, padding = 0, bias = False),
    nn.BatchNorm2d(num_features = 8*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.075),
    # 2nd layer
    nn.ConvTranspose2d(in_channels = 8*dimHidden, out_channels = 4*dimHidden, kernel_size = 3, stride = 2, padding = 0, bias = False),
    nn.BatchNorm2d(num_features = 4*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.075),
    # 3rd layer
    nn.ConvTranspose2d(in_channels = 4*dimHidden, out_channels = 2*dimHidden, kernel_size = 3, stride = (1, 2), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = 2*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.075),
    # 4th layer
    nn.ConvTranspose2d(in_channels = 2*dimHidden, out_channels = 2*dimHidden, kernel_size = 3, stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = 2*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.075),
    # 5th layer
    nn.ConvTranspose2d(in_channels = 2*dimHidden, out_channels = dimHidden, kernel_size = 3, stride = (1, 1), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.075),
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

from ENERCOOP.preproc import revert_reshape_arr #should revert the changes made by `reshape_arr`, needed to restructure GAN output

####################################################################################################

params = {
    'trackProgress': trackProgress,
    'batchSize': batchSize,
    'lossFct': lossFct,
    'lrGen': lrGen,
    'lrDis': lrDis,
    'betas': betas,
    'device': device,
    'epochCount': epochCount,
    'modelSaveFreq': modelSaveFreq,
    'loopCountGen': loopCountGen,
    'thresh': thresh,
    'threshEpochMin': threshEpochMin,
    'dimNoise': dimNoise,
    'dimHidden': dimHidden,
    'channelCount': channelCount,
    'modelGen': modelGen,
    'modelDis': modelDis,
    'restructure_GAN_output': revert_reshape_arr
}