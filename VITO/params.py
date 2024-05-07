import torch
from torch import nn

####################################################################################################

trackProgress = False
batchSize = 12
lossFct = 'BCE'
lrGen = 1e-5
lrDis = 1e-5
betas = (0.5, 0.999)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
epochCount = 400
modelSaveFreq = 500
loopCountGen = 5
thresh = None
threshEpochMin = 100

####################################################################################################

dimNoise = 90
#dimHidden = 24
dimHidden = 32
channelCount = 1

modelGen = nn.Sequential(
    # 1st layer 
    nn.ConvTranspose2d(in_channels = dimNoise, out_channels = 8*dimHidden, kernel_size = (12, 46), stride = (1, 1), padding = (0, 0), bias = False),
    nn.BatchNorm2d(num_features = 8*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.065),
    # 2nd layer
    nn.ConvTranspose2d(in_channels = 8*dimHidden, out_channels = 4*dimHidden, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = 4*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.065),
    # 3rd layer
    nn.ConvTranspose2d(in_channels = 4*dimHidden, out_channels = 2*dimHidden, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = 2*dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.065),
    # 4th layer
    nn.ConvTranspose2d(in_channels = 2*dimHidden, out_channels = dimHidden, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = dimHidden),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = 0.065),
    # Output layer
    nn.ConvTranspose2d(in_channels = dimHidden, out_channels = channelCount, kernel_size = (10, 6), stride = (1, 1), padding = (1, 0), bias = False),
    nn.Tanh()
)

modelDis = nn.Sequential(
    # 1st layer
    nn.Conv2d(in_channels = channelCount, out_channels = dimHidden, kernel_size = (3, 3), stride = (1, 1), padding = (1, 0), bias = False),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # 2nd layer
    nn.Conv2d(in_channels = dimHidden, out_channels = 2*dimHidden, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = 2*dimHidden),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # 3rd layer
    nn.Conv2d(in_channels = 2*dimHidden, out_channels = 4*dimHidden, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = 4*dimHidden),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # 4th layer
    nn.Conv2d(in_channels = 4*dimHidden, out_channels = 8*dimHidden, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False),
    nn.BatchNorm2d(num_features = 8*dimHidden),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # Output layer
    nn.Conv2d(in_channels = 8*dimHidden, out_channels = 1, kernel_size = (12, 46), stride = (1, 1), padding = (0, 0), bias = False),
    nn.Sigmoid()
)

####################################################################################################

from VITO.preproc import revert_reshape_arr #should revert the changes made by `reshape_arr`, needed to restructure GAN output

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