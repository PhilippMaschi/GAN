import torch
from torch import nn


# Parameters and their default values
params = {
    'batchSize': 40,
    'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
    'lossFct': nn.BCELoss(),
    'lrGen': 1e-4/3.25,
    'lrDis': 1e-4/2.25,
    'betas': (0.5, 0.999),
    'epochCount': 250,
    'labelReal': 1,
    'labelFake': 0,
    'saveFreq': 1000,
    'loopCountGen': 3,
    'saveSamples': False,
    'saveModels': False,
    'dimNoise': 128,
    'dimHidden': 16,
    'channelCount': 1,
    'outputFormat': '.npy'
}