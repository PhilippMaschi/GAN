import torch
from torch import nn


params = {
    'batchSize': 15,
    'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
    'lossFct': nn.BCELoss(),
    'lrGen': 1e-4/3.25,
    'lrDis': 1e-4/2.25,
    'betas': (0.5, 0.999),
    'epochCount': 550,
    'labelReal': 1,
    'labelFake': 0,
    'modelSaveFreq': 500,
    'loopCountGen': 2,
    'trackProgress': False,
    'dimNoise': 64,
    'dimHidden': 16,
    'channelCount': 1
}