from torch import nn

from model.params import params


DIM_NOISE = params['dimNoise']
DIM_HIDDEN = params['dimHidden']
CHANNEL_COUNT = params['channelCount']
P_DROPOUT = 0.2


layersGen = nn.Sequential(
    # 1st layer
    nn.ConvTranspose2d(in_channels = DIM_NOISE, out_channels = 8*DIM_HIDDEN, kernel_size = (24, 22), stride = (1, 1), padding = (0, 0), bias = False),
    nn.BatchNorm2d(num_features = 8*DIM_HIDDEN),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = P_DROPOUT),
    # 2nd layer
    nn.ConvTranspose2d(in_channels = 8*DIM_HIDDEN, out_channels = 4*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = 4*DIM_HIDDEN),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = P_DROPOUT),
    # 3rd layer
    nn.ConvTranspose2d(in_channels = 4*DIM_HIDDEN, out_channels = 2*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = 2*DIM_HIDDEN),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = P_DROPOUT),
    # 4th layer
    nn.ConvTranspose2d(in_channels = 2*DIM_HIDDEN, out_channels = DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = DIM_HIDDEN),
    nn.ReLU(inplace = True),
    nn.Dropout2d(p = P_DROPOUT),
    # Output layer
    nn.ConvTranspose2d(in_channels = DIM_HIDDEN, out_channels = CHANNEL_COUNT, kernel_size = (3, 4), stride = (1, 2), padding = (1, 0), bias = False),
    nn.Tanh()
)

layersDis = nn.Sequential(
    # 1st layer
    nn.Conv2d(in_channels = CHANNEL_COUNT, out_channels = DIM_HIDDEN, kernel_size = (3, 4), stride = (1, 2), padding = (1, 0), bias = False),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # 2nd layer
    nn.Conv2d(in_channels = DIM_HIDDEN, out_channels = 2*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = 2*DIM_HIDDEN),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # 3rd layer
    nn.Conv2d(in_channels = 2*DIM_HIDDEN, out_channels = 4*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = 4*DIM_HIDDEN),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # 4th layer
    nn.Conv2d(in_channels = 4*DIM_HIDDEN, out_channels = 8*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
    nn.BatchNorm2d(num_features = 8*DIM_HIDDEN),
    nn.LeakyReLU(negative_slope = 0.2, inplace = True),
    # Output layer
    nn.Conv2d(in_channels = 8*DIM_HIDDEN, out_channels = 1, kernel_size = (24, 22), stride = (1, 1), padding = (0, 0), bias = False),
    nn.Sigmoid()
)