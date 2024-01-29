import pandas as pd
print(pd.__version__)
import numpy as np
import torch
print(torch.__version__)


from data_manip import remove_incomplete_days

df_loadProfiles = pd.read_parquet(r'data/load_profiles.parquet.gzip')   #import data

df_loadProfiles = remove_incomplete_days(df_loadProfiles)

##################################################

from sklearn.preprocessing import MinMaxScaler

profile = '16'

df_profile = df_loadProfiles[['date', 'hour of the day', profile]]
df_profile = df_profile.pivot_table(columns = 'hour of the day', index = 'date', values = profile)

labels = np.array(range(len(df_profile)))
samples = df_profile.to_numpy()

scaler = MinMaxScaler(feature_range = (-1, 1))
samplesScaled = scaler.fit_transform(samples.T).T

##################################################

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(torch.Tensor(samplesScaled), torch.Tensor(labels))
dataLoader = DataLoader(dataset)

##################################################

dataset = TensorDataset(torch.Tensor(samplesScaled), torch.Tensor(labels))
dataLoader = DataLoader(dataset)

##################################################

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU is used.')
else:
    device = torch.device('cpu')
    print('CPU is used.')

name = 'model_test'
dimLatent = 32
featureCount = samplesScaled.shape[1]
classCount = len(set(labels))
dimEmbedding = classCount
lr = 2*1e-4/3
maxNorm = 1e6
epochCount = 250
#testLabel = 0

##################################################

from GAN import GAN

model = GAN(
    name = name,
    device = device,
    dataLoader = dataLoader,
    dimLatent = dimLatent,
    featureCount = featureCount,
    classCount = classCount,
    dimEmbedding = dimEmbedding,
    lr = lr,
    maxNorm = maxNorm,
    epochCount = epochCount,
    #testLabel = testLabel
)
model.train()

##################################################

torch.save(model, f'models/{model.name}.pt')