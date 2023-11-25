

import cryptpandas as crp
import os
import getpass
import pandas as pd
import numpy as np
import torch
print(f'torch {torch.__version__}')
from sklearn.preprocessing import minmax_scale, MinMaxScaler
import json
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from datetime import date

from data_manip import remove_incomplete_days
from preproc import import_and_preprocess_data, create_and_add_datetime_features
from GAN import GAN
from plot import plot_losses
from plot import plot_synthetic_vs_real_samples

####################
#
# Data import
#
#######################
GAN_data_path = Path().absolute().parent / 'GAN_data'

df_loadProfiles = crp.read_encrypted(path = os.path.join(GAN_data_path, 'all_profiles.crypt'), password=getpass.getpass('Password: '))
####################
#
# 
#
#######################


df_labels = pd.read_csv(os.path.join(GAN_data_path, 'DBSCAN_15_clusters_labels.csv'), sep = ';')
df_labels['name'] = df_labels['name'].str.split('_', expand = True)[1]

####################
#
# Create a dataframe for one cluster
#
#######################

clusterLabel = 1

profiles = df_labels.loc[df_labels['labels'] == clusterLabel, 'name'].to_list()[:10]
print(len(profiles))

df_profiles = df_loadProfiles[df_loadProfiles.columns[:13].tolist() + [item for item in profiles if item in df_loadProfiles.columns]].copy()
df_plot = df_profiles.iloc[:, 13:].reset_index(drop = True).copy()    #save for later

df_profiles = df_profiles.melt(id_vars = df_loadProfiles.columns[:13], value_vars = df_profiles.columns[13:], var_name = 'profile')
df_profiles = df_profiles.pivot_table(values = 'value', index = ['date', 'profile'], columns = 'hour of the day')


####################
#
# Create and scale samples and labels
#
#######################

samples = df_profiles.to_numpy()
labels = np.array(range(len(df_profiles)))

####################
#
# Configure GAN
#
#######################

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU is used.')
else:
    device = torch.device('cpu')
    print('CPU is used.')

batchSize = 1000
dimLatent = 32
featureCount = samples.shape[1]
classCount = len(set(labels))
dimEmbedding = classCount
lr = 1e-5
maxNorm = 1e6
epochCount = 1000
#testLabel = 0

####################
#
# Create and run model
#
#######################

model = GAN(
    name = "current_model",
    device = device,
    batchSize = batchSize,
    samples = samples,
    labels = labels,
    dimLatent = dimLatent,
    featureCount = featureCount,
    classCount = classCount,
    dimEmbedding = dimEmbedding,
    lr = lr,
    maxNorm = maxNorm,
    epochCount = epochCount,
    #testLabel = testLabel
)
torch.cuda.empty_cache()
model.train()


####################
#
# Save model
#
#######################
model.name = 'model_test_andi'
torch.save(model, f'models/{model.name}.pt')


