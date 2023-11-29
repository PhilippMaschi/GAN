

import cryptpandas as crp
import os,  sys, signal 
import getpass
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
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
from GAN_andi import GAN
from plot import plot_losses
from plot import plot_synthetic_vs_real_samples
import argparse

import gc

def conv_index_to_bins(index):
    """Calculate bins to contain the index values.
    The start and end bin boundaries are linearly extrapolated from 
    the two first and last values. The middle bin boundaries are 
    midpoints.

    Example 1: [0, 1] -> [-0.5, 0.5, 1.5]
    Example 2: [0, 1, 4] -> [-0.5, 0.5, 2.5, 5.5]
    Example 3: [4, 1, 0] -> [5.5, 2.5, 0.5, -0.5]"""
    assert index.is_monotonic_increasing or index.is_monotonic_decreasing

    # the beginning and end values are guessed from first and last two
    start = index[0] - (index[1]-index[0])/2
    end = index[-1] + (index[-1]-index[-2])/2

    # the middle values are the midpoints
    middle = pd.DataFrame({'m1': index[:-1], 'p1': index[1:]})
    middle = middle['m1'] + (middle['p1']-middle['m1'])/2

    if isinstance(index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(middle).union([start,end])
    elif isinstance(index, (pd.Float64Index,pd.RangeIndex,pd.Int64Index)):
        idx = pd.Float64Index(middle).union([start,end])
    else:
        print('Warning: guessing what to do with index type %s' % 
              type(index))
        idx = pd.Float64Index(middle).union([start,end])

    return idx.sort_values(ascending=index.is_monotonic_increasing)

def calc_df_mesh(df):
    """Calculate the two-dimensional bins to hold the index and 
    column values."""
    return np.meshgrid(conv_index_to_bins(df.index),
                       conv_index_to_bins(df.columns))
                       
                       

def main(password_, num_prof,  dim_emmbedded= 1000):
    ####################
    #
    # Data import
    #
    #######################
    GAN_data_path = Path().absolute().parent / 'GAN_data'
        
    df_loadProfiles = crp.read_encrypted(path = os.path.join(GAN_data_path, 'all_profiles.crypt'), password=password_)
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

    profiles = df_labels.loc[df_labels['labels'] == clusterLabel, 'name'].to_list()[:num_prof]
    print(len(profiles))
    
    some_list = [item for item in profiles if item in df_loadProfiles.columns]
    df_profiles = df_loadProfiles[df_loadProfiles.columns[:13].tolist() + some_list].copy()
    df_plot = df_profiles.iloc[:, 13:].reset_index(drop = True).copy()    #save for later
    print(f"Shape df_profiles {df_profiles.shape} ")
    df_profiles = df_profiles.melt(id_vars = df_loadProfiles.columns[:13], value_vars = df_profiles.columns[13:], var_name = 'profile')
    df_profiles = df_profiles.pivot_table(values = 'value', index = ['profile',  'date'], columns = 'hour of the day')  # Rows: 1 entry for each day and profile, columns: hour of day 
    
    fig = go.Figure(data=go.Heatmap(
        z=df_profiles,
        x=df_profiles.columns,
        y=df_profiles.index,
        colorscale='Viridis'))

    fig.update_layout(
        title='GitHub commits per day',
        xaxis_nticks=24)

    #fig.show()

    
    print(f"Shape df_profiles {df_profiles.shape} ")
    
    ####################
    #
    # Create and scale samples and labels
    #
    #######################

    samples = df_profiles.to_numpy().astype("f4")
    labels = np.tile(np.array(range(395)) ,  num_prof)    #each day gets the same label without considering which profile
    #labels = np.array(range(len(df_profiles)))  # labels is index vector from 0 to x that indicates the day

    ####################
    #
    # Configure GAN
    #
    #######################

    if 1 and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('GPU is used.')
    else:
        device = torch.device('cpu')
        print('CPU is used.')

    batchSize = int(samples.shape[0] / num_prof)
    dimLatent = 10
    featureCount = samples.shape[1]  # stunden pro tag (pro label hat das model 24 werte)
    classCount = len(set(labels))  # labels bzw anzahl der Tage times number of load profiles
    dimEmbedding = dim_emmbedded    #classCount
    lr = 1e-5
    maxNorm = 1e6
    epochCount =300
    #testLabel = 0

    ####################
    #
    # Create and run model
    #
    #######################
    """
    for obj in gc.get_objects():
        a = sys.getsizeof(obj) / 10**6
        if a > 100:
            
            print(obj)
            print(f"Type: {type(obj)}")
            print(f"Size: {a} MB")
    #sys.exit()
    """
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
    
    
    model.train()


    ####################
    #
    # Save model
    #
    #######################
    model.name = 'model_test_andi'
    torch.save(model, f'models/{model.name}.pt')
    print("Done""")

def load_model(model_name = "model_test_andi"):
    
    import torch
    import GAN
    model = torch.load(f"models/{model_name}.pt")
    array = model.generate_sample()
    df_synthProfiles = df_profiles.copy()
    df_synthProfiles[::] = array
    df2 = df_synthProfiles.reset_index().melt(id_vars=["date","profile"]).pivot_table(values="value", columns="profile", index=["date", "hour of the day"])
    df2.to_csv(f"{model_name}_synthetic.csv")
    
    df_synthProfiles
    numberOfProfiles = 90

    synthSamplesScaled_list = [model.generate_sample() for i in range(numberOfProfiles)]
    synthSamples_list = [scaler.inverse_transform(item.T).T for item in synthSamplesScaled_list]
    df_synthProfiles = df_profiles.copy()
    df_synthProfiles[::] = scaler.inverse_transform(model.generate_sample().T).T
    df_synthProfiles.iloc[5].plot()
    pd.DataFrame(scaler.inverse_transform(model.generate_sample().T).T, columns = df_profiles)
    model.generate_sample().shape

    synthSamplesScaled = np.dstack(synthSamplesScaled_list)
    synthSamples = np.dstack(synthSamples_list)

    df_profiles.iloc[5].plot()

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(
        description='launches test torch')
    parser.add_argument("--password", default= "Ene123Elec#4")
    args = parser.parse_args()
    
    password = args.password
    if password == "":
        password = getpass.getpass('Password: ')
        print("No password given. Exit now.")
        sys.exit()
    
    dim_embed = 100
    pid = (os.getpid()) 
    print(pid)
    
    
    for i in [30]:
        print(f"Size  {i=}")
        main(password,  i,  dim_embed)
        torch.cuda.empty_cache()

os.kill(pid, signal.SIGTERM) 
sys.exit()

