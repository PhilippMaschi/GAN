from datetime import datetime
import pandas as pd
import wandb
from pathlib import Path
import os
import numpy as np

from model.main import GAN, export_synthetic_data
from model.plots import plot_wrapper

##################################################

PROJECT_NAME = 'ENERCOOP'
#WANDB_MODE = 'online'
WANDB_MODE = 'offline'
#X_TRAIN = pd.read_excel(...)
from preproc import X_TRAIN
#MODEL_STATE_PATH = ...   # Specify if training of existing model should be continued
MODEL_STATE_PATH = None
OUTPUT_FILE_FORMAT = 'npy'  # options: npy, csv, xlsx

#################
##### Start #####
#################

if __name__ == '__main__':
    wandb.init(
        project = 'GAN',
        mode = WANDB_MODE
    )
    modelName = wandb.run.name
    runNameTSSuffix = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]   #added to the end of the run name
    runName = f'{modelName}_{runNameTSSuffix}' if len(modelName) > 0 else runNameTSSuffix
    outputPath = Path().absolute() / 'runs' / PROJECT_NAME / runName
    os.makedirs(outputPath)

    model = GAN(
        dataset = X_TRAIN,
        outputPath = outputPath,
        wandb = wandb,
        modelStatePath = MODEL_STATE_PATH
    )
    model.train()
    X_synth = model.generate_data()
    export_synthetic_data(X_synth, outputPath, OUTPUT_FILE_FORMAT)
    plot_wrapper(X_TRAIN, X_synth, outputPath)