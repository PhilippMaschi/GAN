from pathlib import Path
from model.params import params
import pandas as pd
from model.data_manip import get_sep
from main import run

####################################################################################################

# Project name
PROJECT_NAME = 'test'

# Input file path
INPUT_PATH = Path.cwd() / 'data' / 'enercoop' / 'ENERCOOP_1year_noOutliers_alpha0,15.csv'

# Output file format ('npy', 'csv' or 'xlsx')
OUTPUT_FORMAT = '.npy'

# Use Wandb (if True, metric will be tracked online; Wandb account required)
USE_WANDB = False

# Set the number of epochs
EPOCH_COUNT = 2

# Change the result save frequency; save all samples/models in addition to visualizations
SAVE_FREQ = 1000
SAVE_SAMPLES = False
SAVE_MODELS = False

####################################################################################################

# Model state path (optional, for continuation of training or generation of data)
MODEL_PATH = None

# Create synthetic data from existing model (if True, there is no training)
CREATE_DATA = True

####################################################################################################

if __name__ == '__main__':
    params['outputFormat'] = OUTPUT_FORMAT
    params['epochCount'] = EPOCH_COUNT
    params['saveFreq'] = SAVE_FREQ
    params['saveSamples'] = SAVE_SAMPLES
    params['saveModels'] = SAVE_MODELS
    inputFile = pd.read_csv(INPUT_PATH, sep = get_sep(INPUT_PATH))
    inputFile = inputFile.set_index(inputFile.columns[0])
    run(params, PROJECT_NAME, inputFile, USE_WANDB, MODEL_PATH, CREATE_DATA)