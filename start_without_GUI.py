from pathlib import Path
from model.params import params
import pandas as pd
from model.data_manip import get_sep
from main import run

####################################################################################################

# Project name
PROJECT_NAME = 'test'

# Input file path
INPUT_PATH = Path.cwd() / 'data' / 'sample_data' / 'opendata_fluvius' / 'P6269_1_50_DMK_Sample_Elek_Volume_Afname_kWh_HP_resampled.csv'

# Output file format ('npy', 'csv' or 'xlsx')
OUTPUT_FORMAT = '.npy'

# Use Wandb (if True, metric will be tracked online; Wandb account required)
USE_WANDB = False

# Set the number of epochs
EPOCH_COUNT = 100

# Change the result save frequency; save all samples/models in addition to visualizations
SAVE_FREQ = 1000
SAVE_SAMPLES = False
SAVE_MODELS = False

####################################################################################################

# Model state path (optional, for continuation of training or generation of data)
MODEL_PATH = None
MODEL_PATH = r'C:\Users\Arbeit\Projekte\Git\GAN\runs\test\2024_12_19_102947716\models\epoch_100\epoch_100.pt.gz'

# Create synthetic data from existing model (if True, there is no training)
CREATE_DATA = False

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