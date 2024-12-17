from pathlib import Path
from datetime import datetime
import pandas as pd
import wandb

from model.params import params
from model.main import GAN, export_synthetic_data, generate_data_from_saved_model
from model.data_manip import get_sep

####################################################################################################

# Project name
PROJECT_NAME = 'enercoop_wo_outliers_alpha015'

# Input file path
INPUT_PATH = Path.cwd() / 'data' / 'enercoop' / 'ENERCOOP_1year_noOutliers_alpha0,15.csv'

# Model state path (optional, for continuation of training or generation of data)
MODEL_PATH = None
#MODEL_PATH = r''

# Create synthetic data from existing model (if True, there is no training)
CREATE_DATA = False

# Output file format ('npy', 'csv' or 'xlsx')
OUTPUT_FILE_FORMAT = '.npy'

# Use Wandb (if True, metric will be tracked online; Wandb account required)
USE_WANDB = False

# Set the number of epochs
EPOCH_COUNT = 750

# Change the result save frequency; save all samples/models
RESULT_SAVE_FREQ = 50
SAVE_SAMPLES = False
SAVE_MODELS = False

####################################################################################################

if __name__ == '__main__':
    if not CREATE_DATA:
        wandb.init(
            project = 'GAN',
            mode = 'online' if USE_WANDB else 'offline'
        )
        modelName = wandb.run.name
        runNameTSSuffix = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]   #added to the end of the run name
        runName = f'{modelName}_{runNameTSSuffix}' if len(modelName) > 0 else runNameTSSuffix
        outputPath = Path().absolute() / 'runs' / PROJECT_NAME / runName
        outputPath.mkdir(parents = True, exist_ok = True)
        X_train = pd.read_csv(INPUT_PATH, sep = get_sep(INPUT_PATH))
        X_train = X_train.set_index(X_train.columns[0])
        params['epochCount'] = EPOCH_COUNT
        params['resultSaveFreq'] = RESULT_SAVE_FREQ
        params['saveSamples'] = SAVE_SAMPLES
        params['saveModels'] = SAVE_MODELS
        params['outputFileFormat'] = OUTPUT_FILE_FORMAT

        model = GAN(
            dataset = X_train,
            outputPath = outputPath,
            params = params,
            wandb = wandb,
            modelStatePath = MODEL_PATH
        )
        model.train(None)

    else:
        outputPath = Path(MODEL_PATH).parent.parent.parent / 'sample_data' / Path(MODEL_PATH).parent.name
        outputPath.mkdir(parents = True, exist_ok = True)
        X_synth = generate_data_from_saved_model(MODEL_PATH)
        export_synthetic_data(X_synth, outputPath, OUTPUT_FILE_FORMAT)