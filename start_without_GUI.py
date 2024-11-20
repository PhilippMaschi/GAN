# Project name
PROJECT_NAME = 'project_1'

# Input file path
INPUT_PATH = r'data\sample_data\opendata_fluvius\P6269_1_50_DMK_Sample_Elek_Volume_Afname_kWh_HP_resampled.csv'

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
EPOCH_COUNT = 5

# Change the model save frequency
MODEL_SAVE_FREQ = 50
TRACK_PROGRESS = 'on'

####################################################################################################

from datetime import datetime
import pandas as pd
import wandb
from pathlib import Path

from model.params import params
from model.main import GAN, export_synthetic_data, generate_data_from_saved_model
from model.data_manip import get_sep


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
        params['modelSaveFreq'] = MODEL_SAVE_FREQ
        params['trackProgress'] = TRACK_PROGRESS

        model = GAN(
            dataset = X_train,
            outputPath = outputPath,
            params = params,
            wandb = wandb,
            modelStatePath = MODEL_PATH
        )
        
        model.train(None)
        X_synth = model.generate_data()
        export_synthetic_data(X_synth, outputPath, OUTPUT_FILE_FORMAT)

    else:
        outputPath = Path(MODEL_PATH).parent.parent / Path(MODEL_PATH).name[:-6] / 'generated_profiles'
        outputPath.mkdir(parents = True, exist_ok = True)
        X_synth = generate_data_from_saved_model(MODEL_PATH)
        export_synthetic_data(X_synth, outputPath, OUTPUT_FILE_FORMAT, 'synth_profiles')