# Project name
PROJECT_NAME = 'test_VITO'

# Input file path
INPUT_PATH = r'C:\Users\Daniel\Projekte\Git\GAN\data\VITO\electricityConsumptionrPVHPRen.csv'

# Model state path (optional, for continuation of training or generation of data)
MODEL_PATH = None

# Create synthetic data from existing model (if True, there is no training)
CREATE_DATA = False

# Output file format ('npy', 'csv' or 'xlsx')
OUTPUT_FILE_FORMAT = '.csv'

# Use Wandb (if True, metric will be tracked online)
USE_WANDB = False

# Set the number of epochs
EPOCH_COUNT = 20

####################################################################################################

from datetime import datetime
import pandas as pd
import wandb
from pathlib import Path

from model.params import params
from model.main import GAN, export_synthetic_data, generate_data_from_saved_model
from model.plots import plot_wrapper


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
        X_train = pd.read_csv(INPUT_PATH)
        X_train = X_train.set_index(X_train.columns[0])
        params['epochCount'] = EPOCH_COUNT

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
        plot_wrapper(X_train, X_synth, outputPath)

    else:
        outputPath = Path(MODEL_PATH).parent.parent / Path(MODEL_PATH).name[:-3] / 'generated_profiles'
        outputPath.mkdir(parents = True, exist_ok = True)
        X_synth = generate_data_from_saved_model(MODEL_PATH)
        export_synthetic_data(X_synth, outputPath, OUTPUT_FILE_FORMAT, 'synth_profiles')