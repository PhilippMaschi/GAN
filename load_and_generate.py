from pathlib import Path
import torch
import numpy as np

from GAN import generate_data_from_any_saved_model

####################################################################################################

PROJECT_NAME = 'VITO'
RUN_NAME = '2024_05_27_133927705'
MODEL_NAME = 'epoch_100.pt'
OUTPUT_FILENAME = f'new_synth_profiles.npy'

runPath = Path().absolute() / 'runs' / PROJECT_NAME / RUN_NAME

from VITO.preproc import revert_reshape_arr #project-specific, needed for restructuring generator output

####################################################################################################

model = torch.load(runPath / 'models' / MODEL_NAME)

X_synth = generate_data_from_any_saved_model(
    model = model,
    runPath = runPath
)
X_synth = revert_reshape_arr(X_synth)
np.save(file = runPath / OUTPUT_FILENAME, arr = X_synth)