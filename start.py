from datetime import datetime
import wandb
import os
from pathlib import Path
import numpy as np

from GAN import min_max_scaler, GAN, generate_data_from_saved_model
from config import config_wrapper
from plots import plot_wrapper

####################################################################################################

#from ENERCOOP.X_train import X_trainResh, X_train
#from ENERCOOP.params import params

from VITO.X_train import X_trainResh, X_train
from VITO.params import params

####################################################################################################

locals().update(params)
runName = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]

####################################################################################################

wandbHyperparams = {
    'batchSize': batchSize,
    'lossFct': lossFct,
    'lrGen': lrGen,
    'lrDis': lrDis,
    'device': device,
    'epochCount': epochCount,
    'dimNoise': dimNoise,
    'dimHidden': dimHidden,
    'channelCount': channelCount,
    'betas': betas
}

###########################
########## Start ##########
###########################

if __name__ == '__main__':
    wandb.init( #start a new wandb run to track this script
        project = 'GAN',    #set the wandb project where this run will be logged
        mode = 'offline',
        config = wandbHyperparams    #track hyperparameters and run metadata
    )
    modelName = wandb.run.name
    folderName = f'{modelName}_{runName}' if len(modelName) > 0 else runName
    outputPath = Path().absolute() / 'runs' / folderName
    os.makedirs(outputPath)

    ################################################################################################

    X_trainNormd, valMin, valMax = min_max_scaler(X_trainResh)  #Normd... normalized
    minMax = np.array([valMin, valMax])
    np.save(file = outputPath / 'min_max.npy', arr = minMax)

    ################################################################################################

    model = GAN(
        dataset = X_trainNormd,
        batchSize = batchSize,
        modelGen = modelGen,
        modelDis = modelDis,
        lossFct = lossFct,
        lrGen = lrGen,
        lrDis = lrDis,
        betas = betas,
        device = device,
        epochCount = epochCount,
        dimNoise = dimNoise,
        outputPath = outputPath,
        modelSaveFreq = modelSaveFreq,
        loopCountGen = loopCountGen,
        thresh = thresh,
        threshEpochMin = threshEpochMin,
        trackProgress = trackProgress,
        wandb = wandb
    )
    config_wrapper(model, outputPath)
    wandb.watch(model)

    ################################################################################################

    model.train()
    wandb.finish()

    ################################################################################################

    X_synth = generate_data_from_saved_model(
        runPath = outputPath,
        modelGen = modelGen,
        device = device,
        profileCount = X_trainNormd.shape[0],
        dimNoise = dimNoise
    )
    X_synth = restructure_GAN_output(X_synth)
    np.save(file = outputPath / 'synth_profiles.npy', arr = X_synth)

    ################################################################################################

    plot_wrapper(X_train, X_synth, outputPath)