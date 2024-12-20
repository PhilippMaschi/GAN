import wandb
from datetime import datetime
from pathlib import Path
from model.main import GAN, generate_data_from_saved_model, export_synthetic_data


def run(params, projectName, inputFile, useWandb, modelPath, createData, useMarimo = False):
    if modelPath and createData:
        outputPath = Path(modelPath).parent.parent.parent / 'sample_data' / Path(modelPath).parent.name
        outputPath.mkdir(parents = True, exist_ok = True)
        X_synth = generate_data_from_saved_model(modelPath)
        export_synthetic_data(X_synth, outputPath, params['outputFormat'])
    else:
        wandb.init(
            project = 'GAN',
            mode = 'online' if useWandb else 'offline'
        )
        modelName = wandb.run.name
        runNameTSSuffix = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]   #added to the end of the run name
        runName = f'{modelName}_{runNameTSSuffix}' if len(modelName) > 0 else runNameTSSuffix
        outputPath = Path().absolute() / 'runs' / projectName / runName
        outputPath.mkdir(parents = True, exist_ok = True)
        model = GAN(
            dataset = inputFile,
            outputPath = outputPath,
            params = params,
            wandb = wandb,
            modelStatePath = modelPath,
            useMarimo = useMarimo
        )
        model.train()