import marimo as mo
from io import StringIO
from datetime import datetime
import pandas as pd
import wandb
from pathlib import Path

from model.params import params
from model.main import GAN, export_synthetic_data, generate_data_from_saved_model
from model.data_manip import get_sep_marimo
from model.plots import plot_wrapper


# Functions
def read_data(file):
    str_ = str(file.contents(), 'utf-8')
    data = StringIO(str_)
    df = pd.read_csv(data, sep = get_sep_marimo(data))
    df = df.set_index(df.columns[0])
    return df


# Default values
DEFAULT_PROJECTNAME = 'project_1'
DEFAULT_EPOCHCOUNT = params['epochCount']
DEFAULT_BATCHSIZE = params['batchSize']
DEFAULT_LRGEN = params['lrGen']
DEFAULT_LRDIS = params['lrDis']
DEFAULT_MODELSAVEFREQ = params['modelSaveFreq']


# Create project
projectName = mo.ui.text(label = 'Enter a project name:', value = DEFAULT_PROJECTNAME)
outputDir = mo.ui.file_browser(label = 'Select a project directory:', selection_mode = 'directory', multiple = False)


# Basic options
inputFile = mo.ui.file(label = 'Data', filetypes = ['.csv'])
modelFile = mo.ui.file(label = 'Model (optional):')
modelRadio = mo.ui.radio(options = ['Continue training', 'Generate profiles'], value = 'Continue training', inline = True)
outputFormat = mo.ui.dropdown(options = ['.npy', '.csv', '.xslx'], value = '.npy', label = 'Select an output file format:')
useWandb = mo.ui.dropdown(options = ['off', 'on'], value = 'off', label = 'Wandb:')


# Advanced options
epochCount = mo.ui.text(label = 'Number of epochs:', value = str(DEFAULT_EPOCHCOUNT))
batchSize = mo.ui.text(label = 'Batch size:', value = str(DEFAULT_BATCHSIZE))
lrGen = mo.ui.text(label = 'Generator learning rate:', value = str(DEFAULT_LRGEN))
lrDis = mo.ui.text(label = 'Discriminator learning rate:', value = str(DEFAULT_LRDIS))
modelSaveFreq = mo.ui.text(label = 'Model save frequency:', value = str(DEFAULT_MODELSAVEFREQ))
trackProgress = mo.ui.dropdown(options = ['off', 'on'], value = 'off', label = 'Track progress (memory-intensive):')


# Basic tab
basicTab = mo.vstack([
    '\n',
    mo.plain_text('Upload files:'),
    inputFile,
    '\n',
    modelFile,
    modelRadio,
    mo.md('---'),
    outputFormat,
    useWandb
])


# Advanced tab
advancedTab = mo.vstack([
    '\n',
    epochCount,
    batchSize,
    lrGen,
    lrDis,
    modelSaveFreq,
    trackProgress
])


# Tabs
tabs = mo.ui.tabs({
    '🔧 Basic': basicTab,
    '🛠️ Advanced': advancedTab
})


# Run functions
def start():
    if (not modelFile.value or modelRadio != 'Generate profiles'):
        # Initialize wandb
        wandb.init(
            project = 'GAN',
            mode = useWandb.value + 'line'
        )

        # Create output folder
        modelName = wandb.run.name
        runNameTSSuffix = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]
        runName = f'{modelName}_{runNameTSSuffix}' if len(modelName) > 0 else runNameTSSuffix
        outputPath = Path(outputDir.path()) / projectName.value / runName
        outputPath.mkdir(parents = True, exist_ok = True)

        # Read input data
        X_train = read_data(inputFile)

        # Adjust params
        params['epochCount'] = int(epochCount.value)
        params['batchSize'] = int(batchSize.value)
        params['lrGen'] = float(lrGen.value)
        params['lrDis'] = float(lrDis.value)
        params['modelSaveFreq'] = int(modelSaveFreq.value)
        params['trackProgress'] = True if trackProgress.value == 'on' else False

        # Create & train model
        model = GAN(
            dataset = X_train,
            outputPath = outputPath,
            params = params,
            wandb = wandb,
            modelStatePath = None if (not modelFile.value or modelRadio.value != 'Continue training') else modelFile.value,
        )
        model.train()

        # Create results
        X_synth = model.generate_data()
        export_synthetic_data(X_synth, outputPath, outputFormat.value)
        #fig_comp, fig_peaks, fig_means = plot_wrapper(X_train, X_synth, outputPath, True)
        #return fig_comp, fig_peaks, fig_means


    elif modelRadio == 'Generate profiles':
        # Create output folder
        runNameTSSuffix = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]
        outputPath = Path(outputDir.path()) / runNameTSSuffix
        outputPath.mkdir(parents = True, exist_ok = True)

        # Create results
        X_synth = generate_data_from_saved_model(modelFile.contents())
        export_synthetic_data(X_synth, outputPath, outputFormat.value, 'synth_profiles')


# Run button
runButton = mo.ui.run_button(label = '## 🔘 Run')


# Menus
create_project = mo.vstack([mo.md('# Create project'), projectName, outputDir])
options = mo.vstack([mo.md('# Options'), '\n', tabs])