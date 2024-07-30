import PySimpleGUI as sg
from datetime import datetime
import pandas as pd
import wandb
from pathlib import Path

from model.params import params
from model.main import GAN, export_synthetic_data
from model.plots import plot_wrapper


DEFAULT_PROJECT_NAME = 'VITO'
DEFAULT_INPUT_PATH = r'C:\Users\Daniel\Projekte\Git\GAN\data\VITO\electricityConsumptionrPVHPRen.csv'
DEFAULT_MODEL_PATH = ''
DEFAULT_OUTPUT_FILE_FORMAT = '.npy'
DEFAULT_WANDB_MODE = 'off'


if __name__ == '__main__':
    layout = [
        [sg.Text('Enter a project name:'), sg.InputText(default_text = DEFAULT_PROJECT_NAME, s = (10))],
        [sg.Text('Enter an input file path:'), sg.InputText(default_text = DEFAULT_INPUT_PATH, s = 70)],
        [sg.Text('Enter a model path to continue training (optional):'), sg.InputText(default_text = DEFAULT_MODEL_PATH, s = 70)],
        [sg.Text('Select an output file format:'), sg.Combo(['.npy', '.csv', '.xlsx'], default_value = DEFAULT_OUTPUT_FILE_FORMAT)],
        [sg.Text('Wandb:'), sg.Combo(['off', 'on'], default_value = DEFAULT_WANDB_MODE)],
        [sg.Button('Run'), sg.Button('Cancel')],
        [sg.ProgressBar(params['epochCount'] + 3, key = 'PROGRESS', s = (65, 12))]
    ]

    window = sg.Window('GAN', layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        elif event == 'Run':
            wandb.init(
                project = 'GAN',
                mode = values[4] + 'line'
            )
            modelName = wandb.run.name
            runNameTSSuffix = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]   #added to the end of the run name
            runName = f'{modelName}_{runNameTSSuffix}' if len(modelName) > 0 else runNameTSSuffix
            outputPath = Path().absolute() / 'runs' / values[0] / runName
            outputPath.mkdir(parents = True, exist_ok = True)
            X_train = pd.read_csv(values[1])
            X_train = X_train.set_index(X_train.columns[0])

            model = GAN(
                dataset = X_train,
                outputPath = outputPath,
                wandb = wandb,
                modelStatePath = None if len(values[2]) == 0 else values[2]
            )
            model.train(window)
            X_synth = model.generate_data()
            window['PROGRESS'].update(current_count = params['epochCount'] + 1)
            export_synthetic_data(X_synth, outputPath, values[3])
            window['PROGRESS'].update(current_count = params['epochCount'] + 2)
            plot_wrapper(X_train, X_synth, outputPath)
            window['PROGRESS'].update(current_count = params['epochCount'] + 3)
            break