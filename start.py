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
        [
            sg.TabGroup([[
                sg.Text('Options', ),
                sg.Tab('Basic', [
                    [sg.Text('Enter a project name:')],
                    [sg.InputText(default_text = DEFAULT_PROJECT_NAME, s = (75))],
                    [sg.Text('Enter an input file path:')],
                    [sg.InputText(default_text = DEFAULT_INPUT_PATH, s = 75)],
                    [sg.Text('Enter a model path to continue training (optional):')],
                    [sg.InputText(default_text = DEFAULT_MODEL_PATH, s = 75)],
                    [sg.Text('Select an output file format:')],
                    [sg.Combo(['.npy', '.csv', '.xlsx'], default_value = DEFAULT_OUTPUT_FILE_FORMAT)],
                    [sg.Text('Wandb:')],
                    [sg.Combo(['off', 'on'], default_value = DEFAULT_WANDB_MODE)]
                ]),
                sg.Tab('Advanced', [
                    [sg.Text('Number of epochs:')],
                    [sg.InputText(default_text = params['epochCount'], s = (25))],
                    [sg.Text('Batch size:')],
                    [sg.InputText(default_text = params['batchSize'], s = (25))],
                    [sg.Text('Generator learning rate:')],
                    [sg.InputText(default_text = params['lrGen'], s = (25))],
                    [sg.Text('Discriminator learning rate:')],
                    [sg.InputText(default_text = params['lrDis'], s = (25))],
                    [sg.Text('Model save frequency:')],
                    [sg.InputText(default_text = params['modelSaveFreq'], s = (25))],
                    [sg.Text('Track progress (memory-intensive):')],
                    [sg.Combo(['off', 'on'], default_value = params['trackProgress'])]
                ]),
            ]])
        ],
        [sg.Button('Run'), sg.Button('Cancel')],
        [sg.ProgressBar(21, key = 'PROGRESS', s = (42, 12))]
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

            params['epochCount'] = int(values[5])
            params['batchSize'] = int(values[6])
            params['lrGen'] = float(values[7])
            params['lrDis'] = float(values[8])
            params['modelSaveFreq'] = int(values[9])
            params['trackProgress'] = True if values[10] == 'on' else False

            model = GAN(
                dataset = X_train,
                outputPath = outputPath,
                params = params,
                wandb = wandb,
                modelStatePath = None if len(values[2]) == 0 else values[2],
            )
            model.train(window)
            X_synth = model.generate_data()
            window['PROGRESS'].update(current_count = 19)
            export_synthetic_data(X_synth, outputPath, values[3])
            window['PROGRESS'].update(current_count = 20)
            plot_wrapper(X_train, X_synth, outputPath)
            window['PROGRESS'].update(current_count = 21)
            break