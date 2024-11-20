import PySimpleGUI as sg
from datetime import datetime
import pandas as pd
import wandb
from pathlib import Path

from model.params import params
from model.main import GAN, export_synthetic_data, generate_data_from_saved_model
from model.plots import plot_wrapper


DEFAULT_PROJECT_NAME = 'project_1'
DEFAULT_INPUT_PATH = r''
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
                    [sg.Text('Enter a model path (optional):')],
                    [sg.Radio('Continue training', group_id = 1, default = True), sg.Radio('Generate profiles', group_id = 1)],
                    [sg.InputText(default_text = DEFAULT_MODEL_PATH, s = 75)],
                    [sg.Text('Select an output file format:')],
                    [sg.Combo(['.npy', '.csv', '.xlsx'], default_value = DEFAULT_OUTPUT_FILE_FORMAT)],
                    [sg.Text('Wandb:')],
                    [sg.Combo(['off', 'on'], default_value = DEFAULT_WANDB_MODE)]
                ]),
                sg.Tab('Advanced', [
                    [sg.Col([
                        [sg.Text('Number of epochs:')],
                        [sg.InputText(default_text = params['epochCount'], s = (25))],
                        [sg.Text('Batch size:')],
                        [sg.InputText(default_text = params['batchSize'], s = (25))],
                        [sg.Text('Generator learning rate:')],
                        [sg.InputText(default_text = params['lrGen'], s = (25))]
                    ]),
                    sg.Col([
                        [sg.Text('Discriminator learning rate:')],
                        [sg.InputText(default_text = params['lrDis'], s = (25))],
                        [sg.Text('Model save frequency:')],
                        [sg.InputText(default_text = params['modelSaveFreq'], s = (25))],
                        [sg.Text('Track progress (memory-intensive):')],
                        [sg.Combo(['off', 'on'], default_value = params['trackProgress'])]
                    ])]
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
        elif event == 'Run' and (values[2] or len(values[4]) == 0):
            wandb.init(
                project = 'GAN',
                mode = values[6] + 'line'
            )
            modelName = wandb.run.name
            runNameTSSuffix = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]   #added to the end of the run name
            runName = f'{modelName}_{runNameTSSuffix}' if len(modelName) > 0 else runNameTSSuffix
            outputPath = Path().absolute() / 'runs' / values[0] / runName
            outputPath.mkdir(parents = True, exist_ok = True)
            X_train = pd.read_csv(values[1])
            X_train = X_train.set_index(X_train.columns[0])

            params['epochCount'] = int(values[7])
            params['batchSize'] = int(values[8])
            params['lrGen'] = float(values[9])
            params['lrDis'] = float(values[10])
            params['modelSaveFreq'] = int(values[11])
            params['trackProgress'] = True if values[12] == 'on' else False

            model = GAN(
                dataset = X_train,
                outputPath = outputPath,
                params = params,
                wandb = wandb,
                modelStatePath = None if len(values[4]) == 0 else values[4],
            )
            model.train(window)
            X_synth = model.generate_data()
            window['PROGRESS'].update(current_count = 19)
            export_synthetic_data(X_synth, outputPath, values[5])
            window['PROGRESS'].update(current_count = 20)
            plot_wrapper(X_train, X_synth, outputPath)
            window['PROGRESS'].update(current_count = 21)
            break
        elif event == 'Run' and values[3]:
            outputPath = Path(values[4]).parent.parent / Path(values[4]).name[:-3] / 'generated_profiles'
            outputPath.mkdir(parents = True, exist_ok = True)
            window['PROGRESS'].update(current_count = 7)
            X_synth = generate_data_from_saved_model(values[4])
            window['PROGRESS'].update(current_count = 14)
            export_synthetic_data(X_synth, outputPath, values[5], 'synth_profiles')
            window['PROGRESS'].update(current_count = 21)
            break