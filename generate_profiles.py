import PySimpleGUI as sg
from pathlib import Path

from model.main import generate_data_from_saved_model, export_synthetic_data


DEFAULT_MODEL_PATH = r'C:\Users\Daniel\Projekte\Git\GAN\runs\VITO\2024_07_28_221501450\models\epoch_500.pt'
DEFAULT_OUTPUT_FILE_FORMAT = '.npy'


if __name__ == '__main__':
    layout = [
        [sg.Text('Enter a model path:'), sg.InputText(default_text = DEFAULT_MODEL_PATH, s = 80)],
        [sg.Text('Select an output file format:'), sg.Combo(['.npy', '.csv', '.xlsx'], default_value = DEFAULT_OUTPUT_FILE_FORMAT)],
        [sg.Button('Run'), sg.Button('Cancel')]
    ]

    window = sg.Window('Synthetic data generator', layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        elif event == 'Run':
            outputPath = Path(values[0]).parent.parent / Path(values[0]).name[:-3] / 'generated_profiles'
            outputPath.mkdir(parents = True, exist_ok = True)
            X_synth = generate_data_from_saved_model(values[0])
            export_synthetic_data(X_synth, outputPath, values[1])
            break