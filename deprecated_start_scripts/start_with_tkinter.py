DEFAULT_PROJECT_NAME = 'project_1'
DEFAULT_INPUT_PATH = r''

####################################################################################################

import tkinter as tk
from tkinter import ttk
from datetime import datetime
import pandas as pd
import wandb
from pathlib import Path

from model.params import params
from model.main import GAN, export_synthetic_data, generate_data_from_saved_model
from model.plots import plot_wrapper

####################################################################################################

def add_input_field(defaultValue, tab, width = 105):
    var = tk.StringVar()
    var.set(defaultValue)
    entry = tk.Entry(tab, textvariable = var, width = width)
    return entry


def add_dropdown_menu(tab, options):
    text = tk.StringVar()
    menu = ttk.Combobox(tab, textvariable = text.get())
    menu['values'] = options
    menu.current(0)
    return menu

####################################################################################################

root = tk.Tk()
root.geometry('640x480')
root.title('GAN')

tabs = ttk.Notebook(root)
basic_tab = ttk.Frame(tabs)
advanced_tab = ttk.Frame(tabs)
tabs.add(basic_tab, text = 'Basic')
tabs.add(advanced_tab, text = 'Advanced')
tabs.pack(expand = 1, fill = 'both')

####################################################################################################

# Basic tab
# Project name
projectName_title = tk.Label(basic_tab, text = 'Enter a project name:')
projectName_title.grid(row = 0, column = 0, sticky = 'w')
projectName_entry = add_input_field(DEFAULT_PROJECT_NAME, basic_tab)
projectName_entry.grid(row = 1, column = 0, sticky = 'w', pady = (0, 10))
# Input file path
inputPath_title = tk.Label(basic_tab, text = 'Enter an input file path:')
inputPath_title.grid(row = 2, column = 0, sticky = 'w')
inputPath_entry = add_input_field(DEFAULT_INPUT_PATH, basic_tab)
inputPath_entry.grid(row = 3, column = 0, sticky = 'w', pady = (0, 10))
# Model path
modelPath_title = tk.Label(basic_tab, text = 'Enter a model path (optional):')
modelPath_title.grid(row = 4, column = 0, sticky = 'w')
modelPathRadioButton_frame = tk.Frame(basic_tab)
modelPathRadioButton_frame.grid(row = 5, column = 0, sticky = 'w')
modelPathRadio = tk.IntVar()
modelPathContTraining_button = tk.Radiobutton(modelPathRadioButton_frame, text = 'Continue training', variable = modelPathRadio, value = 1)
modelPathContTraining_button.grid(row = 0, column = 0, sticky = 'w')
modelPathGenProfiles_button = tk.Radiobutton(modelPathRadioButton_frame, text = 'Generate profiles', variable = modelPathRadio, value = 2)
modelPathGenProfiles_button.grid(row = 0, column = 1, sticky = 'w')
modelPath_entry = add_input_field('', basic_tab)
modelPath_entry.grid(row = 6, column = 0, sticky = 'w', pady = (0, 10))
# Output file format
outputFormat_title = tk.Label(basic_tab, text = 'Select an output file format:')
outputFormat_title.grid(row = 7, column = 0, sticky = 'w')
outputFormat_menu = add_dropdown_menu(basic_tab, ['.npy', '.csv', '.xslx'])
outputFormat_menu.grid(row = 8, column = 0, sticky = 'w', pady = (0, 10))
# Wandb
wandb_title = tk.Label(basic_tab, text = 'Wandb:')
wandb_title.grid(row = 9, column = 0, sticky = 'w')
wandb_menu = add_dropdown_menu(basic_tab, ['off', 'on'])
wandb_menu.grid(row = 10, column = 0, sticky = 'w', pady = (0, 10))

# Advanced tab
# Epoch count
epochCount_title = tk.Label(advanced_tab, text = 'Number of epochs:')
epochCount_title.grid(row = 0, column = 0, sticky = 'w')
epochCount_entry = add_input_field(params['epochCount'], advanced_tab, 25)
epochCount_entry.grid(row = 1, column = 0, sticky = 'w', padx = (0, 10), pady = (0, 10))
# Batch size
batchSize_title = tk.Label(advanced_tab, text = 'Batch size:')
batchSize_title.grid(row = 2, column = 0, sticky = 'w')
batchSize_entry = add_input_field(params['batchSize'], advanced_tab, 25)
batchSize_entry.grid(row = 3, column = 0, sticky = 'w', pady = (0, 10))
# Generator learning rate
lrGen_title = tk.Label(advanced_tab, text = 'Generator learning rate:')
lrGen_title.grid(row = 4, column = 0, sticky = 'w')
lrGen_entry = add_input_field(params['lrGen'], advanced_tab, 25)
lrGen_entry.grid(row = 5, column = 0, sticky = 'w', pady = (0, 10))
# Discriminator learning rate
lrDis_title = tk.Label(advanced_tab, text = 'Discriminator learning rate:')
lrDis_title.grid(row = 0, column = 1, sticky = 'w')
lrDis_entry = add_input_field(params['lrDis'], advanced_tab, 25)
lrDis_entry.grid(row = 1, column = 1, sticky = 'w', pady = (0, 10))
# Model save frequency
modelSaveFreq_title = tk.Label(advanced_tab, text = 'Model save frequency:')
modelSaveFreq_title.grid(row = 2, column = 1, sticky = 'w')
modelSaveFreq_entry = add_input_field(params['modelSaveFreq'], advanced_tab, 25)
modelSaveFreq_entry.grid(row = 3, column = 1, sticky = 'w', pady = (0, 10))
# Track progress
trackProgress_title = tk.Label(advanced_tab, text = 'Track progress (memory-intensive):')
trackProgress_title.grid(row = 4, column = 1, sticky = 'w')
trackProgress_menu = add_dropdown_menu(advanced_tab, ['off', 'on'])
trackProgress_menu.grid(row = 5, column = 1, sticky = 'w', pady = (0, 10))

# Bottom
bottom_frame = tk.Frame(root)
bottom_frame.pack(pady = (10, 10))
run_button = tk.Button(bottom_frame, text = 'Run', command = lambda: run())
run_button.grid(row = 0, column = 0, padx = (0, 2.5))
cancel_button = tk.Button(bottom_frame, text = 'Cancel', command = lambda: root.destroy())
cancel_button.grid(row = 0, column = 1, padx = (2.5, 0))
global progress
progress = ttk.Progressbar(root, length = 620)
progress.pack(pady = (0, 10))

####################################################################################################

def run():
    if (int(modelPathRadio.get()) != 2 or len(modelPath_entry.get()) == 0):
        wandb.init(
            project = 'GAN',
            mode = wandb_menu.get() + 'line'
        )
        modelName = wandb.run.name
        runNameTSSuffix = datetime.today().strftime('%Y_%m_%d_%H%M%S%f')[:-3]   #added to the end of the run name
        runName = f'{modelName}_{runNameTSSuffix}' if len(modelName) > 0 else runNameTSSuffix
        outputPath = Path().absolute() / 'runs' / projectName_entry.get() / runName
        outputPath.mkdir(parents = True, exist_ok = True)
        X_train = pd.read_csv(inputPath_entry.get())
        X_train = X_train.set_index(X_train.columns[0])

        params['epochCount'] = int(epochCount_entry.get())
        params['batchSize'] = int(batchSize_entry.get())
        params['lrGen'] = float(lrGen_entry.get())
        params['lrDis'] = float(lrDis_entry.get())
        params['modelSaveFreq'] = int(modelSaveFreq_entry.get())
        params['trackProgress'] = True if trackProgress_menu.get() == 'on' else False

        model = GAN(
            dataset = X_train,
            outputPath = outputPath,
            params = params,
            wandb = wandb,
            modelStatePath = None if (len(modelPath_entry.get()) == 0 or int(modelPathRadio.get()) != 1) else modelPath_entry.get(),
        )
        model.train(progress, root)
        X_synth = model.generate_data()
        progress['value'] = 80
        root.update()
        export_synthetic_data(X_synth, outputPath, outputFormat_menu.get())
        progress['value'] = 90
        root.update()
        plot_wrapper(X_train, X_synth, outputPath)
        progress['value'] = 100
        root.update()
        root.destroy()
    elif int(modelPathRadio.get()) == 2:
        outputPath = Path(modelPath_entry.get()).parent.parent / Path(modelPath_entry.get()).name[:-3] / 'generated_profiles'
        outputPath.mkdir(parents = True, exist_ok = True)
        progress['value'] = 33
        root.update()
        X_synth = generate_data_from_saved_model(modelPath_entry.get())
        progress['value'] = 66
        root.update()
        export_synthetic_data(X_synth, outputPath, outputFormat_menu.get(), 'synth_profiles')
        progress['value'] = 100
        root.update()
        root.destroy()

####################################################################################################

root.mainloop()