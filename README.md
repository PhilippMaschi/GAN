
# Synthetic Load Profile GAN

In this project, a GAN for synthesizing electricity load profiles was developed.



## Getting started

#### Prerequisites

The recommended Python version for running this code is 3.11.

#### Installation

1) Clone the repository to your local machine:

```sh
git clone https://github.com/MODERATE-Project/GAN.git
```

2) Navigate to the project directory:

```sh
cd path/to/repository/GAN
```

3) Create an enviroment:
    Conda command for creating a suitable environment (replace myenv with the desired enviroment name):

```sh
conda create --name myenv python=3.11
```

4) Activate the enviroment:
    Conda command for activating the created enviroment (replace myenv with the selected name):

```sh
conda activate myenv
```

5) Install required Python packages:

```sh
conda install pip
```

```sh
pip install -r requirements.txt
```



## Preparing the input data

The input data needs to be provided in form of a CSV file.

The data should roughly cover one year (max 368 days) of hourly electricity consumption values.

Each column of the CSV file should correspond to a single profile/household.

The first column of the CSV file needs to be an index column (ideally containing timestamps).

Example:

![Example_CSV_structure](/readme/Example_CSV_structure.png)



## Creating a project and running the GAN to train a model

There are two ways to run the code:

#### Marimo notebook

A marimo notebook is provided for easily uploading files, creating projects and training models.

The notebook can be accessed by running the following command in the project directory:

```sh
marimo edit start_with_marimo.py
```

The notebook consists of two blocks:

1. "Create project": Here, the name of the project can be entered and a project directory can be selected by using the provided file browser.
2. "Options": Contains basic as well as advanced options. The "Data" button is used to upload the CSV file containing the electricity consumption profiles. A detailed description of all parameters can be found in the consequent chapter.

<span style='color:red'>**! marimo notebooks only allow file sizes up to 100 MB, for larger input files, the Python script has to be used !**</span>

After uploading the required file(s) and adjusting the settings, the program can be started by pressing the "Run" button below the options menu. A new cell displaying the training progress will appear:

![Marimo_Run](/readme/Marimo_Run.png)

#### Python script

As an alternative to the marimo notebook, a Python script ("start_without_GUI.py") can be used to create projects and train models.

Settings have to be adjusted directly in the script and file paths have to be provided for the input files. Adjustable parameters can be found at the top of the script, above the dividing line.

Advanced options are not provided here, however, a multitude of underlying parameters can be adjusted in "model" → "params.py".



## Training parameters

The following (hyper)parameters can be adjusted:

* <ins>output file format</ins>: Lets you choose between three possible file formats for the synthetic data: ".npy", ".csv" and ".xlsx".
* <ins>useWandb/Wandb</ins>: Whether or not to track certain parameters online.
* <ins>epochCount/Number of epochs</ins>: Amount of epochs for training. A low number of training epochs causes the GAN to not learn the real distribution; a very high number of training epochs might cause the Generator to outperform the Discriminator, leading to wore results as well.

- <ins>trackProgress</ins>: if set to true, the training progress will be tracked and can be visualized. This slows down the training.
- <ins>batchSize/Batch size</ins>: The batch size which is used for training. A high batch size will speed up the training and generalize the learned data, a small batch size will result in more defined profiles but can lead to overfitting.
- <ins>lrGen/Generator learning rate</ins>: Defines the learning rate of the Generator.
- <ins>lrDis/Discriminator learning rate</ins>: Defines the learning rate of the Discriminator.
- <ins>lossFct</ins>: We use the binary cross entropy loss function (BCE). If another loss function is choosen, additional adaptions to the code might be needed.
- <ins>modelSaveFreq/Model save frequency</ins>: Defines the frequency of epochs at which models should be saved.
- <ins>trackProgress/Track progress</ins>: Whether or not plots should be created whenever a model is saved.
- <ins>betas</ins>: We use the AdamOptimizor in both the Generator and Discriminator. The beta values define the moving averages.
- <ins>device</ins>: Can be used to choose between GPU or CPU for training. For automatic detection of GPU, leave the standard value, which is: ```torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')```
- <ins>loopCountGen</ins>: When a model is trained, in the beginning, the Discriminator tends to outperform the Generator, leading to no training effect. The Generator can be trained multiple times per iteration, defined by this variable. A low loopCountGen will increase training speed. A high loopCountGen might result in overfitting of the Generator.

Note that we included standard values for all the parameters for our test cases. These parameters can be used as a first approximation when using the model for new data. Most likely they have to be changed to some extend to generate satisfactory results.
