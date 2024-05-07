
# Synthetic Load Profile GAN

In this project a GAN was developed to synthesize electricity load profiles.

## Getting started

### Prerequisites

The recommended Python version for running this code is 3.11
(libraries like Pillow might not be compatible with Python versions > 3.11).

#### Installation

1) clone the repository to your local machine:

```
git clone https://github.com/MODERATE-Project/YourToolName.git
```

2) Navigate to the project directory:

```
cd YourToolName
```

3) create an enviroment:
Conda command for creating a suitable environment (replace myenv with the desired enviroment name):

```
conda create --name myenv python=3.11
```

4) Install required Python packages:

```
conda install pip
pip install -r requirements.txt
```

## Running the GAN to train a model

To run the code, run the start.py file.

````
python start.py
````
By default the model will use the data from the VITO folder to train the model on the profiles in this folder.
To use the model on other data eg. the data in the Enercoop folder, uncomment the respective import statement in the start.py file and comment out the import from the VITO folder.

```
from ENERCOOP.X_train import X_trainResh, X_train
from ENERCOOP.params import params

# from VITO.X_train import X_trainResh, X_train
# from VITO.params import params
```

To use other data, than the two datasets provided the code needs to be adapted accordingly and is explained in the following.

## Usage
By running the start.py script the model will be trained on the data you provided based on the hyperparameters defined in the respective params.py file in each folder for the data (in this case the VITO or ENERCOOP folder).

#### Training parameters

Hyperparameters include the following:

- trackProgress: if set to true, the training progress will be tracked and can be visualized. This slows down the training.
- batchSize: the batchsize which is used for training. A high batchsize will speed up the training and generalize the learned data, a small batchsize will result in more defined profiles but can lead to overfitting.
- lossFct: We use the binary cross entropy loss function (BCE). If another loss function is choosen, additional adaptions to the code might be needed.
- lrGen: define the learning rate of the Generator. 
- lrDis: define the learning rate of the Discriminator.
- betas: We use the AdamOptimizor in both the Generator and Discriminator. The beta values define the moving averages.
- device: devine if you want to use GPU or CPU for training. For automatic detection of GPU leave the standard value which is: ```torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')```
- epochCount: Amount of epochs for training. With epochs too low the GAN can not learn the real distribution, with too many training epochs the Generator tends to outperform the Discriminator leading to wore results as well. For the provided data we found 400 epochs to be a sufficient good value.
- modelSaveFreq = 500
- loopCountGen = 5
- thresh = None
- threshEpochMin = 100

#### using wandb library

to track the training process the [wandb](https://wandb.ai/) library can be used. To do so, follow the set up guide of wandb. In the start.py file you can initalize your wandb run and devine the name of your project and if you want to see the training process online.

```
wandb.init( 
        project = 'GAN',    #set the wandb project where this run will be logged
        mode = 'offline',
        config = wandbHyperparams    #track hyperparameters and run metadata
    )
```



## Licensing

include the license here
