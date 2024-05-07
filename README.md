
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
By running the start.py script the model will be trained on the data you provided based on the hyperparameters defined in the respective params.py file in each folder for the data (in this case the VITO or ENERCOOP folder). When running the script a subfolder is created with the model name. If [wandb](#Using-wandb-library) is used a random name with the date of the start of this script is used, otherwise the foldername will be just the date. Within this folder the model will be saved as well as the result figures.

### Training parameters

Hyperparameters include the following:

- <ins>trackProgress</ins>: if set to true, the training progress will be tracked and can be visualized. This slows down the training.
- <ins>batchSize</ins>: the batchsize which is used for training. A high batchsize will speed up the training and generalize the learned data, a small batchsize will result in more defined profiles but can lead to overfitting.
- <ins>lossFct</ins>: We use the binary cross entropy loss function (BCE). If another loss function is choosen, additional adaptions to the code might be needed.
- <ins>lrGen</ins>: define the learning rate of the Generator. 
- <ins>lrDis</ins>: define the learning rate of the Discriminator.
- <ins>betas</ins>: We use the AdamOptimizor in both the Generator and Discriminator. The beta values define the moving averages.
- <ins>device</ins>: devine if you want to use GPU or CPU for training. For automatic detection of GPU leave the standard value which is: ```torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')```
- <ins>epochCount</ins>: Amount of epochs for training. With epochs too low the GAN can not learn the real distribution, with too many training epochs the Generator tends to outperform the Discriminator leading to wore results as well. For the provided data we found 400 epochs to be a sufficient good value.
- <ins>modelSaveFreq</ins>: defines for every each epoch the model should be saved. This can help if too many epochs are chosen, to use the model of an earlier training state as it might perform better without having to train again.
- <ins>loopCountGen</ins>: When training, in the beginning the Discriminator tends to outperform the Generator leading to no training effect. Therefore the Generator is trained multiple times defined by this variable. A low loopCountGen will increase training speed. A too high loopCountGen will result in overfitting of the Generator also leading to no training effect.
- <ins>thresh</ins>: If a value is provided also a thresEpochMin value has to be provided. This value has to be experimentally figured out. If this value is larger than the defined criterion of ```2*lossGen - lossDisReal - lossDisFake``` then the training can be altered. This alteration of training can be defined in the GAN.py file. For example the learning rate of the Discriminator and Generator can be adapted to keep training stable for a longer period. In our exmple in the GAN.py file we change the Dropout rate and the learning rate. This should only be done after other parameters have been tested and suffenciently well identified.

```
 if epoch > self.threshEpochMin and abs(threshCriterion) > self.thresh and self.paramChange == 0:
    self.changePoint = epoch
    self.Gen.model[3].p = self.pDropoutNew
    self.Gen.model[7].p = self.pDropoutNew
    self.Gen.model[11].p = self.pDropoutNew
    self.Gen.model[15].p = self.pDropoutNew
    self.Gen.model[19].p = self.pDropoutNew
    self.optimGen.param_groups[0]['lr']/= 2 #! generalization needed
    self.optimDis.param_groups[0]['lr']/= 2
    self.paramChange = 1
```

- <ins>threshEpochMin</ins>: if ```thres!=None``` then this variable defines after how many epochs the thres criterion should be used.

Note that we included standard values for all the parameters for our test cases. These parameters can be used as a first approximation when using the model for new data. Most likely they have to be changed to some extend to generate satisfactory results.

#### Using wandb library

to track the training process the [wandb](https://wandb.ai/) library can be used. To do so, follow the set up guide of wandb. In the start.py file you can initalize your wandb run and devine the name of your project and if you want to see the training process online.

```
wandb.init( 
        project = 'GAN',    #set the wandb project where this run will be logged
        mode = 'offline',
        config = wandbHyperparams    #track hyperparameters and run metadata
    )
```
If you dont want to use wandb explicitly with your own account leave the code as it is.

### Plots

After the training is finished plots are generated to compare generated synthetic data with the real data. The plots show the distribution of frequencies, the comparison of mean values and peak values.

## Licensing

include the license here
