
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

To use other data, than the two datasets provided the code needs to be adapted accordingly.

## Usage

Here is a quick guide on how to use the MyTool for your data analysis:

Either describe how to use the tool if it is simple enough or provide a notebook with example code that
can be directly used.

## Licensing

include the license here
