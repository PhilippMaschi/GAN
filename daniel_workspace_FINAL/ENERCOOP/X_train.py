from pathlib import Path

from ENERCOOP.preproc import data_preparation_wrapper, data_preparation_wrapper

####################################################################################################

inputPath = Path().absolute().parent / 'GAN_data'
print(inputPath)
inputFilename = 'all_profiles.crypt'
inputPassword = 'Ene123Elec#4'
labelsFilename = 'DBSCAN_15_clusters_labels.csv'
#clusterLabels = list(range(0, 15))
clusterLabels = [13]
maxProfileCount = None
dimData = 3

X_trainResh, X_train = data_preparation_wrapper(
    dataFilePath = inputPath / inputFilename,
    password = inputPassword,
    labelsFilePath = inputPath / labelsFilename,
    clusterLabels = clusterLabels,
    maxProfileCount = maxProfileCount
)