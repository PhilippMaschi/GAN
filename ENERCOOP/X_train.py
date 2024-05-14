from pathlib import Path

from ENERCOOP.preproc import data_preparation_wrapper, data_preparation_wrapper

####################################################################################################

inputPath = Path().absolute() / 'ENERCOOP'
print(inputPath)
inputFilename = 'all_profiles.csv'
labelsFilename = 'DBSCAN_15_clusters_labels.csv'
#clusterLabels = list(range(0, 15))
clusterLabels = [1]  # if list is empty, all labels are used so the whole dataset (not just one cluster)
maxProfileCount = None
dimData = 3

X_trainResh, X_train = data_preparation_wrapper(
    dataFilePath = inputPath / inputFilename,
    labelsFilePath = inputPath / labelsFilename,
    clusterLabels = clusterLabels,
    maxProfileCount = maxProfileCount
)