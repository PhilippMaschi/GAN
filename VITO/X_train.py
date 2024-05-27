from pathlib import Path

from VITO.preproc import data_preparation_wrapper

####################################################################################################

path = Path().absolute() / 'data' / 'VITO' / 'Time-series end use consumer profiles'
folder = '100 Renovated Building with PV and HP'
filename = 'electricityConsumptionrPVHPRen.csv'
#filename = 'domesticHotWaterConsumptionPVHPRen.csv'

X_trainResh, X_train = data_preparation_wrapper(path, folder, filename)