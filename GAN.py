import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch as t
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType, Normalization

from import_data import df

train_df = df[["Date", "Consumed energy [Wh]"]]

fig = plt.figure(figsize=(8,4))
plt.plot(train_df["Date"], train_df["Consumed energy [Wh]"])
plt.xticks(rotation=90)
plt.ylabel("Consumed energy")
plt.xlabel("Date")
plt.show()
