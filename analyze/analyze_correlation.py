import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join

# path = "D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/".replace('\\', '/')
#
# files = [f for f in listdir(path) if isfile(join(path, f))]
#
# data = pd.DataFrame()
# for file in files:
#     data = data.append(pd.read_csv(path + file), ignore_index=True)

"""
If only a single file needs to be assessed.
"""
data = pd.read_csv('D:\Accelerometer Data\Processed\LSM2\Week 1\Wednesday/not_filtered/LSM204_(2016-11-02)_row_16560_to_18440.csv'.replace('\\', '/'))
del data['Unnamed: 0']

print("Pearson <parameters>:", round(stats.pearsonr(data['raw_band_vm'], data['actilife_wrist_vm_15'])[0], 2))
