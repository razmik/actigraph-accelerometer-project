import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join


"""
If multiple files to be assessed.
"""
experiment = 'LSM2'
week = 'Week 1'
day = 'Wednesday'

path_filtered = "D:/Accelerometer Data/Processed/"+experiment+"/"+week+"/"+day+"/filtered/".replace('\\', '/')
files_filtered = [f for f in listdir(path_filtered) if isfile(join(path_filtered, f))]
data_filtered = pd.DataFrame()
for file in files_filtered:
    data_filtered = data_filtered.append(pd.read_csv(path_filtered + file), ignore_index=True)

path_not_filtered = "D:/Accelerometer Data/Processed/"+experiment+"/"+week+"/"+day+"/not_filtered/".replace('\\', '/')
files_not_filtered = [f for f in listdir(path_not_filtered) if isfile(join(path_not_filtered, f))]
data_not_filtered = pd.DataFrame()
for file in files_not_filtered:
    data_not_filtered = data_not_filtered.append(pd.read_csv(path_not_filtered + file), ignore_index=True)

"""
If only a single file needs to be assessed.
"""
# data_not_filtered = pd.read_csv('D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/not_filtered/LSM203_(2016-11-02)_row_16320_to_19200.csv'.replace('\\', '/'))
# data_filtered = pd.read_csv('D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/filtered/LSM203_(2016-11-02)_row_16320_to_19200.csv'.replace('\\', '/'))


del data_not_filtered['Unnamed: 0']
del data_filtered['Unnamed: 0']


raw_variable = 'raw_total_power'

print("Pearson <"+raw_variable+" vs. not filtered actilife_waist_intensity>:", round(stats.pearsonr(data_not_filtered[raw_variable], data_not_filtered['actilife_waist_intensity'])[0], 2))
print("Pearson <"+raw_variable+" vs. not filtered actilife_waist_ee>:", round(stats.pearsonr(data_not_filtered[raw_variable], data_not_filtered['actilife_waist_ee'])[0], 2))
print("Pearson <"+raw_variable+" vs. not filtered actilife_waist_vm_15>:", round(stats.pearsonr(data_not_filtered[raw_variable], data_not_filtered['actilife_waist_vm_15'])[0], 2))
print("Pearson <"+raw_variable+" vs. not filtered actilife_waist_vm_cpm>:", round(stats.pearsonr(data_not_filtered[raw_variable], data_not_filtered['actilife_waist_vm_cpm'])[0], 2))
print("Pearson <"+raw_variable+" vs. not filtered actilife_waist_cpm>:", round(stats.pearsonr(data_not_filtered[raw_variable], data_not_filtered['actilife_waist_cpm'])[0], 2))
print('')
print("Pearson <"+raw_variable+" vs. filtered actilife_waist_intensity>:", round(stats.pearsonr(data_filtered[raw_variable], data_filtered['actilife_waist_intensity'])[0], 2))
print("Pearson <"+raw_variable+" vs. filtered actilife_waist_ee>:", round(stats.pearsonr(data_filtered[raw_variable], data_filtered['actilife_waist_ee'])[0], 2))
print("Pearson <"+raw_variable+" vs. filtered actilife_waist_vm_15>:", round(stats.pearsonr(data_filtered[raw_variable], data_filtered['actilife_waist_vm_15'])[0], 2))
print("Pearson <"+raw_variable+" vs. filtered actilife_waist_vm_cpm>:", round(stats.pearsonr(data_filtered[raw_variable], data_filtered['actilife_waist_vm_cpm'])[0], 2))
print("Pearson <"+raw_variable+" vs. filtered actilife_waist_cpm>:", round(stats.pearsonr(data_filtered[raw_variable], data_filtered['actilife_waist_cpm'])[0], 2))
