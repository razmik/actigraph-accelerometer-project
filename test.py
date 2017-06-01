import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import time

# waist_raw_data_filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM 146 Waist (2016-10-19)RAW.csv"
# wrist_raw_data_filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM 146 Wrist (2016-10-19)RAW.csv"
waist_raw_data_filename = "D:/Accelerometer Data/Sample/LSM246/LSM 246 Waist (2016-11-03)RAW.csv"
wrist_raw_data_filename = "D:/Accelerometer Data/Sample/LSM246/LSM 246 Wrist (2016-11-03)RAW.csv"
waist_epoch_filename = "D:/Accelerometer Data/Sample/LSM246/LSM 246 Waist (2016-11-03)15sec.csv"
wrist_epoch_filename = "D:/Accelerometer Data/Sample/LSM246/LSM 246 Wrist (2016-11-03)15sec.csv"
print(wrist_raw_data_filename)

# epoch granularity
n = 1500
starting_row = 0
end_row = 360000

start = starting_row + 10
row_count = end_row - starting_row

epoch_start = int(starting_row/n) + 10
# epoch_raw_count = int(row_count/n) #This needs to be same as aggregated dataframe row count.

raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count)
raw_data_waist = pd.read_csv(waist_raw_data_filename, skiprows=start, nrows=row_count)

raw_data_wrist.columns = ['X', 'Y', 'Z']
raw_data_waist.columns = ['X', 'Y', 'Z']

raw_data_wrist['X'] = raw_data_wrist['X'].abs()
raw_data_wrist['Y'] = raw_data_wrist['Y'].abs()
raw_data_wrist['Z'] = raw_data_wrist['Z'].abs()
raw_data_waist['X'] = raw_data_waist['X'].abs()
raw_data_waist['Y'] = raw_data_waist['Y'].abs()
raw_data_waist['Z'] = raw_data_waist['Z'].abs()

"""
Calculate the statistical inputs (Features)
"""

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]
raw_data_waist['vm'] = np.sqrt([(raw_data_waist.X ** 2) + (raw_data_waist.Y ** 2) + (raw_data_waist.Z ** 2)])[0]

aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_waist = raw_data_waist.groupby(np.arange(len(raw_data_waist))//n).mean()

aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm']
aggregated_waist.columns = ['X', 'Y', 'Z', 'mvm']


def getWristSD(row):
    return (row.vm - aggregated_wrist['mvm'][int(row.name/n)]) ** 2


def getWaistSD(row):
    return (row.vm - aggregated_waist['mvm'][int(row.name/n)]) ** 2

raw_data_wrist['sd'] = raw_data_wrist.apply(getWristSD, axis=1)
raw_data_waist['sd'] = raw_data_waist.apply(getWaistSD, axis=1)

aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_waist = raw_data_waist.groupby(np.arange(len(raw_data_waist))//n).mean()

aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm', 'sdvm']
aggregated_waist.columns = ['X', 'Y', 'Z', 'mvm', 'sdvm']

start_time = time.time()

wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)
waist_grouped_temp = raw_data_waist.groupby(raw_data_waist.index // n)

aggregated_wrist['maxvm'] = wrist_grouped_temp.max()['vm']
aggregated_waist['maxvm'] = waist_grouped_temp.max()['vm']

aggregated_wrist['minvm'] = wrist_grouped_temp.min()['vm']
aggregated_waist['minvm'] = waist_grouped_temp.min()['vm']

aggregated_wrist['10perc'] = wrist_grouped_temp.quantile(.1)['vm']
aggregated_waist['10perc'] = waist_grouped_temp.quantile(.1)['vm']

aggregated_wrist['25perc'] = wrist_grouped_temp.quantile(.25)['vm']
aggregated_waist['25perc'] = waist_grouped_temp.quantile(.25)['vm']

aggregated_wrist['50perc'] = wrist_grouped_temp.quantile(.5)['vm']
aggregated_waist['50perc'] = waist_grouped_temp.quantile(.5)['vm']

aggregated_wrist['75perc'] = wrist_grouped_temp.quantile(.75)['vm']
aggregated_waist['75perc'] = waist_grouped_temp.quantile(.75)['vm']

aggregated_wrist['90perc'] = wrist_grouped_temp.quantile(.9)['vm']
aggregated_waist['90perc'] = waist_grouped_temp.quantile(.9)['vm']

end_time = time.time()

"""
Include the epoch counts for 15 seconds and CPM values in aggregated dataframe
"""
