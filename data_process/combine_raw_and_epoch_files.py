"""
1. Read wrist raw data file
2. Aggregate into time interval - epoch
3. Calculate time domain statistical parameters - eg. vm, max, min, percentiles, etc.
4. Calculate frequency domain statistical parameters
5. Combine statistical parameters with ActiGraph processed wrist-waist epoch counts
"""

import numpy as np
import pandas as pd
import sys

wrist_raw_data_filename = "D:/Accelerometer Data/LSM2/Week 1/Wednesday/LSM255 Wrist (2016-11-01)RAW.csv"
wrist_raw_data_filename = wrist_raw_data_filename.replace('\\', '/')
epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs\LSM2\Week 1\Wednesday/processed/LSM255_LSM2_Week_1_Wednesday_(2016-11-01).csv"
epoch_filename = epoch_filename.replace('\\', '/')
path_components = wrist_raw_data_filename.split('/')

output_path = "D:/Accelerometer Data/Processed"
output_path = output_path + '/' + path_components[2] + '/' + path_components[3] + '/' + path_components[4]
filename_components = path_components[5].split(' ')

# epoch granularity
n = 1500
starting_row = 360000*5
end_row = 360000*10
epoch_start = int(starting_row/n)

start = starting_row + 10
row_count = end_row - starting_row

output_filename = output_path + '/' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv', '_') \
                  + 'row_' + str(int(starting_row/n)) + '_to_' + str(int(end_row/n)) + '.csv'

print("Duration:", ((end_row-starting_row)/(100*3600)), "hours")
print("Reading raw data file.")

raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count)

raw_data_wrist.columns = ['X', 'Y', 'Z']

raw_data_wrist['X'] = raw_data_wrist['X'].abs()
raw_data_wrist['Y'] = raw_data_wrist['Y'].abs()
raw_data_wrist['Z'] = raw_data_wrist['Z'].abs()

"""
Calculate the statistical inputs (Features)
"""
print("Calculating statistical parameters.")

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]
aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm']


def getWristSD(row):
    return (row.vm - aggregated_wrist['mvm'][int(row.name/n)]) ** 2

raw_data_wrist['sd'] = raw_data_wrist.apply(getWristSD, axis=1)

aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm', 'sdvm']
wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)

del aggregated_wrist['X']
del aggregated_wrist['Y']
del aggregated_wrist['Z']

aggregated_wrist['maxvm'] = wrist_grouped_temp.max()['vm']
aggregated_wrist['minvm'] = wrist_grouped_temp.min()['vm']
aggregated_wrist['10perc'] = wrist_grouped_temp.quantile(.1)['vm']
aggregated_wrist['25perc'] = wrist_grouped_temp.quantile(.25)['vm']
aggregated_wrist['50perc'] = wrist_grouped_temp.quantile(.5)['vm']
aggregated_wrist['75perc'] = wrist_grouped_temp.quantile(.75)['vm']
aggregated_wrist['90perc'] = wrist_grouped_temp.quantile(.9)['vm']


# frequency domain features




"""
Include the epoch counts for 15 seconds and CPM values in aggregated dataframe
"""
print("Combining with ActiGraph processed epoch count data as target variables")
epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_wrist), usecols=[1, 2, 3, 4, 5])
epoch_data.columns = ['wrist_vm_15', 'wrist_vm_60',	'waist_eq_wrist_vm_60',	'waist_vm_60', 'waist_intensity']

epoch_data['wrist_mvm'] = aggregated_wrist['mvm']
epoch_data['wrist_sdvm'] = aggregated_wrist['sdvm']
epoch_data['wrist_maxvm'] = aggregated_wrist['maxvm']
epoch_data['wrist_minvm'] = aggregated_wrist['minvm']
epoch_data['wrist_10perc'] = aggregated_wrist['10perc']
epoch_data['wrist_25perc'] = aggregated_wrist['25perc']
epoch_data['wrist_50perc'] = aggregated_wrist['50perc']
epoch_data['wrist_75perc'] = aggregated_wrist['75perc']
epoch_data['wrist_90perc'] = aggregated_wrist['90perc']


# Save file
epoch_data.to_csv(output_filename, sep=',')
print("File saved as", output_filename)
