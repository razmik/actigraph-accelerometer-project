import numpy as np
import pandas as pd
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt

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
starting_row = 0
end_row = 360000*3

# summarize parameters
one_hour = 4 * 60 * n  # 1 hour = (15 seconds epoch * 4 * 60)
min_5 = n * 4 * 5
summarize_duration = min_5 * 1
processed_epoch_summarize_duration = summarize_duration / n

timeline = "5 minute epochs"

start = starting_row + 10
row_count = end_row - starting_row
epoch_start = int(starting_row/processed_epoch_summarize_duration)

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
aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//summarize_duration).mean()
aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm']


def get_wrist_sd(row):
    return (row.vm - aggregated_wrist['mvm'][int(row.name/summarize_duration)]) ** 2

raw_data_wrist['sd'] = raw_data_wrist.apply(get_wrist_sd, axis=1)

aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//summarize_duration).mean()
aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm', 'sdvm']
wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // summarize_duration)

del aggregated_wrist['X']
del aggregated_wrist['Y']
del aggregated_wrist['Z']

aggregated_wrist['mvm'] = wrist_grouped_temp['vm'].mean()
aggregated_wrist['sdvm'] = wrist_grouped_temp['vm'].std()
aggregated_wrist['maxvm'] = wrist_grouped_temp.max()['vm']
aggregated_wrist['minvm'] = wrist_grouped_temp.min()['vm']
aggregated_wrist['10perc'] = wrist_grouped_temp.quantile(.1)['vm']
aggregated_wrist['25perc'] = wrist_grouped_temp.quantile(.25)['vm']
aggregated_wrist['50perc'] = wrist_grouped_temp.quantile(.5)['vm']
aggregated_wrist['75perc'] = wrist_grouped_temp.quantile(.75)['vm']
aggregated_wrist['90perc'] = wrist_grouped_temp.quantile(.9)['vm']


"""
Include the epoch counts for 15 seconds and CPM values in aggregated dataframe
"""
print("Combining with ActiGraph processed epoch count data as target variables")
epoch_row_count = len(aggregated_wrist) * processed_epoch_summarize_duration
processed_epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=epoch_row_count, usecols=[1, 2, 3, 4, 5])
processed_epoch_data.columns = ['mean_wrist_vm_15', 'mean_wrist_vm_60',	'mean_waist_eq_wrist_vm_60', 'mean_waist_vm_60', 'mean_waist_intensity']

# summerize
summarized_wrist_epoch_data = processed_epoch_data.groupby(np.arange(len(processed_epoch_data))//processed_epoch_summarize_duration).mean()
summarized_wrist_epoch_data.columns = ['mean_wrist_vm_15', 'mean_wrist_vm_60',
                            'mean_waist_eq_wrist_vm_60', 'mean_waist_vm_60', 'mean_waist_intensity']

"""
Normalize 0-1
z = (value - min(array)) / (max(array) - min(array))
"""

normalize_wrist_cpm = (summarized_wrist_epoch_data['mean_wrist_vm_60'] - np.amin(summarized_wrist_epoch_data['mean_wrist_vm_60'])) / (np.amax(summarized_wrist_epoch_data['mean_wrist_vm_60']) - np.amin(summarized_wrist_epoch_data['mean_wrist_vm_60']))
normalize_waist_cpm = (summarized_wrist_epoch_data['mean_waist_vm_60'] - np.amin(summarized_wrist_epoch_data['mean_waist_vm_60'])) / (np.amax(summarized_wrist_epoch_data['mean_waist_vm_60']) - np.amin(summarized_wrist_epoch_data['mean_waist_vm_60']))
normalize_activity_intensity = (summarized_wrist_epoch_data['mean_waist_intensity'] - np.amin(summarized_wrist_epoch_data['mean_waist_intensity'])) / (np.amax(summarized_wrist_epoch_data['mean_waist_intensity']) - np.amin(summarized_wrist_epoch_data['mean_waist_intensity']))


normalize_wrist_mvm = (aggregated_wrist['mvm'] - np.amin(aggregated_wrist['mvm'])) / (np.amax(aggregated_wrist['mvm']) - np.amin(aggregated_wrist['mvm']))
normalize_wrist_sd = (aggregated_wrist['sdvm'] - np.amin(aggregated_wrist['sdvm'])) / (np.amax(aggregated_wrist['sdvm']) - np.amin(aggregated_wrist['sdvm']))
normalize_wrist_maxvm = (aggregated_wrist['maxvm'] - np.amin(aggregated_wrist['maxvm'])) / (np.amax(aggregated_wrist['maxvm']) - np.amin(aggregated_wrist['maxvm']))
normalize_wrist_minvm = (aggregated_wrist['minvm'] - np.amin(aggregated_wrist['minvm'])) / (np.amax(aggregated_wrist['minvm']) - np.amin(aggregated_wrist['minvm']))
normalize_wrist_10perc = (aggregated_wrist['10perc'] - np.amin(aggregated_wrist['10perc'])) / (np.amax(aggregated_wrist['10perc']) - np.amin(aggregated_wrist['10perc']))
normalize_wrist_25perc = (aggregated_wrist['25perc'] - np.amin(aggregated_wrist['25perc'])) / (np.amax(aggregated_wrist['25perc']) - np.amin(aggregated_wrist['25perc']))
normalize_wrist_50perc = (aggregated_wrist['50perc'] - np.amin(aggregated_wrist['50perc'])) / (np.amax(aggregated_wrist['50perc']) - np.amin(aggregated_wrist['50perc']))
normalize_wrist_75perc = (aggregated_wrist['75perc'] - np.amin(aggregated_wrist['75perc'])) / (np.amax(aggregated_wrist['75perc']) - np.amin(aggregated_wrist['75perc']))
normalize_wrist_90perc = (aggregated_wrist['90perc'] - np.amin(aggregated_wrist['90perc'])) / (np.amax(aggregated_wrist['90perc']) - np.amin(aggregated_wrist['90perc']))


# Correlation

print("Wrist CPM vs Mean VM:", round(stats.pearsonr(normalize_wrist_mvm, normalize_waist_cpm)[0], 2))
print("Wrist CPM vs Mean SD:", round(stats.pearsonr(normalize_wrist_sd, normalize_waist_cpm)[0], 2))
print("Wrist CPM vs Max:", round(stats.pearsonr(normalize_wrist_maxvm, normalize_waist_cpm)[0], 2))
print("Wrist CPM vs Min:", round(stats.pearsonr(normalize_wrist_minvm, normalize_waist_cpm)[0], 2))
print("Wrist CPM vs 10th perc:", round(stats.pearsonr(normalize_wrist_10perc, normalize_waist_cpm)[0], 2))
print("Wrist CPM vs 25th perc:", round(stats.pearsonr(normalize_wrist_25perc, normalize_waist_cpm)[0], 2))
print("Wrist CPM vs 50th perc:", round(stats.pearsonr(normalize_wrist_50perc, normalize_waist_cpm)[0], 2))
print("Wrist CPM vs 75th perc:", round(stats.pearsonr(normalize_wrist_75perc, normalize_waist_cpm)[0], 2))
print("Wrist CPM vs 90th perc:", round(stats.pearsonr(normalize_wrist_90perc, normalize_waist_cpm)[0], 2))

print("Activity Intensity (Hip) vs Mean VM:", round(stats.pearsonr(normalize_wrist_mvm, normalize_activity_intensity)[0], 2))
print("Activity Intensity (Hip) vs Mean SD:", round(stats.pearsonr(normalize_wrist_sd, normalize_activity_intensity)[0], 2))
print("Activity Intensity (Hip) vs Max:", round(stats.pearsonr(normalize_wrist_maxvm, normalize_activity_intensity)[0], 2))
print("Activity Intensity (Hip) vs Min:", round(stats.pearsonr(normalize_wrist_minvm, normalize_activity_intensity)[0], 2))
print("Activity Intensity (Hip) vs 10th perc:", round(stats.pearsonr(normalize_wrist_10perc, normalize_activity_intensity)[0], 2))
print("Activity Intensity (Hip) vs 25th perc:", round(stats.pearsonr(normalize_wrist_25perc, normalize_activity_intensity)[0], 2))
print("Activity Intensity (Hip) vs 50th perc:", round(stats.pearsonr(normalize_wrist_50perc, normalize_activity_intensity)[0], 2))
print("Activity Intensity (Hip) vs 75th perc:", round(stats.pearsonr(normalize_wrist_75perc, normalize_activity_intensity)[0], 2))
print("Activity Intensity (Hip) vs 90th perc:", round(stats.pearsonr(normalize_wrist_90perc, normalize_activity_intensity)[0], 2))

"""
Visualize
b : blue.
g : green.
r : red.
c : cyan.
m : magenta.
y : yellow.
k : black.
w : white.
"""

plt.figure(1)
x_range = np.arange(len(aggregated_wrist))
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Wrist CPM, Blue - Wrist Mean VM, Green - Wrist Mean SD')
plt.plot(x_range, normalize_wrist_cpm, 'r', x_range, normalize_wrist_mvm, 'b', x_range, normalize_wrist_sd, 'g')

plt.figure(2)
x_range = np.arange(len(aggregated_wrist))
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Wrist CPM, Blue - Wrist Max VM, Green - Wrist Min SD')
plt.plot(x_range, normalize_wrist_cpm, 'r', x_range, normalize_wrist_maxvm, 'b', x_range, normalize_wrist_minvm, 'g')

plt.figure(3)
x_range = np.arange(len(aggregated_wrist))
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Wrist CPM, Blue - 10th perc VM, Green - 25th perc, Magenta - 50th perc, Yellow - 75th perc, Black - 90th perc')
plt.plot(x_range, normalize_wrist_cpm, 'r', x_range, normalize_wrist_10perc, 'b', x_range, normalize_wrist_25perc, 'g', x_range, normalize_wrist_50perc, 'm', x_range, normalize_wrist_75perc, 'y', x_range, normalize_wrist_90perc, 'k')




plt.figure(4)
x_range = np.arange(len(aggregated_wrist))
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Waist Processed CPM, Blue - Wrist Mean VM, Green - Wrist Mean SD')
plt.plot(x_range, normalize_waist_cpm, 'r', x_range, normalize_wrist_mvm, 'b', x_range, normalize_wrist_sd, 'g')

plt.figure(5)
x_range = np.arange(len(aggregated_wrist))
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Waist Processed CPM, Blue - Wrist Max VM, Green - Wrist Min SD')
plt.plot(x_range, normalize_waist_cpm, 'r', x_range, normalize_wrist_maxvm, 'b', x_range, normalize_wrist_minvm, 'g')

plt.figure(6)
x_range = np.arange(len(aggregated_wrist))
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Waist Processed CPM, Blue - 10th perc VM, Green - 25th perc, Magenta - 50th perc, Yellow - 75th perc, Black - 90th perc')
plt.plot(x_range, normalize_waist_cpm, 'r', x_range, normalize_wrist_10perc, 'b', x_range, normalize_wrist_25perc, 'g', x_range, normalize_wrist_50perc, 'm', x_range, normalize_wrist_75perc, 'y', x_range, normalize_wrist_90perc, 'k')





plt.figure(7)
x_range = np.arange(len(aggregated_wrist))
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Activity Intensity, Blue - Wrist Mean VM, Green - Wrist Mean SD')
plt.plot(x_range, normalize_activity_intensity, 'r', x_range, normalize_wrist_mvm, 'b', x_range, normalize_wrist_sd, 'g')

plt.figure(8)
x_range = np.arange(len(aggregated_wrist))
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Activity Intensity, Blue - Wrist Max VM, Green - Wrist Min SD')
plt.plot(x_range, normalize_activity_intensity, 'r', x_range, normalize_wrist_maxvm, 'b', x_range, normalize_wrist_minvm, 'g')

plt.figure(9)
x_range = np.arange(len(aggregated_wrist))
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Activity Intensity, Blue - 10th perc VM, Green - 25th perc, Magenta - 50th perc, Yellow - 75th perc, Black - 90th perc')
plt.plot(x_range, normalize_activity_intensity, 'r', x_range, normalize_wrist_10perc, 'b', x_range, normalize_wrist_25perc, 'g', x_range, normalize_wrist_50perc, 'm', x_range, normalize_wrist_75perc, 'y', x_range, normalize_wrist_90perc, 'k')

plt.show()
