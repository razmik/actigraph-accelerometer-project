"""
1. Read wrist raw data file
2. Aggregate into time interval - epoch
3. Calculate time domain statistical parameters - eg. vm, max, min, percentiles, etc.
4. Calculate frequency domain statistical parameters
5. Combine statistical parameters with ActiGraph processed wrist-waist epoch counts
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import math, sys, time

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
end_row = 2880000
epoch_start = int(starting_row/n)

# Sample rate and desired cutoff frequencies (in Hz).
fs = 100.0
lowcut = 0.25   # 0.25
highcut = 2.5   # 2.5
nsamples = n
order = 4

start = starting_row + 10
start_time = time.time()
print("Reading raw data file.")
if end_row < 0:
    row_count = -1
    output_filename = output_path + '/' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv', '_') + 'row_' + str(int(starting_row/n)) + '_to_' + str(int(end_row/n)) + '.csv'
    print("Duration: 7 days")
    raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start)
else:
    row_count = end_row - starting_row
    output_filename = output_path + '/' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv', '_') + 'row_' + str(int(starting_row/n)) + '_to_' + str(int(end_row/n)) + '.csv'
    print("Duration:", ((end_row-starting_row)/(100*3600)), "hours")
    raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count)

reading_end_time = time.time()
print("Completed reading in", str(round(reading_end_time-start_time, 2)), "seconds")
raw_data_wrist.columns = ['X', 'Y', 'Z']

"""
Calculate the statistical inputs (Features)
"""
print("Calculating statistical parameters.")

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]

# Calculate the angle of arcsin from X and VM, arcsin(axis used/vector magnitude)/(pi/2)
raw_data_wrist['angle'] = (90 * np.arcsin(raw_data_wrist.X/raw_data_wrist['vm'])) / (math.pi/2)

aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm', 'mangle']

# def getSD(row):
#     return (row.vm - aggregated_wrist['mvm'][int(row.name / n)]) ** 2, (row.angle - aggregated_wrist['mangle'][int(row.name/n)]) ** 2
# raw_data_wrist['sd'], raw_data_wrist['sdangle'] = raw_data_wrist.apply(getSD, axis=1)

def getWristSD(row):
    return (row.vm - aggregated_wrist['mvm'][int(row.name/n)]) ** 2
raw_data_wrist['sd'] = raw_data_wrist.apply(getWristSD, axis=1)

def getSDAngle(row):
    return (row.angle - aggregated_wrist['mangle'][int(row.name/n)]) ** 2
raw_data_wrist['sdangle'] = raw_data_wrist.apply(getSDAngle, axis=1)

raw_data_wrist['enmo'] = raw_data_wrist['vm'] - 1
raw_data_wrist.loc[raw_data_wrist['enmo'] < 0, 'enmo'] = 0
raw_data_wrist.loc[raw_data_wrist['enmo'] >= 0, 'enmo'] = raw_data_wrist['enmo']

aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()

aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm', 'mangle', 'sdvm', 'sdangle', 'menmo']

del aggregated_wrist['X']
del aggregated_wrist['Y']
del aggregated_wrist['Z']

wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)

print("Calculating max, min, and percentiles.")

aggregated_wrist['maxvm'] = wrist_grouped_temp.max()['vm']
aggregated_wrist['minvm'] = wrist_grouped_temp.min()['vm']
aggregated_wrist['10perc'] = wrist_grouped_temp.quantile(.1)['vm']
aggregated_wrist['25perc'] = wrist_grouped_temp.quantile(.25)['vm']
aggregated_wrist['50perc'] = wrist_grouped_temp.quantile(.5)['vm']
aggregated_wrist['75perc'] = wrist_grouped_temp.quantile(.75)['vm']
aggregated_wrist['90perc'] = wrist_grouped_temp.quantile(.9)['vm']

aggregated_wrist['dom_freq'] = 0
aggregated_wrist['dom_2_freq'] = 0
aggregated_wrist['pow_dom_freq'] = 0
aggregated_wrist['pow_dom_2_freq'] = 0
aggregated_wrist['total_power'] = 0

cal_stats_time_end_time = time.time()
print("Calculating max, min, and percentiles duration", str(round(cal_stats_time_end_time-reading_end_time, 2)), "seconds")
# frequency domain features
print("Calculating frequency domain features.")

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

raw_data_wrist_groups = np.array_split(raw_data_wrist, len(aggregated_wrist))

for i in range(0, len(raw_data_wrist_groups)):

    VM = raw_data_wrist_groups[i]['vm']

    spectrum = np.fft.fft(VM)
    freqs = np.fft.fftfreq(VM.shape[-1], (1/fs))

    sp2 = np.sort(np.abs(spectrum), kind='mergesort')
    index_max2 = np.where(np.abs(spectrum) == sp2[-2])

    aggregated_wrist.set_value(i, 'dom_freq', freqs[np.argmax(np.abs(spectrum))])
    aggregated_wrist.set_value(i, 'dom_2_freq', float(freqs[index_max2[0]][0]))
    aggregated_wrist.set_value(i, 'pow_dom_freq', float(np.amax(np.abs(spectrum))))
    aggregated_wrist.set_value(i, 'pow_dom_2_freq', float(np.abs(sp2[-2])))
    aggregated_wrist.set_value(i, 'total_power', np.sum(spectrum))

    X_bp = butter_bandpass_filter(raw_data_wrist_groups[i]['X'], lowcut, highcut, fs, order)
    Y_bp = butter_bandpass_filter(raw_data_wrist_groups[i]['Y'], lowcut, highcut, fs, order)
    Z_bp = butter_bandpass_filter(raw_data_wrist_groups[i]['Z'], lowcut, highcut, fs, order)

    VM_bp = np.sqrt([(X_bp ** 2) + (Y_bp ** 2) + (Z_bp ** 2)])[0]
    aggregated_wrist.set_value(i, 'band_vm', np.abs(VM_bp).sum())


cal_stats_freq_end_time = time.time()
print("Calculating frequency domain features duration", str(round(cal_stats_freq_end_time - cal_stats_time_end_time, 2)), "seconds")

"""
Include the epoch counts for 15 seconds and CPM values in aggregated dataframe
"""
print("Combining with ActiGraph processed epoch count data as target variables")
epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_wrist), usecols=[1, 2, 3, 4, 5])
epoch_data.columns = ['wrist_vm_15', 'wrist_vm_60',	'waist_eq_wrist_vm_60',	'waist_vm_60', 'waist_intensity']

epoch_data['wrist_mvm'] = aggregated_wrist['mvm']
epoch_data['wrist_sdvm'] = aggregated_wrist['sdvm']
epoch_data['wrist_mangle'] = aggregated_wrist['mangle']
epoch_data['wrist_sdangle'] = aggregated_wrist['sdangle']
epoch_data['wrist_menmo'] = aggregated_wrist['menmo']
epoch_data['wrist_maxvm'] = aggregated_wrist['maxvm']
epoch_data['wrist_minvm'] = aggregated_wrist['minvm']
epoch_data['wrist_10perc'] = aggregated_wrist['10perc']
epoch_data['wrist_25perc'] = aggregated_wrist['25perc']
epoch_data['wrist_50perc'] = aggregated_wrist['50perc']
epoch_data['wrist_75perc'] = aggregated_wrist['75perc']
epoch_data['wrist_90perc'] = aggregated_wrist['90perc']
epoch_data['dom_freq'] = aggregated_wrist['dom_freq']
epoch_data['pow_dom_freq'] = aggregated_wrist['pow_dom_freq']
epoch_data['dom_2_freq'] = aggregated_wrist['dom_2_freq']
epoch_data['pow_dom_2_freq'] = aggregated_wrist['pow_dom_2_freq']
epoch_data['total_power'] = aggregated_wrist['total_power']
epoch_data['band_vm'] = aggregated_wrist['band_vm']

print("Combining duration", str(round(time.time()-cal_stats_freq_end_time, 2)), "seconds")

# Save file
epoch_data.to_csv(output_filename, sep=',')
print("File saved as", output_filename)
print("Total duration", str(round(time.time() - start_time, 2)), "seconds")
