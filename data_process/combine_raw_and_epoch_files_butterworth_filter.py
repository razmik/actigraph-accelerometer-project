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

wrist_raw_data_filename = "D:/Accelerometer Data/LSM2/Week 1/Wednesday/LSM204 Wrist (2016-11-02)RAW.csv"
wrist_raw_data_filename = wrist_raw_data_filename.replace('\\', '/')
epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs\LSM2\Week 1\Wednesday/processed/LSM204_LSM2_Week_1_Wednesday_(2016-11-02).csv"
epoch_filename = epoch_filename.replace('\\', '/')
path_components = wrist_raw_data_filename.split('/')

output_path = "D:/Accelerometer Data/Processed"
output_path = output_path + '/' + path_components[2] + '/' + path_components[3] + '/' + path_components[4] + '/filtered'
filename_components = path_components[5].split(' ')

# epoch granularity
n = 1500
starting_row = 24840010
end_row = 27660000
epoch_start = int(starting_row / n)

# Sample rate and desired cutoff frequencies (in Hz).
fs = 100.0
lowcut = 0.25  # 0.25
highcut = 2.5  # 2.5
nsamples = n
order = 4

start = starting_row + 10
start_time = time.time()
print("Reading raw data file.")
if end_row < 0:
    row_count = -1
    output_filename = output_path + '/' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv',
                                                                                                        '_') + 'row_' + str(
        int(starting_row / n)) + '_to_' + str(int(end_row / n)) + '.csv'
    print("Duration: 7 days")
    raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start)
else:
    row_count = end_row - starting_row
    output_filename = output_path + '/' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv',
                                                                                                        '_') + 'row_' + str(
        int(starting_row / n)) + '_to_' + str(int(end_row / n)) + '.csv'
    print("Duration:", ((end_row - starting_row) / (100 * 3600)), "hours")
    raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count)

reading_end_time = time.time()
print("Completed reading in", str(round(reading_end_time - start_time, 2)), "seconds")
raw_data_wrist.columns = ['X', 'Y', 'Z']

"""
Filter the raw X, Y, Z through 4th order Butterworth filter - 0.5Hz to 2.5Hz
"""


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


raw_data_wrist['X'] = butter_bandpass_filter(raw_data_wrist['X'], lowcut, highcut, fs, order)
raw_data_wrist['Y'] = butter_bandpass_filter(raw_data_wrist['Y'], lowcut, highcut, fs, order)
raw_data_wrist['Z'] = butter_bandpass_filter(raw_data_wrist['Z'], lowcut, highcut, fs, order)

"""
Calculate the statistical inputs (Features)
"""
print("Calculating statistical parameters.")

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]

# Calculate the angle of arcsin from X and VM, arcsin(axis used/vector magnitude)/(pi/2)
raw_data_wrist['angle'] = (90 * np.arcsin(raw_data_wrist.X / raw_data_wrist['vm'])) / (math.pi / 2)
raw_data_wrist['enmo'] = raw_data_wrist['vm'] - 1
raw_data_wrist.loc[raw_data_wrist['enmo'] < 0, 'enmo'] = 0

wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)
aggregated_wrist = pd.DataFrame()

print("Calculating max, min, and percentiles.")

aggregated_wrist['mvm'] = wrist_grouped_temp['vm'].mean()
aggregated_wrist['sdvm'] = wrist_grouped_temp['vm'].std()
aggregated_wrist['mangle'] = wrist_grouped_temp['angle'].mean()
aggregated_wrist['sdangle'] = wrist_grouped_temp['angle'].std()
aggregated_wrist['menmo'] = wrist_grouped_temp['enmo'].mean()
aggregated_wrist['maxvm'] = wrist_grouped_temp['vm'].max()
aggregated_wrist['minvm'] = wrist_grouped_temp['vm'].min()
aggregated_wrist['10perc'] = wrist_grouped_temp['vm'].quantile(.1)
aggregated_wrist['25perc'] = wrist_grouped_temp['vm'].quantile(.25)
aggregated_wrist['50perc'] = wrist_grouped_temp['vm'].quantile(.5)
aggregated_wrist['75perc'] = wrist_grouped_temp['vm'].quantile(.75)
aggregated_wrist['90perc'] = wrist_grouped_temp['vm'].quantile(.9)

cal_stats_time_end_time = time.time()
print("Calculating max, min, and percentiles duration", str(round(cal_stats_time_end_time - reading_end_time, 2)),
      "seconds")

# frequency domain features
print("Calculating frequency domain features.")

aggregated_wrist['dom_freq'] = 0
aggregated_wrist['dom_2_freq'] = 0
aggregated_wrist['pow_dom_freq'] = 0
aggregated_wrist['pow_dom_2_freq'] = 0
aggregated_wrist['total_power'] = 0

raw_data_wrist_groups = np.array_split(raw_data_wrist, len(aggregated_wrist))

for i in range(0, len(raw_data_wrist_groups)):
    VM = raw_data_wrist_groups[i]['vm']

    spectrum = np.fft.fft(VM)
    freqs = np.fft.fftfreq(VM.shape[-1], (1 / fs))

    sp2 = np.sort(np.abs(spectrum), kind='mergesort')
    index_max2 = np.where(np.abs(spectrum) == sp2[-2])

    pow_dom_freq = np.amax(spectrum)
    pow_dom_freq = np.sqrt(pow_dom_freq.real ** 2 + pow_dom_freq.imag ** 2)

    pow_dom_2_freq = sp2[-2]
    pow_dom_2_freq = np.sqrt(pow_dom_2_freq.real ** 2 + pow_dom_2_freq.imag ** 2)

    total_power = np.sum(spectrum)
    total_power = np.sqrt(total_power.real ** 2 + total_power.imag ** 2)

    aggregated_wrist.set_value(i, 'dom_freq', freqs[np.argmax(np.abs(spectrum))])
    aggregated_wrist.set_value(i, 'dom_2_freq', float(freqs[index_max2[0]][0]))
    aggregated_wrist.set_value(i, 'pow_dom_freq', pow_dom_freq)
    aggregated_wrist.set_value(i, 'pow_dom_2_freq', pow_dom_2_freq)
    aggregated_wrist.set_value(i, 'total_power', total_power)

cal_stats_freq_end_time = time.time()
print("Calculating frequency domain features duration",
      str(round(cal_stats_freq_end_time - cal_stats_time_end_time, 2)), "seconds")

"""
Include the epoch counts for 15 seconds and CPM values in aggregated dataframe
"""
print("Combining with ActiGraph processed epoch count data as target variables")
epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_wrist),
                         usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
epoch_data.columns = ['actilife_wrist_vm_15', 'actilife_wrist_vm_60', 'actilife_waist_eq_wrist_vm_60',
                      'actilife_wrist_cpm', 'actilife_wrist_vm_cpm', 'actilife_waist_vm_15', 'actilife_waist_vm_60',
                      'actilife_waist_vm_cpm', 'actilife_waist_cpm', 'actilife_waist_intensity', 'actilife_waist_ee']

epoch_data['raw_wrist_mvm'] = aggregated_wrist['mvm']
epoch_data['raw_wrist_sdvm'] = aggregated_wrist['sdvm']
epoch_data['raw_wrist_mangle'] = aggregated_wrist['mangle']
epoch_data['raw_wrist_sdangle'] = aggregated_wrist['sdangle']
epoch_data['raw_wrist_menmo'] = aggregated_wrist['menmo']
epoch_data['raw_wrist_maxvm'] = aggregated_wrist['maxvm']
epoch_data['raw_wrist_minvm'] = aggregated_wrist['minvm']
epoch_data['raw_wrist_10perc'] = aggregated_wrist['10perc']
epoch_data['raw_wrist_25perc'] = aggregated_wrist['25perc']
epoch_data['raw_wrist_50perc'] = aggregated_wrist['50perc']
epoch_data['raw_wrist_75perc'] = aggregated_wrist['75perc']
epoch_data['raw_wrist_90perc'] = aggregated_wrist['90perc']
epoch_data['raw_dom_freq'] = aggregated_wrist['dom_freq']
epoch_data['raw_pow_dom_freq'] = aggregated_wrist['pow_dom_freq']
epoch_data['raw_dom_2_freq'] = aggregated_wrist['dom_2_freq']
epoch_data['raw_pow_dom_2_freq'] = aggregated_wrist['pow_dom_2_freq']
epoch_data['raw_total_power'] = aggregated_wrist['total_power']

print("Combining duration", str(round(time.time() - cal_stats_freq_end_time, 2)), "seconds")

# Save file
epoch_data.to_csv(output_filename, sep=',')
print("File saved as", output_filename)
print("Total duration", str(round(time.time() - start_time, 2)), "seconds")
