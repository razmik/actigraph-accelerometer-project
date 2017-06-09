import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

wrist_raw_data_filename = "D:/Accelerometer Data/LSM2/Week 1/Wednesday/LSM211 Waist (2016-11-02)RAW.csv"
wrist_raw_data_filename = wrist_raw_data_filename.replace('\\', '/')
epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs\LSM2\Week 1\Wednesday/processed/LSM211_LSM2_Week_1_Wednesday_(2016-11-02).csv"
epoch_filename = epoch_filename.replace('\\', '/')
path_components = wrist_raw_data_filename.split('/')

output_path = "D:/Accelerometer Data/Processed"
output_path = output_path + '/' + path_components[2] + '/' + path_components[3] + '/' + path_components[4]
filename_components = path_components[5].split(' ')

# epoch granularity
n = 1500
starting_row = 0
end_row = 360000*4
epoch_start = int(starting_row/n)

start = starting_row + 10
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

raw_data_wrist.columns = ['X', 'Y', 'Z']

"""
Calculate the statistical inputs (Features)
"""
print("Calculating statistical parameters.")

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]
aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm']

aggregated_wrist['band_vm'] = 0


raw_data_wrist['enmo'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0] - 1
raw_data_wrist.loc[raw_data_wrist['enmo'] < 0, 'enmo'] = 0
raw_data_wrist.loc[raw_data_wrist['enmo'] >= 0, 'enmo'] = raw_data_wrist['enmo']

print(raw_data_wrist)
sys.exit(0)

"""
Add a band pass filter to cutoff unusual vm
https://actigraph.desk.com/customer/en/portal/articles/2515508-actigraph-data-conversion-process
https://stackoverflow.com/questions/16301569/bandpass-filter-in-python
https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
https://dsp.stackexchange.com/questions/19002/butterworth-filter-in-python
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

# Sample rate and desired cutoff frequencies (in Hz).
fs = 100.0
lowcut = 0.25   #0.25
highcut = 2.5   #2.5
nsamples = n
order = 4

raw_data_wrist_groups = np.array_split(raw_data_wrist, len(aggregated_wrist))

for i in range(0, len(raw_data_wrist_groups)):

    X = raw_data_wrist_groups[i]['X']
    Y = raw_data_wrist_groups[i]['Y']
    Z = raw_data_wrist_groups[i]['Z']

    X_bp = butter_bandpass_filter(X, lowcut, highcut, fs, order)
    Y_bp = butter_bandpass_filter(Y, lowcut, highcut, fs, order)
    Z_bp = butter_bandpass_filter(Z, lowcut, highcut, fs, order)

    VM_bp = np.sqrt([(X_bp ** 2) + (Y_bp ** 2) + (Z_bp ** 2)])[0]
    aggregated_wrist.set_value(i, 'band_vm', np.abs(VM_bp).sum())


print("Combining with ActiGraph processed epoch count data as target variables")
epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_wrist), usecols=[1, 2, 3, 4, 5])
epoch_data.columns = ['wrist_vm_15', 'wrist_vm_60',	'waist_eq_wrist_vm_60',	'waist_vm_60', 'waist_intensity']

# print("Pearson 15 VM - Bandpass:", round(stats.pearsonr(epoch_data['wrist_vm_15'], aggregated_wrist['band_vm'])[0], 2))
print("Pearson 15 VM - Bandpass:", round(stats.pearsonr((epoch_data['waist_vm_60']/4), aggregated_wrist['band_vm'])[0], 2))

plt.title("Participant: "+path_components[2] + ' - ' + path_components[3] + ' - ' + path_components[4] + ' - ' + path_components[5])
# plt.plot(np.arange(len(aggregated_wrist)), epoch_data['wrist_vm_15'], 'r',
#          np.arange(len(aggregated_wrist)), aggregated_wrist['band_vm'], 'b')
# plt.plot(np.arange(len(aggregated_wrist)), (epoch_data['waist_vm_60']/4), 'r',
#          np.arange(len(aggregated_wrist)), aggregated_wrist['band_vm'], 'b')
# plt.show()
