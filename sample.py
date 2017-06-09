"""
1. Read wrist raw data file
2. Aggregate into time interval - epoch
3. Calculate time domain statistical parameters - eg. vm, max, min, percentiles, etc.
4. Calculate frequency domain statistical parameters
5. Combine statistical parameters with ActiGraph processed wrist-waist epoch counts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import sys
import math

wrist_raw_data_filename = "D:/Accelerometer Data/Sample/LSM216/LSM216 Waist (2016-11-03)RAW.csv"
epoch_filename = "D:/Accelerometer Data/Sample/LSM216/LSM216 Waist (2016-11-03)15sec.csv"

# epoch granularity
n = 1500
starting_row = 0
end_row = 360000*10
epoch_start = int(starting_row/n) + 10

start = starting_row + 10
row_count = end_row - starting_row

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


aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm']
wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)

# print(raw_data_wrist)

raw_data_wrist_groups = np.array_split(raw_data_wrist, len(aggregated_wrist))

# print(raw_data_wrist_groups[0])
# frequency domain features

for i in range(0, len(raw_data_wrist_groups)):

    x = raw_data_wrist_groups[i]['vm']

    spectrum = np.fft.fft(x)
    freqs = np.fft.fftfreq(x.shape[-1], 0.01)

    sp2 = np.sort(np.abs(spectrum), kind='mergesort')
    index_max2 = np.where(np.abs(spectrum) == sp2[-2])
    index_min2 = np.where(np.abs(spectrum) == sp2[1])

    max_freq = freqs[np.argmax(np.abs(spectrum))]
    max_2_freq = float(freqs[index_max2[0]][0])
    min_freq = freqs[np.argmin(np.abs(spectrum))]
    min_2_freq = float(freqs[index_min2[0][0]])

    sum_of_spectrum = np.sum(spectrum)

    print("Iteration", i, "max", max_freq, max_2_freq, "min", min_freq, min_2_freq)

    # idx_1 = np.argmax(np.abs(spectrum))
    # freq = freqs[idx_1]

    # print("range", i, freq, np.amax(np.abs(spectrum)), idx_1)

    # plt.figure(1)
    # plt.plot(np.arange(len(raw_data_wrist_groups[i])), raw_data_wrist_groups[i]['vm'], 'r')
    #
    # plt.figure(2)
    # plt.plot(freqs, abs(spectrum))
    # plt.show()
