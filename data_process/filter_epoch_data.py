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
import matplotlib.pyplot as plt
import math, sys, time

wrist_raw_data_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs\LSM2\Week 1\Wednesday\LSM204 Wrist (2016-11-02)15sec.csv".replace('\\', '/')
output_filename = 'D:\Accelerometer Data\Processed/filtered/LSM204_Wrist_(2016-11-02)_day_4_Saturday_Actilife_Processed_15s_epochs.csv'.replace('\\', '/')

# starting_row = 16560
# end_row = 18540

starting_row = 16560
end_row = 18440

start = starting_row + 10
row_count = end_row - starting_row

print("Reading epoch data file for", ((end_row-starting_row)/(4 * 60)), "hours")

raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count, usecols=[0, 1, 2])

raw_data_wrist.columns = ['X', 'Y', 'Z']

raw_data_wrist['vm'] = np.sqrt(raw_data_wrist.X ** 2 + raw_data_wrist.Y ** 2 + raw_data_wrist.Z ** 2)
raw_data_wrist['cpm'] = raw_data_wrist['vm'] * 4

plt.plot(np.arange(len(raw_data_wrist)), raw_data_wrist['vm'])
plt.show()

# Save file
raw_data_wrist.to_csv(output_filename, sep=',')
print("File saved as", output_filename)
