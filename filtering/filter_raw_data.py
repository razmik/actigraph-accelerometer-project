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

wrist_raw_data_filename = "D:/Accelerometer Data/LSM2/Week 1/Wednesday/LSM204 Waist (2016-11-02)RAW.csv".replace('\\', '/')
output_filename = 'D:\Accelerometer Data\Processed/filtered/LSM204_Waist_(2016-11-02)_day_4_Saturday_RAW.csv'.replace('\\', '/')

starting_row = 24840010
end_row = 27660000

start = starting_row + 10
row_count = end_row - starting_row

print("Reading raw data file for", ((end_row-starting_row)/(100*3600)), "hours")

raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count)

raw_data_wrist.columns = ['X', 'Y', 'Z']

# Save file
raw_data_wrist.to_csv(output_filename, sep=',')
print("File saved as", output_filename)
