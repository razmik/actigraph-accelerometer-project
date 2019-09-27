from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time

process_start_time = time.time()

time_epoch = 10
multiplication_factor = int(60 / time_epoch)

experiment = 'LSM2'
week = 'Week 1'
day = 'Wednesday'

hip_key = 'Waist'
epoch_start_rows = 11

input_path = 'data/LSM219 Waist (2016-11-02)10sec.csv'
output_path = 'data/LSM219 Waist (2016-11-02)10sec_processed.csv'


"""
Calculate Waist (hip) epoch values and reference parameters
"""
hip_epoch_data = pd.read_csv(input_path, skiprows=epoch_start_rows, usecols=[0, 1, 2], header=None)
# Axis 1 (y) - Goes through head and middle of feet
# Axis 2 (x) - Goes through 2 hips
# Axis 3 (z) - Goes through front and back of the stomach
hip_epoch_data.columns = ['AxisY', 'AxisX', 'AxisZ']

hip_epoch_data['waist_count_p10'] = hip_epoch_data.AxisY

# save output file
hip_epoch_data.to_csv(output_path, sep=',', index=None)
