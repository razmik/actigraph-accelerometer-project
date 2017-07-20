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


def process_without_filter(starting_row, end_row, experiment, week, day, user, date, epoch, epoch_duration,
                           device='Wrist'):
    wrist_raw_data_filename = "D:/Accelerometer Data/" + experiment + "/" + week + "/" + day + "/" + user + " " + device + " " + date + "RAW.csv".replace(
        '\\', '/')
    epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs/" + epoch + "/" + experiment + "/" + week + "/" + day + "/processed_1min_ref/" + user + "_" + experiment + "_" + week.replace(
        ' ', '_') + "_" + day + "_" + date + "30s.csv".replace('\\', '/')
    path_components = wrist_raw_data_filename.split('/')

    output_path = "D:/Accelerometer Data/Assessment/staudenmayer/"
    output_path = output_path + path_components[2] + '/' + path_components[3] + '/' + path_components[4] + "/" + epoch
    filename_components = path_components[5].split(' ')

    n = epoch_duration
    epoch_start = int(starting_row / n)

    start = starting_row + 10
    start_time = time.time()
    print("Reading raw data file.")
    row_count = end_row - starting_row
    output_filename = output_path + '/' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv',
                                                                                                        '_') + 'row_' + str(
        int(starting_row / n)) + '_to_' + str(int(end_row / n)) + '.csv'
    print("Duration:", ((end_row - starting_row) / (100 * 3600)), "hours")
    raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count)

    raw_data_wrist.columns = ['X', 'Y', 'Z']

    """
    Calculate the statistical inputs (Features)
    """

    # Calculate the vector magnitude from X, Y, Z raw readings
    raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]

    # Calculate the angle of arcsin from X and VM, arcsin(axis used/vector magnitude)/(pi/2)
    raw_data_wrist['angle'] = (90 * np.arcsin(raw_data_wrist.X / raw_data_wrist['vm'])) / (math.pi / 2)

    wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)
    aggregated_wrist = pd.DataFrame()

    aggregated_wrist['sdvm'] = wrist_grouped_temp['vm'].std()
    aggregated_wrist['mangle'] = wrist_grouped_temp['angle'].mean()

    aggregated_wrist['mangle'] = aggregated_wrist['mangle'].fillna(1)

    """
    Include the epoch counts for 5 seconds and CPM values in aggregated dataframe
    """
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_wrist))
    epoch_data.columns = ['actilife_waist_AxisX', 'actilife_waist_AxisY', 'actilife_waist_AxisZ',
                          'actilife_waist_vm_60',
                          'actilife_waist_vm_cpm', 'actilife_waist_cpm', 'waist_ee', 'actilife_waist_intensity']

    epoch_data['raw_wrist_sdvm'] = aggregated_wrist['sdvm']
    epoch_data['raw_wrist_mangle'] = aggregated_wrist['mangle']

    # Save file
    epoch_data.to_csv(output_filename, sep=',')
    print("File saved as", output_filename)
    print("Total duration", str(round(time.time() - start_time, 2)), "seconds")
