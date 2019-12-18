"""
1. Read wrist raw data file
2. Aggregate into time interval - epoch
3. Calculate time domain statistical parameters - eg. vm, max, min, percentiles, etc.
4. Calculate frequency domain statistical parameters
5. Combine statistical parameters with ActiGraph processed wrist-waist epoch counts
"""

import numpy as np
import pandas as pd
import math
from os import makedirs
from os.path import exists


def process(input_data_folder, epoch_file_folder, output_folder, starting_row, end_row, experiment, week, day, user, date, time_epoch_dict, device):

    leading_rows_to_skip = 11
    start = starting_row + leading_rows_to_skip
    row_count = end_row - starting_row

    wrist_raw_data_filename = input_data_folder + experiment + "/" + week + "/" + day + "/" + user + " " + device + " " + date + "RAW.csv"

    # raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count, header=None)
    # raw_data_wrist.columns = ['X', 'Y', 'Z']
    # raw_data_wrist.reset_index(inplace=True)

    # for epoch, epoch_duration in time_epoch_dict.items():
    epoch, epoch_duration = list(time_epoch_dict.keys())[0], list(time_epoch_dict.values())[0]

    epoch_digit = epoch.split('Epoch')[1]
    epoch_filename = epoch_file_folder + epoch + "/" + experiment + "/" + week + "/" + day + "/processed_1min_ref/" + user + "_" + experiment + "_" + week.replace(
        ' ', '_') + "_" + day + "_" + date + epoch_digit +"s.csv"
    path_components = wrist_raw_data_filename.split('/')
    output_path = output_folder + "/" + epoch
    if not exists(output_path):
        makedirs(output_path)

    n = epoch_duration
    epoch_start = 1 + int(starting_row / n)

    # check for output folder
    out_folder_checker = '{}/{}/{}'.format(output_path, experiment, week)
    if not exists(out_folder_checker):
        makedirs(out_folder_checker)

    output_filename = '{}/{}/{}/{}_row-{}_to_{}.csv'.format(output_path, experiment, week, path_components[7].replace('RAW.csv', ''), int(starting_row / n), int(end_row / n))

    if exists(output_filename):
        return

    """
    Process RAW data as required by the model features
    """

    raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count, header=None)
    raw_data_wrist.columns = ['X', 'Y', 'Z']
    raw_data_wrist.reset_index(inplace=True)

    # VM and MANGLE needs for calculations by Staudenmayer model
    # Calculate the vector magnitude from X, Y, Z raw readings
    # Calculate the angle of arcsin from X and VM, arcsin(axis used/vector magnitude)/(pi/2)
    # raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]
    # raw_data_wrist['angle'] = (90 * np.arcsin(raw_data_wrist.X / raw_data_wrist['vm'])) / (math.pi / 2)

    # Calculate ENMO (Euclidean Norm Minus One) reguired for Hilderbrand model
    # raw_data_wrist['enmo'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0] - 1
    # raw_data_wrist.loc[raw_data_wrist.enmo < 0, 'enmo'] = 0

    # Load epoch data
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=int(raw_data_wrist.shape[0]/n), header=None)
    epoch_data.columns = ['AxisY', 'AxisX', 'AxisZ', 'waist_vm_60', 'waist_vm_cpm', 'waist_cpm',
                          'waist_ee', 'waist_intensity']

    # remove unwanted columns
    del epoch_data['AxisY']
    del epoch_data['AxisX']
    del epoch_data['AxisZ']
    del epoch_data['waist_vm_60']
    del epoch_data['waist_vm_cpm']
    del epoch_data['waist_cpm']

    # Combine epoch data file with raw file
    epoch_data_repeated = pd.concat([epoch_data] * n).sort_index().reset_index()
    del epoch_data_repeated['index']

    if epoch_data_repeated.shape[0] != raw_data_wrist.shape[0]:
        return

    # Merge the two dataframes
    merge_data = raw_data_wrist.merge(epoch_data_repeated, left_index=True, right_index=True)

    del merge_data['index']

    # Save file
    merge_data.to_csv(output_filename, sep=',', index=None)
