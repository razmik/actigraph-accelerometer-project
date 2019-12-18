"""
1. Read wrist raw data file
2. Aggregate into time interval - epoch
3. Calculate time domain statistical parameters - eg. vm, max, min, percentiles, etc.
4. Calculate frequency domain statistical parameters
5. Combine statistical parameters with ActiGraph processed wrist-waist epoch counts
"""

import numpy as np
import pandas as pd
import math, time
from os import makedirs
from os.path import exists


def process(input_data_folder, epoch_file_folder, output_folder, starting_row, end_row, experiment, week, day, user, date, time_epoch_dict, device):

    leading_rows_to_skip = 11
    start = starting_row + leading_rows_to_skip
    row_count = end_row - starting_row

    wrist_raw_data_filename = input_data_folder + experiment + "/" + week + "/" + day + "/" + user + " " + device + " " + date + "RAW.csv"

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

    # Calculate ENMO (Euclidean Norm Minus One) reguired for Hilderbrand model
    raw_data_wrist['enmo'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0] - 1
    raw_data_wrist.loc[raw_data_wrist.enmo < 0, 'enmo'] = 0

    """
    Calculate the statistical inputs (Features)
    """
    wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)
    aggregated_wrist = pd.DataFrame()

    # Aggregated values required for Hilderbrand model
    aggregated_wrist['enmo'] = wrist_grouped_temp['enmo'].mean()
    aggregated_wrist['enmo'] = aggregated_wrist['enmo'].fillna(1)

    """
    Include the epoch counts for given epoch duration and CPM values in aggregated dataframe
    """
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_wrist), header=None)
    epoch_data.columns = ['AxisY', 'AxisX', 'AxisZ', 'waist_vm_60', 'waist_vm_cpm', 'waist_cpm',
                          'waist_ee', 'waist_intensity']

    # remove unwanted columns
    del epoch_data['AxisY']
    del epoch_data['AxisX']
    del epoch_data['AxisZ']
    del epoch_data['waist_vm_60']
    del epoch_data['waist_vm_cpm']
    del epoch_data['waist_cpm']

    # Hilderbrand
    epoch_data['enmo'] = aggregated_wrist['enmo']

    # Save file
    epoch_data.to_csv(output_filename, sep=',', index=None)
    # print("File saved as", output_filename)
    # print("Total duration", str(round(time.time() - start_time, 2)), "seconds")
