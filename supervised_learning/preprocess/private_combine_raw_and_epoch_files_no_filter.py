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


def process_without_filter(starting_row, end_row, experiment, week, day, user, date, epoch, epoch_duration, device):

    start_time = time.time()
    epoch_digit = epoch.split('Epoch')[1]

    wrist_raw_data_filename = "D:/Accelerometer Data/" + experiment + "/" + week + "/" + day + "/" + user + " " + device + " " + date + "RAW.csv".replace(
        '\\', '/')
    epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs/" + epoch + "/" + experiment + "/" + week + "/" + day + "/processed_1min_ref/" + user + "_" + experiment + "_" + week.replace(
        ' ', '_') + "_" + day + "_" + date + epoch_digit +"s.csv".replace('\\', '/')
    path_components = wrist_raw_data_filename.split('/')

    output_path = "E:/Data/Accelerometer_Processed_Raw_Epoch_Data/"
    output_path = output_path + path_components[2] + '/' + path_components[3] + '/' + path_components[4] + "/" + epoch
    filename_components = path_components[5].split(' ')

    n = epoch_duration
    epoch_start = 1 + int(starting_row / n)

    leading_rows_to_skip = 11
    start = starting_row + leading_rows_to_skip
    row_count = end_row - starting_row

    output_filename = output_path + '/' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv',
                                                                                                        '_') + 'row_' + str(
        int(starting_row / n)) + '_to_' + str(int(end_row / n)) + '.csv'

    raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count, header=None)

    raw_data_wrist.columns = ['X', 'Y', 'Z']

    """
    Process RAW data as required by the model features
    """

    # VM and MANGLE needs for calculations by Staudenmayer model
    # Calculate the vector magnitude from X, Y, Z raw readings
    # Calculate the angle of arcsin from X and VM, arcsin(axis used/vector magnitude)/(pi/2)
    raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]
    raw_data_wrist['angle'] = (90 * np.arcsin(raw_data_wrist.X / raw_data_wrist['vm'])) / (math.pi / 2)

    # Calculate ENMO (Euclidean Norm Minus One) reguired for Hilderbrand model
    raw_data_wrist['enmo'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0] - 1
    raw_data_wrist.loc[raw_data_wrist.enmo < 0, 'enmo'] = 0

    # SVM needs for calculation in Sirichana model
    raw_data_wrist['svm'] = raw_data_wrist['vm'] - 1

    # X shifted needs to calculate covariance for Montoye 2017 model
    raw_data_wrist['X_shifted'] = raw_data_wrist['X'].shift(n)
    raw_data_wrist['Y_shifted'] = raw_data_wrist['Y'].shift(n)
    raw_data_wrist['Z_shifted'] = raw_data_wrist['Z'].shift(n)

    """
    Calculate the statistical inputs (Features)
    """

    wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)
    aggregated_wrist = pd.DataFrame()

    # Aggregated values required for Staudenmayer model
    aggregated_wrist['sdvm'] = wrist_grouped_temp['vm'].std()
    aggregated_wrist['mangle'] = wrist_grouped_temp['angle'].mean()
    aggregated_wrist['mangle'] = aggregated_wrist['mangle'].fillna(1)

    # Aggregated values required for Hilderbrand model
    aggregated_wrist['enmo'] = wrist_grouped_temp['enmo'].mean()
    aggregated_wrist['enmo'] = aggregated_wrist['enmo'].fillna(1)

    # Aggregated values required for Montoye Models
    aggregated_wrist['XMean'] = wrist_grouped_temp['X'].mean()
    aggregated_wrist['YMean'] = wrist_grouped_temp['Y'].mean()
    aggregated_wrist['ZMean'] = wrist_grouped_temp['Z'].mean()
    aggregated_wrist['XVar'] = wrist_grouped_temp['X'].var()
    aggregated_wrist['YVar'] = wrist_grouped_temp['Y'].var()
    aggregated_wrist['ZVar'] = wrist_grouped_temp['Z'].var()

    aggregated_wrist['X10perc'] = wrist_grouped_temp['X'].quantile(.1)
    aggregated_wrist['X25perc'] = wrist_grouped_temp['X'].quantile(.25)
    aggregated_wrist['X50perc'] = wrist_grouped_temp['X'].quantile(.5)
    aggregated_wrist['X75perc'] = wrist_grouped_temp['X'].quantile(.75)
    aggregated_wrist['X90perc'] = wrist_grouped_temp['X'].quantile(.9)

    aggregated_wrist['Y10perc'] = wrist_grouped_temp['Y'].quantile(.1)
    aggregated_wrist['Y25perc'] = wrist_grouped_temp['Y'].quantile(.25)
    aggregated_wrist['Y50perc'] = wrist_grouped_temp['Y'].quantile(.5)
    aggregated_wrist['Y75perc'] = wrist_grouped_temp['Y'].quantile(.75)
    aggregated_wrist['Y90perc'] = wrist_grouped_temp['Y'].quantile(.9)

    aggregated_wrist['Z10perc'] = wrist_grouped_temp['Z'].quantile(.1)
    aggregated_wrist['Z25perc'] = wrist_grouped_temp['Z'].quantile(.25)
    aggregated_wrist['Z50perc'] = wrist_grouped_temp['Z'].quantile(.5)
    aggregated_wrist['Z75perc'] = wrist_grouped_temp['Z'].quantile(.75)
    aggregated_wrist['Z90perc'] = wrist_grouped_temp['Z'].quantile(.9)

    aggregated_wrist['X_cov'] = wrist_grouped_temp.apply(lambda x: x['X'].cov(x['X_shifted']))
    aggregated_wrist['Y_cov'] = wrist_grouped_temp.apply(lambda x: x['Y'].cov(x['Y_shifted']))
    aggregated_wrist['Z_cov'] = wrist_grouped_temp.apply(lambda x: x['Z'].cov(x['Z_shifted']))

    aggregated_wrist['X_cov'] = aggregated_wrist['X_cov'].fillna(0)
    aggregated_wrist['Y_cov'] = aggregated_wrist['Y_cov'].fillna(0)
    aggregated_wrist['Z_cov'] = aggregated_wrist['Z_cov'].fillna(0)

    """
    Include the epoch counts for given epoch duration and CPM values in aggregated dataframe
    """
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_wrist), header=None)
    epoch_data.columns = ['AxisY', 'AxisX', 'AxisZ', 'waist_vm_60', 'waist_vm_cpm', 'waist_cpm',
                          'waist_ee', 'waist_intensity']
    # Staudenmayer
    epoch_data['raw_wrist_sdvm'] = aggregated_wrist['sdvm']
    epoch_data['raw_wrist_mangle'] = aggregated_wrist['mangle']

    # Hilderbrand
    epoch_data['enmo'] = aggregated_wrist['enmo']

    # Montoye
    epoch_data['raw_wrist_XMean'] = aggregated_wrist['XMean']
    epoch_data['raw_wrist_YMean'] = aggregated_wrist['YMean']
    epoch_data['raw_wrist_ZMean'] = aggregated_wrist['ZMean']

    epoch_data['raw_wrist_XVar'] = aggregated_wrist['XVar']
    epoch_data['raw_wrist_YVar'] = aggregated_wrist['YVar']
    epoch_data['raw_wrist_ZVar'] = aggregated_wrist['ZVar']

    epoch_data['raw_wrist_X10perc'] = aggregated_wrist['X10perc']
    epoch_data['raw_wrist_X25perc'] = aggregated_wrist['X25perc']
    epoch_data['raw_wrist_X50perc'] = aggregated_wrist['X50perc']
    epoch_data['raw_wrist_X75perc'] = aggregated_wrist['X75perc']
    epoch_data['raw_wrist_X90perc'] = aggregated_wrist['X90perc']

    epoch_data['raw_wrist_Y10perc'] = aggregated_wrist['Y10perc']
    epoch_data['raw_wrist_Y25perc'] = aggregated_wrist['Y25perc']
    epoch_data['raw_wrist_Y50perc'] = aggregated_wrist['Y50perc']
    epoch_data['raw_wrist_Y75perc'] = aggregated_wrist['Y75perc']
    epoch_data['raw_wrist_Y90perc'] = aggregated_wrist['Y90perc']

    epoch_data['raw_wrist_Z10perc'] = aggregated_wrist['Z10perc']
    epoch_data['raw_wrist_Z25perc'] = aggregated_wrist['Z25perc']
    epoch_data['raw_wrist_Z50perc'] = aggregated_wrist['Z50perc']
    epoch_data['raw_wrist_Z75perc'] = aggregated_wrist['Z75perc']
    epoch_data['raw_wrist_Z90perc'] = aggregated_wrist['Z90perc']

    epoch_data['raw_wrist_X_cov'] = aggregated_wrist['X_cov']
    epoch_data['raw_wrist_Y_cov'] = aggregated_wrist['Y_cov']
    epoch_data['raw_wrist_Z_cov'] = aggregated_wrist['Z_cov']

    # Save file
    epoch_data.to_csv(output_filename, sep=',')
    print("File saved as", output_filename)
    print("Total duration", str(round(time.time() - start_time, 2)), "seconds")
