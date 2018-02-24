"""
1. Read device raw data file
2. Aggregate into time interval - epoch
3. Calculate time domain statistical parameters - eg. vm, max, min, percentiles, etc.
4. Calculate frequency domain statistical parameters
5. Combine statistical parameters with ActiGraph processed wrist-waist epoch counts
"""

import numpy as np
import pandas as pd
import math, sys, time


def process_without_filter(starting_row, end_row, experiment, week, day, user, date, epoch, epoch_duration, device):

    start_time = time.time()
    epoch_digit = epoch.split('Epoch')[1]

    device_raw_data_filename = "D:/Accelerometer Data/" + experiment + "/" + week + "/" + day + "/" + user + " " + device + " " + date + "RAW.csv".replace(
        '\\', '/')
    epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs/" + epoch + "/" + experiment + "/" + week + "/" + day + "/processed_1min_ref/" + user + "_" + experiment + "_" + week.replace(
        ' ', '_') + "_" + day + "_" + date + epoch_digit +"s.csv".replace('\\', '/')
    path_components = device_raw_data_filename.split('/')

    output_path = "E:/Data/Accelerometer_Processed_Raw_Epoch_Data_Unsupervised/"
    output_path = output_path + path_components[2] + '/' + path_components[3] + '/' + path_components[4] + "/" + epoch
    filename_components = path_components[5].split(' ')

    n = epoch_duration
    epoch_start = 1 + int(starting_row / n)

    leading_rows_to_skip = 11
    start = starting_row + leading_rows_to_skip
    row_count = end_row - starting_row

    output_filename = output_path + '/' + device + '_' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv', '_')\
                      + 'row_' + str(int(starting_row / n)) + '_to_' + str(int(end_row / n)) + '.csv'

    raw_data_device = pd.read_csv(device_raw_data_filename, skiprows=start, nrows=row_count, header=None)

    raw_data_device.columns = ['X', 'Y', 'Z']

    """
    Process RAW data as required by the model features
    """

    # VM and MANGLE needs for calculations by Staudenmayer model
    # Calculate the vector magnitude from X, Y, Z raw readings
    # Calculate the angle of arcsin from X and VM, arcsin(axis used/vector magnitude)/(pi/2)
    raw_data_device['vm'] = np.sqrt([(raw_data_device.X ** 2) + (raw_data_device.Y ** 2) + (raw_data_device.Z ** 2)])[0]
    raw_data_device['angle'] = (90 * np.arcsin(raw_data_device.X / raw_data_device['vm'])) / (math.pi / 2)

    # SVM needs for calculation in Sirichana model
    raw_data_device['svm'] = raw_data_device['vm'] - 1

    # X shifted needs to calculate covariance for Montoye 2017 model
    raw_data_device['X_shifted'] = raw_data_device['X'].shift(n)
    raw_data_device['Y_shifted'] = raw_data_device['Y'].shift(n)
    raw_data_device['Z_shifted'] = raw_data_device['Z'].shift(n)

    """
    Calculate the statistical inputs (Features)
    """

    device_grouped_temp = raw_data_device.groupby(raw_data_device.index // n)
    aggregated_device = pd.DataFrame()

    # Aggregated values required for Staudenmayer model
    aggregated_device['sdvm'] = device_grouped_temp['vm'].std()
    aggregated_device['mangle'] = device_grouped_temp['angle'].mean()
    aggregated_device['mangle'] = aggregated_device['mangle'].fillna(1)

    # Aggregated values required for Sirichana model
    aggregated_device['svm'] = device_grouped_temp['svm'].sum() * 0.4

    # Aggregated values required for Montoye Models
    aggregated_device['XMax'] = device_grouped_temp['X'].max()
    aggregated_device['YMax'] = device_grouped_temp['Y'].max()
    aggregated_device['ZMax'] = device_grouped_temp['Z'].max()

    aggregated_device['XMean'] = device_grouped_temp['X'].mean()
    aggregated_device['YMean'] = device_grouped_temp['Y'].mean()
    aggregated_device['ZMean'] = device_grouped_temp['Z'].mean()

    aggregated_device['XVar'] = device_grouped_temp['X'].var()
    aggregated_device['YVar'] = device_grouped_temp['Y'].var()
    aggregated_device['ZVar'] = device_grouped_temp['Z'].var()

    aggregated_device['X10perc'] = device_grouped_temp['X'].quantile(.1)
    aggregated_device['X25perc'] = device_grouped_temp['X'].quantile(.25)
    aggregated_device['X50perc'] = device_grouped_temp['X'].quantile(.5)
    aggregated_device['X75perc'] = device_grouped_temp['X'].quantile(.75)
    aggregated_device['X90perc'] = device_grouped_temp['X'].quantile(.9)

    aggregated_device['Y10perc'] = device_grouped_temp['Y'].quantile(.1)
    aggregated_device['Y25perc'] = device_grouped_temp['Y'].quantile(.25)
    aggregated_device['Y50perc'] = device_grouped_temp['Y'].quantile(.5)
    aggregated_device['Y75perc'] = device_grouped_temp['Y'].quantile(.75)
    aggregated_device['Y90perc'] = device_grouped_temp['Y'].quantile(.9)

    aggregated_device['Z10perc'] = device_grouped_temp['Z'].quantile(.1)
    aggregated_device['Z25perc'] = device_grouped_temp['Z'].quantile(.25)
    aggregated_device['Z50perc'] = device_grouped_temp['Z'].quantile(.5)
    aggregated_device['Z75perc'] = device_grouped_temp['Z'].quantile(.75)
    aggregated_device['Z90perc'] = device_grouped_temp['Z'].quantile(.9)

    aggregated_device['X_cov'] = device_grouped_temp.apply(lambda x: x['X'].cov(x['X_shifted']))
    aggregated_device['Y_cov'] = device_grouped_temp.apply(lambda x: x['Y'].cov(x['Y_shifted']))
    aggregated_device['Z_cov'] = device_grouped_temp.apply(lambda x: x['Z'].cov(x['Z_shifted']))

    aggregated_device['X_cov'] = aggregated_device['X_cov'].fillna(0)
    aggregated_device['Y_cov'] = aggregated_device['Y_cov'].fillna(0)
    aggregated_device['Z_cov'] = aggregated_device['Z_cov'].fillna(0)

    """
    Include the epoch counts for given epoch duration and CPM values in aggregated dataframe
    """
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_device), header=None)
    epoch_data.columns = ['AxisY', 'AxisX', 'AxisZ', 'waist_vm_60', 'waist_vm_cpm', 'waist_cpm',
                          'waist_ee', 'waist_intensity']

    del epoch_data['AxisY']
    del epoch_data['AxisX']
    del epoch_data['AxisZ']

    # Staudenmayer
    epoch_data['raw_' + device + '_sdvm'] = aggregated_device['sdvm']
    epoch_data['raw_' + device + '_mangle'] = aggregated_device['mangle']

    # Sirichana
    epoch_data['svm'] = aggregated_device['svm']

    # Montoye
    epoch_data['raw_' + device + '_XMax'] = aggregated_device['XMax']
    epoch_data['raw_' + device + '_YMax'] = aggregated_device['YMax']
    epoch_data['raw_' + device + '_ZMax'] = aggregated_device['ZMax']

    epoch_data['raw_' + device + '_XMean'] = aggregated_device['XMean']
    epoch_data['raw_' + device + '_YMean'] = aggregated_device['YMean']
    epoch_data['raw_' + device + '_ZMean'] = aggregated_device['ZMean']

    epoch_data['raw_' + device + '_XVar'] = aggregated_device['XVar']
    epoch_data['raw_' + device + '_YVar'] = aggregated_device['YVar']
    epoch_data['raw_' + device + '_ZVar'] = aggregated_device['ZVar']

    epoch_data['raw_' + device + '_X10perc'] = aggregated_device['X10perc']
    epoch_data['raw_' + device + '_X25perc'] = aggregated_device['X25perc']
    epoch_data['raw_' + device + '_X50perc'] = aggregated_device['X50perc']
    epoch_data['raw_' + device + '_X75perc'] = aggregated_device['X75perc']
    epoch_data['raw_' + device + '_X90perc'] = aggregated_device['X90perc']

    epoch_data['raw_' + device + '_Y10perc'] = aggregated_device['Y10perc']
    epoch_data['raw_' + device + '_Y25perc'] = aggregated_device['Y25perc']
    epoch_data['raw_' + device + '_Y50perc'] = aggregated_device['Y50perc']
    epoch_data['raw_' + device + '_Y75perc'] = aggregated_device['Y75perc']
    epoch_data['raw_' + device + '_Y90perc'] = aggregated_device['Y90perc']

    epoch_data['raw_' + device + '_Z10perc'] = aggregated_device['Z10perc']
    epoch_data['raw_' + device + '_Z25perc'] = aggregated_device['Z25perc']
    epoch_data['raw_' + device + '_Z50perc'] = aggregated_device['Z50perc']
    epoch_data['raw_' + device + '_Z75perc'] = aggregated_device['Z75perc']
    epoch_data['raw_' + device + '_Z90perc'] = aggregated_device['Z90perc']

    epoch_data['raw_' + device + '_X_cov'] = aggregated_device['X_cov']
    epoch_data['raw_' + device + '_Y_cov'] = aggregated_device['Y_cov']
    epoch_data['raw_' + device + '_Z_cov'] = aggregated_device['Z_cov']

    # Save file
    epoch_data.to_csv(output_filename, sep=',')
    print("File saved as", output_filename)
    print("Total duration", str(round(time.time() - start_time, 2)), "seconds")
