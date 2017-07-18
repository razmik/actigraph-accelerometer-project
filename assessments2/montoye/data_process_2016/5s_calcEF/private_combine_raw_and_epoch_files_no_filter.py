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


def process_without_filter(starting_row, end_row, experiment, week, day, user, date, device):

    raw_data_filename = "D:/Accelerometer Data/"+experiment+"/"+week+"/"+day+"/"+user+" "+device+" "+date+"RAW.csv".replace('\\', '/')
    epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs/Epoch5/"+experiment+"/"+week+"/"+day+"/processed/"+user+"_Epoch5_"+experiment+"_"+week.replace(' ', '_')+"_"+date+".csv".replace('\\', '/')
    path_components = raw_data_filename.split('/')

    output_path = "D:/Accelerometer Data/Processed"
    output_path = output_path + '/' + path_components[2] + '/' + path_components[3] + '/' + path_components[
        4] + '/not_filtered/epoch_5/montoye_2016'
    filename_components = path_components[5].split(' ')

    n = 500
    epoch_start = int(starting_row / n)

    start = starting_row + 10
    start_time = time.time()
    print("Reading raw data file.")

    row_count = end_row - starting_row
    output_filename = output_path + '/' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv', '_') + 'row_' + str(int(starting_row / n)) + '_to_' + str(int(end_row / n)) + '.csv'
    print("Duration:", ((end_row - starting_row) / (100 * 3600)), "hours")
    raw_data = pd.read_csv(raw_data_filename, skiprows=start, nrows=row_count)

    raw_data.columns = ['X', 'Y', 'Z']

    """
    Calculate the statistical inputs (Features)
    """
    print("Calculating statistical parameters.")

    grouped_temp = raw_data.groupby(raw_data.index // n)
    aggregated = pd.DataFrame()

    aggregated['XMean'] = grouped_temp['X'].mean()
    aggregated['YMean'] = grouped_temp['Y'].mean()
    aggregated['ZMean'] = grouped_temp['Z'].mean()
    aggregated['XVar'] = grouped_temp['X'].var()
    aggregated['YVar'] = grouped_temp['Y'].var()
    aggregated['ZVar'] = grouped_temp['Z'].var()

    aggregated['X10perc'] = grouped_temp['X'].quantile(.1)
    aggregated['X25perc'] = grouped_temp['X'].quantile(.25)
    aggregated['X50perc'] = grouped_temp['X'].quantile(.5)
    aggregated['X75perc'] = grouped_temp['X'].quantile(.75)
    aggregated['X90perc'] = grouped_temp['X'].quantile(.9)

    aggregated['Y10perc'] = grouped_temp['Y'].quantile(.1)
    aggregated['Y25perc'] = grouped_temp['Y'].quantile(.25)
    aggregated['Y50perc'] = grouped_temp['Y'].quantile(.5)
    aggregated['Y75perc'] = grouped_temp['Y'].quantile(.75)
    aggregated['Y90perc'] = grouped_temp['Y'].quantile(.9)

    aggregated['Z10perc'] = grouped_temp['Z'].quantile(.1)
    aggregated['Z25perc'] = grouped_temp['Z'].quantile(.25)
    aggregated['Z50perc'] = grouped_temp['Z'].quantile(.5)
    aggregated['Z75perc'] = grouped_temp['Z'].quantile(.75)
    aggregated['Z90perc'] = grouped_temp['Z'].quantile(.9)

    """
    Include the epoch counts for 15 seconds and CPM values in aggregated dataframe
    """
    print("Combining with ActiGraph processed epoch count data as target variables")
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated),
                             usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    epoch_data.columns = ['actilife_wrist_AxisX', 'actilife_wrist_AxisY', 'actilife_wrist_AxisZ', 'actilife_wrist_vm_5', 'actilife_wrist_vm_60',
                          'actilife_wrist_AxisX_waist_eq', 'actilife_wrist_AxisY_waist_eq', 'actilife_wrist_AxisZ_waist_eq',
                          'actilife_wrist_vm_waist_eq', 'actilife_wrist_cpm', 'actilife_wrist_vm_cpm', 'actilife_waist_vm_5',
                          'actilife_waist_vm_60', 'actilife_waist_vm_cpm', 'actilife_waist_cpm', 'actilife_waist_intensity',
                          'actilife_waist_ee', 'actilife_waist_AxisX', 'actilife_waist_AxisY', 'actilife_waist_AxisZ']

    epoch_data['raw_wrist_XMean'] = aggregated['XMean']
    epoch_data['raw_wrist_YMean'] = aggregated['YMean']
    epoch_data['raw_wrist_ZMean'] = aggregated['ZMean']

    epoch_data['raw_wrist_XVar'] = aggregated['XVar']
    epoch_data['raw_wrist_YVar'] = aggregated['YVar']
    epoch_data['raw_wrist_ZVar'] = aggregated['ZVar']

    epoch_data['raw_wrist_X10perc'] = aggregated['X10perc']
    epoch_data['raw_wrist_X25perc'] = aggregated['X25perc']
    epoch_data['raw_wrist_X50perc'] = aggregated['X50perc']
    epoch_data['raw_wrist_X75perc'] = aggregated['X75perc']
    epoch_data['raw_wrist_X90perc'] = aggregated['X90perc']

    epoch_data['raw_wrist_Y10perc'] = aggregated['Y10perc']
    epoch_data['raw_wrist_Y25perc'] = aggregated['Y25perc']
    epoch_data['raw_wrist_Y50perc'] = aggregated['Y50perc']
    epoch_data['raw_wrist_Y75perc'] = aggregated['Y75perc']
    epoch_data['raw_wrist_Y90perc'] = aggregated['Y90perc']

    epoch_data['raw_wrist_Z10perc'] = aggregated['Z10perc']
    epoch_data['raw_wrist_Z25perc'] = aggregated['Z25perc']
    epoch_data['raw_wrist_Z50perc'] = aggregated['Z50perc']
    epoch_data['raw_wrist_Z75perc'] = aggregated['Z75perc']
    epoch_data['raw_wrist_Z90perc'] = aggregated['Z90perc']

    # Save file
    epoch_data.to_csv(output_filename, sep=',')
    print("File saved as", output_filename)
    print("Total duration", str(round(time.time() - start_time, 2)), "seconds")
