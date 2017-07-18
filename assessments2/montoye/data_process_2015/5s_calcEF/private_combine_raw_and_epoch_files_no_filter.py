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


def process_without_filter(starting_row, end_row, experiment, week, day, user, date, device='Waist'):

    waist_raw_data_filename = "D:/Accelerometer Data/"+experiment+"/"+week+"/"+day+"/"+user+" "+device+" "+date+"RAW.csv".replace('\\', '/')
    epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs/Epoch5/"+experiment+"/"+week+"/"+day+"/processed/"+user+"_Epoch5_"+experiment+"_"+week.replace(' ', '_')+"_"+date+".csv".replace('\\', '/')
    # epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs/Epoch5/"+experiment+"/"+week+"/"+day+"/processed/"+user+"_"+experiment+"_"+week.replace(' ', '_')+"_"+day+"_"+date+".csv".replace('\\', '/')
    path_components = waist_raw_data_filename.split('/')

    output_path = "D:/Accelerometer Data/Processed"
    output_path = output_path + '/' + path_components[2] + '/' + path_components[3] + '/' + path_components[
        4] + '/not_filtered/epoch_5/montoye_2015'
    filename_components = path_components[5].split(' ')

    n = 500
    epoch_start = int(starting_row / n)

    start = starting_row + 10
    start_time = time.time()
    print("Reading raw data file.")

    row_count = end_row - starting_row
    output_filename = output_path + '/' + filename_components[0] + '_' + filename_components[2].replace('RAW.csv', '_') + 'row_' + str(int(starting_row / n)) + '_to_' + str(int(end_row / n)) + '.csv'
    print("Duration:", ((end_row - starting_row) / (100 * 3600)), "hours")
    raw_data_waist = pd.read_csv(waist_raw_data_filename, skiprows=start, nrows=row_count)

    raw_data_waist.columns = ['X', 'Y', 'Z']

    """
    Calculate the statistical inputs (Features)
    """
    print("Calculating statistical parameters.")

    waist_grouped_temp = raw_data_waist.groupby(raw_data_waist.index // n)
    aggregated_waist = pd.DataFrame()

    aggregated_waist['XMean'] = waist_grouped_temp['X'].mean()
    aggregated_waist['YMean'] = waist_grouped_temp['Y'].mean()
    aggregated_waist['ZMean'] = waist_grouped_temp['Z'].mean()
    aggregated_waist['XVar'] = waist_grouped_temp['X'].var()
    aggregated_waist['YVar'] = waist_grouped_temp['Y'].var()
    aggregated_waist['ZVar'] = waist_grouped_temp['Z'].var()

    aggregated_waist['X10perc'] = waist_grouped_temp['X'].quantile(.1)
    aggregated_waist['X25perc'] = waist_grouped_temp['X'].quantile(.25)
    aggregated_waist['X50perc'] = waist_grouped_temp['X'].quantile(.5)
    aggregated_waist['X75perc'] = waist_grouped_temp['X'].quantile(.75)
    aggregated_waist['X90perc'] = waist_grouped_temp['X'].quantile(.9)

    aggregated_waist['Y10perc'] = waist_grouped_temp['Y'].quantile(.1)
    aggregated_waist['Y25perc'] = waist_grouped_temp['Y'].quantile(.25)
    aggregated_waist['Y50perc'] = waist_grouped_temp['Y'].quantile(.5)
    aggregated_waist['Y75perc'] = waist_grouped_temp['Y'].quantile(.75)
    aggregated_waist['Y90perc'] = waist_grouped_temp['Y'].quantile(.9)

    aggregated_waist['Z10perc'] = waist_grouped_temp['Z'].quantile(.1)
    aggregated_waist['Z25perc'] = waist_grouped_temp['Z'].quantile(.25)
    aggregated_waist['Z50perc'] = waist_grouped_temp['Z'].quantile(.5)
    aggregated_waist['Z75perc'] = waist_grouped_temp['Z'].quantile(.75)
    aggregated_waist['Z90perc'] = waist_grouped_temp['Z'].quantile(.9)

    """
    Include the epoch counts for 15 seconds and CPM values in aggregated dataframe
    """
    print("Combining with ActiGraph processed epoch count data as target variables")
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_waist),
                             usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    epoch_data.columns = ['actilife_wrist_AxisX', 'actilife_wrist_AxisY', 'actilife_wrist_AxisZ', 'actilife_wrist_vm_5', 'actilife_wrist_vm_60',
                          'actilife_wrist_AxisX_waist_eq', 'actilife_wrist_AxisY_waist_eq', 'actilife_wrist_AxisZ_waist_eq',
                          'actilife_wrist_vm_waist_eq', 'actilife_wrist_cpm', 'actilife_wrist_vm_cpm', 'actilife_waist_vm_5',
                          'actilife_waist_vm_60', 'actilife_waist_vm_cpm', 'actilife_waist_cpm', 'actilife_waist_intensity',
                          'actilife_waist_ee', 'actilife_waist_AxisX', 'actilife_waist_AxisY', 'actilife_waist_AxisZ']

    epoch_data['raw_waist_XMean'] = aggregated_waist['XMean']
    epoch_data['raw_waist_YMean'] = aggregated_waist['YMean']
    epoch_data['raw_waist_ZMean'] = aggregated_waist['ZMean']

    epoch_data['raw_waist_XVar'] = aggregated_waist['XVar']
    epoch_data['raw_waist_YVar'] = aggregated_waist['YVar']
    epoch_data['raw_waist_ZVar'] = aggregated_waist['ZVar']

    epoch_data['raw_waist_X10perc'] = aggregated_waist['X10perc']
    epoch_data['raw_waist_X25perc'] = aggregated_waist['X25perc']
    epoch_data['raw_waist_X50perc'] = aggregated_waist['X50perc']
    epoch_data['raw_waist_X75perc'] = aggregated_waist['X75perc']
    epoch_data['raw_waist_X90perc'] = aggregated_waist['X90perc']

    epoch_data['raw_waist_Y10perc'] = aggregated_waist['Y10perc']
    epoch_data['raw_waist_Y25perc'] = aggregated_waist['Y25perc']
    epoch_data['raw_waist_Y50perc'] = aggregated_waist['Y50perc']
    epoch_data['raw_waist_Y75perc'] = aggregated_waist['Y75perc']
    epoch_data['raw_waist_Y90perc'] = aggregated_waist['Y90perc']

    epoch_data['raw_waist_Z10perc'] = aggregated_waist['Z10perc']
    epoch_data['raw_waist_Z25perc'] = aggregated_waist['Z25perc']
    epoch_data['raw_waist_Z50perc'] = aggregated_waist['Z50perc']
    epoch_data['raw_waist_Z75perc'] = aggregated_waist['Z75perc']
    epoch_data['raw_waist_Z90perc'] = aggregated_waist['Z90perc']

    # Save file
    epoch_data.to_csv(output_filename, sep=',')
    print("File saved as", output_filename)
    print("Total duration", str(round(time.time() - start_time, 2)), "seconds")
