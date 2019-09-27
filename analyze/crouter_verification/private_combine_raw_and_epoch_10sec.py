"""
1. Read wrist raw data file
2. Aggregate into time interval - epoch
3. Calculate time domain statistical parameters - eg. vm, max, min, percentiles, etc.
4. Calculate frequency domain statistical parameters
5. Combine statistical parameters with ActiGraph processed wrist-waist epoch counts
"""

import numpy as np
import pandas as pd
from scipy.stats import variation
import math, sys, time


def process_without_filter(starting_row, end_row, experiment, week, day, user, date, epoch, epoch_duration, device):
    waist_raw_data_filename = "data/" + user + " " + device + " " + date + "RAW.csv".replace('\\', '/')
    epoch_filename = "data/LSM219 Waist (2016-11-02)10sec_processed.csv".replace('\\', '/')

    n = epoch_duration
    epoch_start = 1 + int(starting_row / n)

    leading_rows_to_skip = 11
    start = starting_row + leading_rows_to_skip
    row_count = end_row - starting_row

    output_filename = "data/ref/" + '/__LSM219 Waist (2016-11-02)10sec_combined' + 'row_' + str(
        int(starting_row / n)) + '_to_' + str(int(end_row / n)) + '.csv'

    raw_data_waist = pd.read_csv(waist_raw_data_filename, skiprows=start, nrows=row_count, header=None)

    raw_data_waist.columns = ['X', 'Y', 'Z']

    """
    Calculate the statistical inputs (Features)
    """
    raw_data_waist = raw_data_waist.loc[:, ['Y']]
    waist_grouped_temp = raw_data_waist.groupby(raw_data_waist.index // n)

    # Calculate coefficient of variation
    """
    http://www.statisticshowto.com/probability-and-statistics/how-to-find-a-coefficient-of-variation/
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.variation.html
    """

    def coeff_of_var(row):
        y_cv = (row['Y_std'] / row['Y_mean']) * 100.0
        if not (row['Y_mean'] > 0):
            y_cv = 0
        return y_cv

    aggregated_wrist = waist_grouped_temp.agg([np.mean, np.std])
    aggregated_wrist.columns = ['_'.join(tup).rstrip('_') for tup in aggregated_wrist.columns.values]

    aggregated_wrist['Y_CV'] = aggregated_wrist.apply(coeff_of_var, axis=1)

    """
    Include the epoch counts for given epoch duration and CPM values in aggregated dataframe
    """
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_wrist), header=None)
    epoch_data.columns = ['AxisY', 'AxisX', 'AxisZ', 'waist_count_p10']

    epoch_data['Y_CV'] = aggregated_wrist['Y_CV']

    # Calculate crouter EE
    def calc_crouter_ee(row):
        cv = epoch_data['waist_count_p10'][row.name]
        waist_count_p10 = epoch_data['Y_CV'][row.name]

        if waist_count_p10 <= 8:
            met_value = 1
        elif cv <= 10:
            met_value = 2.294275 * (math.exp(0.00084679 * waist_count_p10))
        else:
            met_value = 0.749395 + (0.716431 * (math.log(waist_count_p10))) \
                        - (0.179874 * ((math.log(waist_count_p10)) ** 2)) \
                        + (0.033173 * ((math.log(waist_count_p10)) ** 3))

        if met_value > 10:
            print('HIGH MET', met_value)

        return met_value

    epoch_data['crouter_EE'] = epoch_data.apply(calc_crouter_ee, axis=1)

    # Combine 6 10 sec epochs to get average MET
    min_avg_energy_exp = pd.DataFrame()
    min_avg_grouped = epoch_data.groupby(epoch_data.index // 6)

    min_avg_energy_exp['crouter_ee'] = min_avg_grouped['crouter_EE'].mean()

    # Save file
    min_avg_energy_exp.to_csv(output_filename, sep=',', index=None)
