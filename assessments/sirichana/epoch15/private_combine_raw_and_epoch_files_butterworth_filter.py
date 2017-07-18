import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import math, sys, time


def process_with_filter(starting_row, end_row, experiment, week, day, user, date, device='Wrist'):
    wrist_raw_data_filename = ("D:/Accelerometer Data/" + experiment + "/" + week + "/" + day + "/" + user + " "
                               + device + " " + date + "RAW.csv").replace('\\', '/')
    epoch_filename = ("D:\Accelerometer Data\ActilifeProcessedEpochs/Epoch15/" + experiment + "/" + week + "/" + day
                      + "/processed/" + user + "_" + experiment + "_" + week.replace(' ', '_') + "_" + day + "_" + date + ".csv").replace('\\', '/')
    path_components = wrist_raw_data_filename.split('/')

    output_path = "D:/Accelerometer Data/Processed"
    output_path = output_path + '/' + path_components[2] + '/' + path_components[3] + '/' + path_components[
        4] + '/filtered/Epoch15'
    filename_components = path_components[5].split(' ')

    # epoch granularity
    n = 1500
    epoch_start = int(starting_row / n)

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 100.0
    lowcut = 0.25  # 0.25
    highcut = 2.5  # 2.5
    nsamples = n
    order = 4

    start = starting_row + 10
    start_time = time.time()
    print("Reading raw data file.")

    row_count = end_row - starting_row
    output_filename = output_path + '/' + filename_components[0] + '_' \
                      + filename_components[2].replace('RAW.csv', '_') + 'row_' + str(int(starting_row / n)) \
                      + '_to_' + str(int(end_row / n)) + '.csv'

    print("Duration:", ((end_row - starting_row) / (100 * 3600)), "hours")
    raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count)

    reading_end_time = time.time()
    print("Completed reading in", str(round(reading_end_time - start_time, 2)), "seconds")
    raw_data_wrist.columns = ['X', 'Y', 'Z']

    """
    Filter the raw X, Y, Z through 4th order Butterworth filter - 0.5Hz to 2.5Hz
    """

    def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    raw_data_wrist['X_filtered'] = butter_bandpass_filter(raw_data_wrist['X'], lowcut, highcut, fs, order)
    raw_data_wrist['Y_filtered'] = butter_bandpass_filter(raw_data_wrist['Y'], lowcut, highcut, fs, order)
    raw_data_wrist['Z_filtered'] = butter_bandpass_filter(raw_data_wrist['Z'], lowcut, highcut, fs, order)

    """
    Calculate the statistical inputs (Features)
    """
    print("Calculating statistical parameters.")

    # Calculate the vector magnitude from X, Y, Z raw readings
    raw_data_wrist['svm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[
                                0] - 1
    raw_data_wrist['svm_filtered'] = \
    np.sqrt([(raw_data_wrist.X_filtered ** 2) + (raw_data_wrist.Y_filtered ** 2) + (raw_data_wrist.Z_filtered ** 2)])[
        0] - 1

    wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)
    aggregated_wrist = pd.DataFrame()

    print("Calculating max, min, and percentiles.")

    aggregated_wrist['svm'] = wrist_grouped_temp['svm'].sum()
    aggregated_wrist['svm_filtered'] = wrist_grouped_temp['svm_filtered'].sum()

    cal_stats_time_end_time = time.time()
    print("Calculating max, min, and percentiles duration", str(round(cal_stats_time_end_time - reading_end_time, 2)),
          "seconds")

    """
    Include the epoch counts for 15 seconds and CPM values in aggregated dataframe
    """
    print("Combining with ActiGraph processed epoch count data as target variables")
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=len(aggregated_wrist), usecols=[16, 17])
    epoch_data.columns = ['actilife_waist_intensity', 'actilife_waist_ee']

    epoch_data['svm'] = aggregated_wrist['svm']
    epoch_data['svm_filtered'] = aggregated_wrist['svm_filtered']

    # Save file
    epoch_data.to_csv(output_filename, sep=',', index=False)
    print("File saved as", output_filename)
    print("Total duration", str(round(time.time() - start_time, 2)), "seconds")
