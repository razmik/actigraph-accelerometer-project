import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join
import csv
import math
import pickle

result_folders = [
    # 'E:\Data\Accelerometer_LR\staudenmayer\Epoch5'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\staudenmayer\Epoch15'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\staudenmayer\Epoch30'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\staudenmayer\Epoch60'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch5'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch15'.replace('\\', '/'),
    'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch30'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch60'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRB\Epoch5'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRB\Epoch15'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRB\Epoch30'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRB\Epoch60'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/left_wrist/Epoch5/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/right_wrist/Epoch5/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/left_wrist/Epoch15/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/right_wrist/Epoch15/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/left_wrist/Epoch30/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/right_wrist/Epoch30/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/left_wrist/Epoch60/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/right_wrist/Epoch60/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch5/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch5/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch5/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch5/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch15/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch15/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch15/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch15/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch30/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch30/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch30/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch30/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch60/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch60/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch60/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch60/v2/combined'.replace('\\', '/')
]

total_file_count = len(result_folders)
total_completed = 0


def wrist_to_csv(filename, data):
    with open(filename + '.csv', "w") as outfile:
        writer = csv.writer(outfile, lineterminator='\n')
        for key, value in data.items():
            writer.writerow([key, value])


def divide_results(dataframe):
    dataframe_sb = dataframe.loc[dataframe['waist_ee'] <= 1.5]
    dataframe_lpa = dataframe.loc[(1.5 < dataframe['waist_ee']) & (dataframe['waist_ee'] < 3)]
    dataframe_mvpa = dataframe.loc[3 <= dataframe['waist_ee']]
    return dataframe_sb, dataframe_lpa, dataframe_mvpa


def process_overall_activity_distribution(user, dataframe):
    dataframe_sb, dataframe_lpa, dataframe_mvpa = divide_results(dataframe)
    activity_distribution_sb[user] = len(dataframe_sb)
    activity_distribution_lpa[user] = len(dataframe_lpa)
    activity_distribution_mvpa[user] = len(dataframe_mvpa)


activity_distribution_sb = {}
activity_distribution_lpa = {}
activity_distribution_mvpa = {}

if __name__ == '__main__':

    for result_folder in result_folders:

        print('\nProcessing', result_folder)

        result_folder = result_folder + '/'
        result_data_files = [f for f in listdir(result_folder) if isfile(join(result_folder, f))]

        prev_subj = ''
        iter_count = 0
        for file in result_data_files:

            dataframe = pd.read_csv(result_folder + file)
            user = file.split('_(2016-')[0]
            dataframe['subject'] = user

            if prev_subj != user:

                if prev_subj != '':
                    process_overall_activity_distribution(prev_subj, results)

                prev_subj = user
                results = dataframe
            elif iter_count == len(result_data_files) - 1:
                process_overall_activity_distribution(prev_subj, results)
            else:
                results = results.append(dataframe, ignore_index=True)

            iter_count += 1

        total_completed += 1
        print('Completed\t', total_completed, '/', total_file_count)

    wrist_to_csv('activity_distribution_sb', activity_distribution_sb)
    wrist_to_csv('activity_distribution_lpa', activity_distribution_lpa)
    wrist_to_csv('activity_distribution_mvpa', activity_distribution_mvpa)
