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
    'E:\Data\Accelerometer_LR\staudenmayer\Epoch60'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch5'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch15'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch30'.replace('\\', '/'),
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

"""
If multiple files to be assessed.
"""


def refine_met_dict(dictionary):
    keys_remove = []
    for key, value in dictionary.items():
        if len(value) == 0:
            keys_remove.append(key)

    for remove_key in keys_remove:
        dictionary.pop(remove_key, 0)

    return dictionary


def initiate_dicts(folder_name):

    correlation_dict_MET[folder_name] = []
    correlation_dict_MET_sb[folder_name] = []
    correlation_dict_MET_lpa[folder_name] = []
    correlation_dict_MET_mvpa[folder_name] = []
    correlation_dict_Intensity[folder_name] = []
    correlation_dict_Intensity_sb[folder_name] = []
    correlation_dict_Intensity_lpa[folder_name] = []
    correlation_dict_Intensity_mvpa[folder_name] = []


def process_correlation(user_current, res, target_label, predicted_label, filename, correlation_dict):
    corr = round(stats.pearsonr(res[target_label], res[predicted_label])[0], 2)
    correlation_dict[filename].append(corr)

    if math.isnan(corr):
        print('\tIS NAN', filename)
        correlation_dict[filename].append(0)

    # print(user_current, corr)


def divide_results(dataframe):
    dataframe_sb = dataframe.loc[dataframe['waist_ee'] <= 1.5]
    dataframe_lpa = dataframe.loc[(1.5 < dataframe['waist_ee']) & (dataframe['waist_ee'] < 3)]
    dataframe_mvpa = dataframe.loc[3 <= dataframe['waist_ee']]
    return dataframe_sb, dataframe_lpa, dataframe_mvpa


correlation_dict_MET = {}
correlation_dict_MET_sb = {}
correlation_dict_MET_lpa = {}
correlation_dict_MET_mvpa = {}

correlation_dict_Intensity = {}
correlation_dict_Intensity_sb = {}
correlation_dict_Intensity_lpa = {}
correlation_dict_Intensity_mvpa = {}

for result_folder in result_folders:

    print('\nProcessing', result_folder)
    result_folder = result_folder + '/'
    initiate_dicts(result_folder)

    result_data_files = [f for f in listdir(result_folder) if isfile(join(result_folder, f))]

    prev_subj = ''
    iter_count = 0
    for file in result_data_files:

        dataframe = pd.read_csv(result_folder + file)
        user = file.split('_(2016-')[0]
        dataframe['subject'] = user

        if prev_subj != user:

            if prev_subj != '':

                results_sb, results_lpa, results_mvpa = divide_results(results)

                # Process data for the prev_subj
                if 'predicted_ee' in results.columns:
                    target_label, predicted_label = 'waist_ee', 'predicted_ee'
                    process_correlation(prev_subj, results, target_label, predicted_label, result_folder, correlation_dict_MET)
                    process_correlation(prev_subj, results_sb, target_label, predicted_label, result_folder, correlation_dict_MET_sb)
                    process_correlation(prev_subj, results_lpa, target_label, predicted_label, result_folder, correlation_dict_MET_lpa)
                    process_correlation(prev_subj, results_mvpa, target_label, predicted_label, result_folder, correlation_dict_MET_mvpa)
                else:
                    print('MET correlation not found in', file)

                target_label, predicted_label = 'actual_category', 'predicted_category'
                process_correlation(prev_subj, results, target_label, predicted_label, result_folder, correlation_dict_Intensity)

                # Cannot process because actual_category for sb is always - 1 , lpa - 2 and mvpa - 3. So variance between the array is always 0.
                # Therefore cannot measure the correlation.
                # https://stackoverflow.com/questions/7653993/encountered-invalid-value-when-i-use-pearsonr
                # process_correlation(prev_subj, results_sb, target_label, predicted_label, result_folder, correlation_dict_Intensity_sb)
                # process_correlation(prev_subj, results_lpa, target_label, predicted_label, result_folder, correlation_dict_Intensity_lpa)
                # process_correlation(prev_subj, results_mvpa, target_label, predicted_label, result_folder, correlation_dict_Intensity_mvpa)
                print(prev_subj)

            prev_subj = user
            results = dataframe
        elif iter_count == len(result_data_files)-1:

            results_sb, results_lpa, results_mvpa = divide_results(results)

            # Process data for the prev_subj in the last iteration
            if 'predicted_ee' in results.columns:
                target_label, predicted_label = 'waist_ee', 'predicted_ee'
                process_correlation(prev_subj, results, target_label, predicted_label, result_folder, correlation_dict_MET)
                process_correlation(prev_subj, results_sb, target_label, predicted_label, result_folder, correlation_dict_MET_sb)
                process_correlation(prev_subj, results_lpa, target_label, predicted_label, result_folder, correlation_dict_MET_lpa)
                process_correlation(prev_subj, results_mvpa, target_label, predicted_label, result_folder, correlation_dict_MET_mvpa)
            else:
                print('MET correlation not found in', file)

            target_label, predicted_label = 'actual_category', 'predicted_category'
            process_correlation(prev_subj, results, target_label, predicted_label, result_folder, correlation_dict_Intensity)
            # process_correlation(prev_subj, results_sb, target_label, predicted_label, result_folder, correlation_dict_Intensity_sb)
            # process_correlation(prev_subj, results_lpa, target_label, predicted_label, result_folder, correlation_dict_Intensity_lpa)
            # process_correlation(prev_subj, results_mvpa, target_label, predicted_label, result_folder, correlation_dict_Intensity_mvpa)
            print(prev_subj)
        else:
            results = results.append(dataframe, ignore_index=True)

        iter_count += 1

    total_completed += 1
    print('Completed\t', total_completed, '/', total_file_count)

with open('correlation_dict_MET.pickle', 'wb') as handle:
    pickle.dump(correlation_dict_MET, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('correlation_dict_MET_sb.pickle', 'wb') as handle:
    pickle.dump(correlation_dict_MET_sb, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('correlation_dict_MET_lpa.pickle', 'wb') as handle:
    pickle.dump(correlation_dict_MET_lpa, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('correlation_dict_MET_mvpa.pickle', 'wb') as handle:
    pickle.dump(correlation_dict_MET_mvpa, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('correlation_dict_Intensity.pickle', 'wb') as handle:
    pickle.dump(correlation_dict_Intensity, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('correlation_dict_Intensity_sb.pickle', 'wb') as handle:
    pickle.dump(correlation_dict_Intensity_sb, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('correlation_dict_Intensity_lpa.pickle', 'wb') as handle:
    pickle.dump(correlation_dict_Intensity_lpa, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('correlation_dict_Intensity_mvpa.pickle', 'wb') as handle:
    pickle.dump(correlation_dict_Intensity_mvpa, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Draw the CVS file
correlation_dict_MET = refine_met_dict(correlation_dict_MET)
with open("pearson_correlation_MET.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(correlation_dict_MET.keys())
   writer.writerows(zip(*correlation_dict_MET.values()))

correlation_dict_MET_sb = refine_met_dict(correlation_dict_MET_sb)
with open("pearson_correlation_MET_sb.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(correlation_dict_MET_sb.keys())
   writer.writerows(zip(*correlation_dict_MET_sb.values()))

correlation_dict_MET_lpa = refine_met_dict(correlation_dict_MET_lpa)
with open("pearson_correlation_MET_lpa.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(correlation_dict_MET_lpa.keys())
   writer.writerows(zip(*correlation_dict_MET_lpa.values()))

correlation_dict_MET_mvpa = refine_met_dict(correlation_dict_MET_mvpa)
with open("pearson_correlation_MET_mvpa.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(correlation_dict_MET_mvpa.keys())
   writer.writerows(zip(*correlation_dict_MET_mvpa.values()))


with open("pearson_correlation_Intensity.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(correlation_dict_Intensity.keys())
   writer.writerows(zip(*correlation_dict_Intensity.values()))

with open("pearson_correlation_Intensity_sb.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(correlation_dict_Intensity_sb.keys())
   writer.writerows(zip(*correlation_dict_Intensity_sb.values()))

with open("pearson_correlation_Intensity_lpa.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(correlation_dict_Intensity_lpa.keys())
   writer.writerows(zip(*correlation_dict_Intensity_lpa.values()))

with open("pearson_correlation_Intensity_mvpa.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(correlation_dict_Intensity_mvpa.keys())
   writer.writerows(zip(*correlation_dict_Intensity_mvpa.values()))