import pandas as pd
import scipy.stats as stats
from os import listdir
from os.path import isfile, join
import csv
import math
import pickle

result_folders = [
    # 'E:\Data\Accelerometer_LR\staudenmayer\Epoch5'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch5'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/left_wrist/Epoch5/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/right_wrist/Epoch5/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch5/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch5/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch5/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch5/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\staudenmayer\Epoch15'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch15'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/left_wrist/Epoch15/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/right_wrist/Epoch15/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch15/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch15/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch15/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch15/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\staudenmayer\Epoch30'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch30'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/left_wrist/Epoch30/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/right_wrist/Epoch30/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch30/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch30/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch30/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch30/v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\staudenmayer\Epoch60'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_LR\sirichana\LRA\Epoch60'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/left_wrist/Epoch60/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2016/right_wrist/Epoch60/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch60/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch60/v1v2/combined'.replace('\\', '/'),
    # 'E:\Data\Accelerometer_Montoye_ANN/2017/left_wrist/Epoch60/v2/combined'.replace('\\', '/'),
    'E:\Data\Accelerometer_Montoye_ANN/2017/right_wrist/Epoch60/v2/combined'.replace('\\', '/')
]

total_file_count = len(result_folders)
total_completed = 0


def save_pickle(filename, data):
    with open('pickles/' + filename + '.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def refine_met_dict(dictionary):
    keys_remove = []
    for key, value in dictionary.items():
        if len(value) == 0:
            keys_remove.append(key)

    for remove_key in keys_remove:
        dictionary.pop(remove_key, 0)

    return dictionary


def wrist_to_csv(filename, data):
    data = refine_met_dict(data)
    with open(filename + '.csv', "w") as outfile:
        writer = csv.writer(outfile, lineterminator='\n')
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))


def clean_data_points(data):
    # Remove row if reference MET value is less than 1
    # data = data[data.waist_ee >= 1]

    # Shift predicted EE to 1, if it's less than 1
    data.loc[(data['predicted_ee'] < 1), 'predicted_ee'] = 1

    return data


def initiate_dicts(folder_name):
    correlation_dict_MET[folder_name] = []
    correlation_dict_MET_sb[folder_name] = []
    correlation_dict_MET_lpa[folder_name] = []
    correlation_dict_MET_mvpa[folder_name] = []
    correlation_dict_Intensity[folder_name] = []
    correlation_dict_Intensity_sb[folder_name] = []
    correlation_dict_Intensity_lpa[folder_name] = []
    correlation_dict_Intensity_mvpa[folder_name] = []

    pvalue_dict_MET[folder_name] = []
    pvalue_dict_MET_sb[folder_name] = []
    pvalue_dict_MET_lpa[folder_name] = []
    pvalue_dict_MET_mvpa[folder_name] = []
    pvalue_dict_Intensity[folder_name] = []
    pvalue_dict_Intensity_sb[folder_name] = []
    pvalue_dict_Intensity_lpa[folder_name] = []
    pvalue_dict_Intensity_mvpa[folder_name] = []


def find_correlation(user_current, res, target_label, predicted_label, filename, correlation_dict, p_val_dict,
                     method='pearson'):
    if method == 'pearson':
        corr, pval = stats.pearsonr(res[target_label], res[predicted_label])
    elif method == 'spearman':
        corr, pval = stats.spearmanr(res[target_label], res[predicted_label])
    corr = round(corr, 2)
    pval = round(pval, 2)
    correlation_dict[filename].append(corr)
    p_val_dict[filename].append(pval)

    if math.isnan(corr):
        print('\tIS NAN', filename)
        correlation_dict[filename].append(0)

        # print(user_current, corr)


def divide_results(dataframe):
    dataframe_sb = dataframe.loc[dataframe['waist_ee'] <= 1.5]
    dataframe_lpa = dataframe.loc[(1.5 < dataframe['waist_ee']) & (dataframe['waist_ee'] < 3)]
    dataframe_mvpa = dataframe.loc[3 <= dataframe['waist_ee']]
    return dataframe_sb, dataframe_lpa, dataframe_mvpa


def process_correlations(current_results):

    current_results = clean_data_points(current_results)

    results_sb, results_lpa, results_mvpa = divide_results(current_results)

    # Process data for the prev_subj
    if 'predicted_ee' in results.columns:
        target_label, predicted_label = 'waist_ee', 'predicted_ee'
        find_correlation(prev_subj, results, target_label, predicted_label, result_folder, correlation_dict_MET,
                         pvalue_dict_MET, method='pearson')
        find_correlation(prev_subj, results_sb, target_label, predicted_label, result_folder, correlation_dict_MET_sb,
                         pvalue_dict_MET_sb, method='pearson')
        find_correlation(prev_subj, results_lpa, target_label, predicted_label, result_folder, correlation_dict_MET_lpa,
                         pvalue_dict_MET_lpa, method='pearson')
        find_correlation(prev_subj, results_mvpa, target_label, predicted_label, result_folder,
                         correlation_dict_MET_mvpa, pvalue_dict_MET_mvpa, method='pearson')
    else:
        print('MET correlation not found in', prev_subj)

    target_label, predicted_label = 'actual_category', 'predicted_category'
    find_correlation(prev_subj, results, target_label, predicted_label, result_folder, correlation_dict_Intensity,
                     pvalue_dict_Intensity, method='spearman')
    # find_correlation(prev_subj, results_sb, target_label, predicted_label, result_folder, correlation_dict_Intensity_sb,
    #                  pvalue_dict_Intensity_sb, method='spearman')
    # find_correlation(prev_subj, results_lpa, target_label, predicted_label, result_folder,
    #                  correlation_dict_Intensity_lpa, pvalue_dict_Intensity_lpa, method='spearman')
    # find_correlation(prev_subj, results_mvpa, target_label, predicted_label, result_folder,
    #                  correlation_dict_Intensity_mvpa, pvalue_dict_Intensity_mvpa, method='spearman')

    # Cannot process pearson/spearman correlation because actual_category for sb is always - 1 , lpa - 2 and mvpa - 3. So variance between the array is always 0.
    # Therefore cannot measure the correlation.
    # https://stackoverflow.com/questions/7653993/encountered-invalid-value-when-i-use-pearsonr

    # print(prev_subj)


correlation_dict_MET = {}
correlation_dict_MET_sb = {}
correlation_dict_MET_lpa = {}
correlation_dict_MET_mvpa = {}
pvalue_dict_MET = {}
pvalue_dict_MET_sb = {}
pvalue_dict_MET_lpa = {}
pvalue_dict_MET_mvpa = {}

correlation_dict_Intensity = {}
correlation_dict_Intensity_sb = {}
correlation_dict_Intensity_lpa = {}
correlation_dict_Intensity_mvpa = {}
pvalue_dict_Intensity = {}
pvalue_dict_Intensity_sb = {}
pvalue_dict_Intensity_lpa = {}
pvalue_dict_Intensity_mvpa = {}

if __name__ == '__main__':

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
                    process_correlations(results)

                prev_subj = user
                results = dataframe
            elif iter_count == (len(result_data_files) - 1):
                process_correlations(results)
            else:
                results = results.append(dataframe, ignore_index=True)

            iter_count += 1

        total_completed += 1
        print('Completed\t', total_completed, '/', total_file_count)

    save_pickle('correlation_dict_MET', correlation_dict_MET)
    save_pickle('correlation_dict_MET_sb', correlation_dict_MET_sb)
    save_pickle('correlation_dict_MET_lpa', correlation_dict_MET_lpa)
    save_pickle('correlation_dict_MET_mvpa', correlation_dict_MET_mvpa)
    save_pickle('correlation_dict_Intensity', correlation_dict_Intensity)
    save_pickle('pvalue_dict_MET', pvalue_dict_MET)
    save_pickle('pvalue_dict_MET_sb', pvalue_dict_MET_sb)
    save_pickle('pvalue_dict_MET_lpa', pvalue_dict_MET_lpa)
    save_pickle('pvalue_dict_MET_mvpa', pvalue_dict_MET_mvpa)
    save_pickle('pvalue_dict_Intensity', pvalue_dict_Intensity)
    # save_pickle('pvalue_dict_Intensity_sb', pvalue_dict_Intensity_sb)
    # save_pickle('pvalue_dict_Intensity_lpa', pvalue_dict_Intensity_lpa)
    # save_pickle('pvalue_dict_Intensity_mvpa', pvalue_dict_Intensity_mvpa)

    wrist_to_csv('correlation_dict_MET', correlation_dict_MET)
    wrist_to_csv('correlation_dict_MET_sb', correlation_dict_MET_sb)
    wrist_to_csv('correlation_dict_MET_lpa', correlation_dict_MET_lpa)
    wrist_to_csv('correlation_dict_MET_mvpa', correlation_dict_MET_mvpa)
    wrist_to_csv('correlation_dict_Intensity', correlation_dict_Intensity)
    wrist_to_csv('pvalue_dict_MET', pvalue_dict_MET)
    wrist_to_csv('pvalue_dict_MET_sb', pvalue_dict_MET_sb)
    wrist_to_csv('pvalue_dict_MET_lpa', pvalue_dict_MET_lpa)
    wrist_to_csv('pvalue_dict_MET_mvpa', pvalue_dict_MET_mvpa)
    wrist_to_csv('pvalue_dict_Intensity', pvalue_dict_Intensity)
    # wrist_to_csv('pvalue_dict_Intensity_sb', pvalue_dict_Intensity_sb)
    # wrist_to_csv('pvalue_dict_Intensity_lpa', pvalue_dict_Intensity_lpa)
    # wrist_to_csv('pvalue_dict_Intensity_mvpa', pvalue_dict_Intensity_mvpa)
