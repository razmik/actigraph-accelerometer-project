from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import listdir, makedirs
from os.path import join, isfile, exists
import numpy as np
import pandas as pd
from tqdm import tqdm
import keras
from keras.models import load_model
import itertools
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, confusion_matrix, mean_squared_error
import supervised_learning.modeling.statistical_extensions as SE


def load_data_overall(filenames, demo=False):

    X_data = []
    Y_data = []
    ID_user = []
    counter = 0
    for filename in tqdm(filenames):
        npy = np.load(filename, allow_pickle=True)
        X_data.append(npy.item().get('segments'))
        Y_data.append(npy.item().get('energy_e'))

        user_id = filename.split('/')[-1][:6]
        data_length = npy.item().get('energy_e').shape[0]
        ID_user.extend([user_id for _ in range(data_length)])

        if demo:
            counter += 1
            if counter > 10:
                break

    X_data = np.concatenate(X_data, axis=0)
    Y_data = np.concatenate(Y_data, axis=0)

    return X_data, Y_data, ID_user


def load_data_individual(filenames, demo=False):

    # Single row for each
    data_dict = {}
    ccc = 1
    for filename in tqdm(filenames, desc='Loading data'):

        npy = np.load(filename, allow_pickle=True)
        user_id = filename.split('/')[-1][:6]

        if user_id in data_dict.keys():
            data_dict[user_id]['X_data'] = np.concatenate([data_dict[user_id]['X_data'], npy.item().get('segments')], axis=0)
            data_dict[user_id]['Y_data_regress'] = np.concatenate([data_dict[user_id]['Y_data_regress'], npy.item().get('energy_e')], axis=0)
            data_dict[user_id]['Y_data_classif'] = np.concatenate([data_dict[user_id]['Y_data_classif'], npy.item().get('activity_classes')], axis=0)
        else:
            data_dict[user_id] = {}
            data_dict[user_id]['X_data'] = npy.item().get('segments')
            data_dict[user_id]['Y_data_regress'] = npy.item().get('energy_e')
            data_dict[user_id]['Y_data_classif'] = npy.item().get('activity_classes')

        if demo:
            ccc += 1
            if ccc > 20:
                break

    return data_dict


def get_time_in(labels, time_epoch):

    unique, counts = np.unique(labels, return_counts=True)
    outcomes = {}
    for i, j in zip(unique, counts):
        outcomes[i] = j

    SB = outcomes[0] / (6000 / time_epoch) if 0 in outcomes else 0
    LPA = outcomes[1] / (6000 / time_epoch) if 1 in outcomes else 0
    MVPA = outcomes[2] / (6000 / time_epoch) if 2 in outcomes else 0

    return SB, LPA, MVPA


def calculate_error(X, y, model):

    X, y = np.asarray(X), np.asarray(y)

    num_time_periods, num_sensors = X.shape[1], X.shape[2]
    input_shape = (num_time_periods * num_sensors)
    X = X.reshape(X.shape[0], input_shape)

    # Convert type for Keras otherwise Keras cannot process the data
    X = X.astype("float32")
    y = y.astype("float32")

    # Evaluate against test data
    y_pred_test = model.predict(X)

    y_pred_test_1d_list = [list(i)[0] for i in list(y_pred_test)]
    person_corr, pearson_pval = pearsonr(list(y), y_pred_test_1d_list)

    # rmse = np.sqrt(mean_squared_error(y, y_pred_test))
    r_2_error = r2_score(y, y_pred_test)

    return person_corr, float(r_2_error), y_pred_test_1d_list


def evaluate_regression_modal(model_b, REG_RESULTS_FOLDER, data_dict, TIME_PERIODS, group):

    out_col_names = ['participant', 'R2_Overall', 'R2_SB', 'R2_LPA', 'R2_MVPA', 'Pearson_Overall', 'Pearson_SB', 'Pearson_LPA', 'Pearson_MVPA',
                     'accuracy', 'accuracy_ci', 'accuracy_2class', 'accuracy_ci_2class',
                     'spearman_corr_overall', 'spearman_corr_overall_2cls', #'spearman_corr_sb', 'spearman_corr_lpa',
                     #'spearman_corr_sblpa', 'spearman_corr_mvpa',
                     'sensitivity SB', 'sensitivity LPA', 'sensitivity SBLPA', 'sensitivity MVPA', 'sensitivity MVPA2',
                     'sensitivity_ci_SB', 'sensitivity_ci_LPA', 'sensitivity_ci_SBLPA', 'sensitivity_ci_MVPA',
                     'sensitivity_ci_MVPA2',
                     'specificity_SB', 'specificity_LPA', 'specificity_SBLPA', 'specificity_MVPA', 'specificity_MVPA2',
                     'specificity_ci_SB', 'specificity_ci_LPA', 'specificity_ci_SBLPA', 'specificity_ci_MVPA',
                     'specificity_ci_MVPA2',
                     'actual_SB', 'predicted_SB', 'actual_LPA', 'predicted_LPA', 'actual_SBLPA', 'predicted_SBLPA',
                     'actual_MVPA', 'predicted_MVPA'
                     ]
    output_results = []
    for key, data in tqdm(data_dict.items(), desc='Predicting'):

        # Test data placeholder
        X_test, y_test, y_class = data['X_data'], data['Y_data_regress'], data['Y_data_classif']

        """Regression"""
        # Overall
        p_overall, r2_overall, y_pred_test = calculate_error(X_test, y_test, model_b)

        # For each PA category
        X_test_sb, y_test_sb = [X_test[i] for i, c in enumerate(y_class) if c == 1], [y_test[i] for i, c in enumerate(y_class) if c == 1]
        if len(y_test_sb) > 0:
            p_sb, r2_sb, _ = calculate_error(X_test_sb, y_test_sb, model_b)
        else:
            p_sb, r2_sb = 0, 0

        X_test_lpa, y_test_lpa = [X_test[i] for i, c in enumerate(y_class) if c == 2], [y_test[i] for i, c in enumerate(y_class) if c == 2]
        if len(y_test_lpa) > 0:
            p_lpa, r2_lpa, _ = calculate_error(X_test_lpa, y_test_lpa, model_b)
        else:
            p_lpa, r2_lpa = 0, 0

        X_test_mvpa, y_test_mvpa = [X_test[i] for i, c in enumerate(y_class) if c > 3], [y_test[i] for i, c in enumerate(y_class) if c > 3]
        if len(y_test_mvpa) > 0:
            p_mvpa, r2_mvpa, _ = calculate_error(X_test_mvpa, y_test_mvpa, model_b)
        else:
            p_mvpa, r2_mvpa = 0, 0

        """Classification"""
        y_pred_test = np.asarray(y_pred_test)
        max_y_test = SE.EnergyTransform.met_to_intensity(y_test)
        max_y_pred_test = SE.EnergyTransform.met_to_intensity(y_pred_test)

        """Evaluation matrices (3 CLASS)"""
        cnf_matrix = confusion_matrix(max_y_test, max_y_pred_test)
        stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)

        if len(stats['sensitivity']) == 3:
            sensitivity_sb, sensitivity_lpa, sensitivity_mvpa = stats['sensitivity']
        else:
            sensitivity_sb, sensitivity_lpa = stats['sensitivity']
            sensitivity_mvpa = 'NA'

        if len(stats['sensitivity_ci']) == 3:
            sensitivity_ci_sb, sensitivity_ci_lpa, sensitivity_ci_mvpa = stats['sensitivity_ci']
        else:
            sensitivity_ci_sb, sensitivity_ci_lpa = stats['sensitivity_ci']
            sensitivity_ci_mvpa = 'NA'

        if len(stats['specificity']) == 3:
            specificity_sb, specificity_lpa, specificity_mvpa = stats['specificity']
        else:
            specificity_sb, specificity_lpa = stats['specificity']
            specificity_mvpa = 'NA'

        if len(stats['specificity_ci']) == 3:
            specificity_ci_sb, specificity_ci_lpa, specificity_ci_mvpa = stats['specificity_ci']
        else:
            specificity_ci_sb, specificity_ci_lpa = stats['specificity_ci']
            specificity_ci_mvpa = 'NA'

        # Spearman's correlation
        spearman_corr, spearman_pval = spearmanr(max_y_test, max_y_pred_test)

        # max_y_test_sb, max_y_pred_test_sb = [max_y_test[i] for i, c in enumerate(max_y_test) if c == 0], [
        #     max_y_pred_test[i] for i, c in enumerate(max_y_test) if c == 0]
        # spearman_corr_sb, _ = spearmanr(max_y_test_sb, max_y_pred_test_sb)
        #
        # max_y_test_lpa, max_y_pred_test_lpa = [max_y_test[i] for i, c in enumerate(max_y_test) if c == 1], [
        #     max_y_pred_test[i] for i, c in enumerate(max_y_test) if c == 1]
        # spearman_corr_lpa, _ = spearmanr(max_y_test_lpa, max_y_pred_test_lpa)
        #
        # max_y_test_mvpa, max_y_pred_test_mvpa = [max_y_test[i] for i, c in enumerate(max_y_test) if c == 2], [
        #     max_y_pred_test[i] for i, c in enumerate(max_y_test) if c == 2]
        # spearman_corr_mvpa, _ = spearmanr(max_y_test_mvpa, max_y_pred_test_mvpa)

        """Evaluation matrices (2 CLASS)"""
        max_y_test_2cls = np.where(max_y_test == 1, 2, max_y_test)
        max_y_pred_test_2cls = np.where(max_y_pred_test == 1, 2, max_y_pred_test)

        cnf_matrix_2cls = confusion_matrix(max_y_test_2cls, max_y_pred_test_2cls)
        stats_2cls = SE.GeneralStats.evaluation_statistics(cnf_matrix_2cls)

        accuracy_2cls, accuracy_ci_2cls = stats_2cls['accuracy'], stats_2cls['accuracy_ci']
        sensitivity_sblpa, sensitivity_mvpa2 = stats_2cls['sensitivity']
        sensitivity_ci_sblpa, sensitivity_ci_mvpa2 = stats_2cls['sensitivity_ci']
        specificity_sblpa, specificity_mvpa2 = stats_2cls['specificity']
        specificity_ci_sblpa, specificity_ci_mvpa2 = stats_2cls['specificity_ci']

        # Spearman's correlation
        spearman_corr_2cls, _ = spearmanr(max_y_test_2cls, max_y_pred_test_2cls)

        # max_y_test_sblpa, max_y_pred_test_sblpa = [max_y_test_2cls[i] for i, c in enumerate(max_y_test_2cls) if
        #                                            c == 1], [
        #                                               max_y_pred_test_2cls[i] for i, c in enumerate(max_y_test_2cls) if
        #                                               c == 1]
        # spearman_corr_sblpa, _ = spearmanr(max_y_test_sblpa, max_y_pred_test_sblpa)
        """END 2 CLASS"""

        # Calculate Time in
        actual_SB, actual_LPA, actual_MVPA = get_time_in(y_class, TIME_PERIODS)
        predicted_SB, predicted_LPA, predicted_MVPA = get_time_in(max_y_pred_test, TIME_PERIODS)

        result_row = [key, r2_overall, r2_sb, r2_lpa, r2_mvpa, p_overall, p_sb, p_lpa, p_mvpa,
                      stats['accuracy'], stats['accuracy_ci'], accuracy_2cls, accuracy_ci_2cls,
                      spearman_corr, spearman_corr_2cls, #spearman_corr_sb, spearman_corr_lpa, spearman_corr_sblpa, spearman_corr_mvpa,
                      sensitivity_sb, sensitivity_lpa, sensitivity_sblpa, sensitivity_mvpa, sensitivity_mvpa2,
                      sensitivity_ci_sb, sensitivity_ci_lpa, sensitivity_ci_sblpa, sensitivity_ci_mvpa,
                      sensitivity_ci_mvpa2,
                      specificity_sb, specificity_lpa, specificity_sblpa, specificity_mvpa, specificity_mvpa2,
                      specificity_ci_sb, specificity_ci_lpa, specificity_ci_sblpa, specificity_ci_mvpa,
                      specificity_ci_mvpa2,
                      actual_SB, predicted_SB, actual_LPA, predicted_LPA, (actual_LPA + actual_LPA),
                      (predicted_SB + predicted_LPA), actual_MVPA, predicted_MVPA
                      ]
        output_results.append(result_row)

    # test completed
    pd.DataFrame(output_results, columns=out_col_names).to_csv(join(REG_RESULTS_FOLDER, 'results_reg_{}.csv'.format(group)), index=None)


def bland_altman_eval(model_b, REG_RESULTS_FOLDER, X_data, Y_data, ID_user, group):

    X, y = np.asarray(X_data), np.asarray(Y_data)

    num_time_periods, num_sensors = X.shape[1], X.shape[2]
    input_shape = (num_time_periods * num_sensors)
    X = X.reshape(X.shape[0], input_shape)

    # Convert type for Keras otherwise Keras cannot process the data
    X = X.astype("float32")
    y = y.astype("float32")

    # Evaluate against test data
    y_pred_test = model_b.predict(X)
    y_pred_test_1d_list = [list(i)[0] for i in list(y_pred_test)]

    # Overall analysis
    grp_results = []
    corr = pearsonr(list(y), y_pred_test_1d_list)
    grp_results.append('\n\n -------RESULTS-------\n\n')
    grp_results.append('Pearsons Correlation = {}'.format(corr))
    grp_results.append('RMSE - {}'.format(np.sqrt(mean_squared_error(y, y_pred_test))))
    grp_results.append('R2 Error - {}'.format(r2_score(y, y_pred_test)))

    result_string = '\n'.join(grp_results)
    with open(join(REG_RESULTS_FOLDER, 'overall_regress_report.txt'), "w") as text_file:
        text_file.write(result_string)

    # BA analysis
    results_df = pd.DataFrame(
        {'subject': ID_user,
         'waist_ee': list(Y_data),
         'predicted_ee': y_pred_test_1d_list
         })

    def clean_data_points(data):
        data = data.assign(waist_ee_cleaned=data['waist_ee'])
        data = data.assign(predicted_ee_cleaned=data['predicted_ee'])
        data.loc[(data['predicted_ee'] < 1), 'predicted_ee_cleaned'] = 1
        return data

    results_df = clean_data_points(results_df)

    ba_outfolder = join(REG_RESULTS_FOLDER, 'bland_altman', 'ba')
    if not exists(ba_outfolder):
        makedirs(ba_outfolder)

    SE.BlandAltman.bland_altman_paired_plot_tested(results_df, 'Bland Altman Analysis', 1, log_transformed=True,
                                                   min_count_regularise=False,
                                                   output_filename=ba_outfolder)


def run(FOLDER_NAME, training_version, group, data_root, demo=False):

    TEST_DATA_FOLDER = data_root + '/{}/{}/'.format(FOLDER_NAME, group)
    REGRESSION_MODEL_ROOT_FOLDER = 'E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}/regression/{}/model_out/'.format(training_version, FOLDER_NAME)
    OUTPUT_FOLDER_ROOT = 'E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}/regression/{}/individual_results/{}/'.format(training_version, FOLDER_NAME, group)
    OUTPUT_FOLDER_ROOT_OVERALL = 'E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}/regression/{}/results/{}/'.format(training_version, FOLDER_NAME, group)

    # Create output folders
    for f in [OUTPUT_FOLDER_ROOT, OUTPUT_FOLDER_ROOT_OVERALL]:
        if not exists(f):
            makedirs(f)

    # The number of steps within one time segment
    TIME_PERIODS = int(FOLDER_NAME.split('-')[1])

    # Select best model
    model_files = [join(REGRESSION_MODEL_ROOT_FOLDER, f) for f in listdir(REGRESSION_MODEL_ROOT_FOLDER) if
                   isfile(join(REGRESSION_MODEL_ROOT_FOLDER, f)) and '.h5' in f]
    model_b = load_model(model_files[0])

    all_files_test = [join(TEST_DATA_FOLDER, f) for f in listdir(TEST_DATA_FOLDER) if isfile(join(TEST_DATA_FOLDER, f))]

    # Load data for overall assessment
    test_X_data, test_Y_data, test_ID_user = load_data_overall(all_files_test, demo)
    bland_altman_eval(model_b, OUTPUT_FOLDER_ROOT_OVERALL, test_X_data, test_Y_data, test_ID_user, group)
    del test_X_data
    del test_Y_data
    del test_ID_user

    # Load data for individual assessment
    # data_dictionary = load_data_individual(all_files_test, demo=demo)
    # evaluate_regression_modal(model_b, OUTPUT_FOLDER_ROOT, data_dictionary, TIME_PERIODS, group)


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1_Combined\model_ready/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f)) and (f.split('-')[1] != f.split('-')[3])]

    training_version = '1-12_Dec'
    allowed_list = [3000, 6000]
    groups = ['test', 'train_test']

    for f, grp in itertools.product(all_files, groups):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\n\nProcessing {} for {}'.format(f, grp))
        run(f, training_version, grp, temp_folder, demo=True)

