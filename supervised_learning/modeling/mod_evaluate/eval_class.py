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
from keras.utils import np_utils
import itertools
from scipy.stats.stats import spearmanr
from sklearn.metrics import confusion_matrix
import supervised_learning.modeling.statistical_extensions as SE

print('Keras version ', keras.__version__)


def get_time_in(labels, time_epoch):

    unique, counts = np.unique(labels, return_counts=True)
    outcomes = {}
    for i, j in zip(unique, counts):
        outcomes[i] = j

    SB = outcomes[0] / (6000 / time_epoch) if 0 in outcomes else 0
    LPA = outcomes[1] / (6000 / time_epoch) if 1 in outcomes else 0
    MVPA = outcomes[2] / (6000 / time_epoch) if 2 in outcomes else 0

    return SB, LPA, MVPA


def load_data(filenames, demo=False):

    # Single row for each
    data_dict = {}
    cc = 1
    for filename in tqdm(filenames, desc='Loading data'):

        npy = np.load(filename, allow_pickle=True)
        user_id = filename.split('/')[-1][:6]

        if user_id in data_dict.keys():
            data_dict[user_id]['X_data'] = np.concatenate([data_dict[user_id]['X_data'], npy.item().get('segments')], axis=0)
            data_dict[user_id]['Y_data_classif'] = np.concatenate([data_dict[user_id]['Y_data_classif'], npy.item().get('activity_classes')], axis=0)
            data_dict[user_id]['Y_data_regress'] = np.concatenate([data_dict[user_id]['Y_data_regress'], npy.item().get('energy_e')], axis=0)
        else:
            data_dict[user_id] = {}
            data_dict[user_id]['X_data'] = npy.item().get('segments')
            data_dict[user_id]['Y_data_classif'] = npy.item().get('activity_classes')
            data_dict[user_id]['Y_data_regress'] = npy.item().get('energy_e')

        if demo:
            cc += 1
            if cc > 25:
                break

    # Data relabeling from index 0 (use only 3 classes)
    for key in tqdm(data_dict.keys(), desc='Relabeling'):

        data_dict[key]['Y_data_classif'] = np.where(data_dict[key]['Y_data_classif'] == 1, 0, data_dict[key]['Y_data_classif'])
        data_dict[key]['Y_data_classif'] = np.where(data_dict[key]['Y_data_classif'] == 2, 1, data_dict[key]['Y_data_classif'])
        data_dict[key]['Y_data_classif'] = np.where(data_dict[key]['Y_data_classif'] == 3, 2, data_dict[key]['Y_data_classif'])
        data_dict[key]['Y_data_classif'] = np.where(data_dict[key]['Y_data_classif'] == 4, 2, data_dict[key]['Y_data_classif'])

    return data_dict


def evaluate_classification_modal(CLASSIF_MODEL_ROOT_FOLDER, CLASSIFICATION_RESULTS_FOLDER, data_dict, TIME_PERIODS, group):

    # Select best model
    model_files = [join(CLASSIF_MODEL_ROOT_FOLDER, f) for f in listdir(CLASSIF_MODEL_ROOT_FOLDER) if
                   isfile(join(CLASSIF_MODEL_ROOT_FOLDER, f)) and '.h5' in f]
    model_b = load_model(model_files[0])

    out_col_names = ['participant',
                     'accuracy', 'accuracy_ci', 'accuracy_2class', 'accuracy_ci_2class',
                     'spearman_corr_overall', 'spearman_corr_overall_2cls', 'spearman_corr_sb', 'spearman_corr_lpa', 'spearman_corr_sblpa', 'spearman_corr_mvpa',
                     'sensitivity SB', 'sensitivity LPA', 'sensitivity SBLPA', 'sensitivity MVPA', 'sensitivity MVPA2',
                     'sensitivity_ci_SB', 'sensitivity_ci_LPA', 'sensitivity_ci_SBLPA', 'sensitivity_ci_MVPA', 'sensitivity_ci_MVPA2',
                     'specificity_SB', 'specificity_LPA', 'specificity_SBLPA', 'specificity_MVPA', 'specificity_MVPA2',
                     'specificity_ci_SB', 'specificity_ci_LPA', 'specificity_ci_SBLPA', 'specificity_ci_MVPA', 'specificity_ci_MVPA2',
                     'actual_SB', 'predicted_SB', 'actual_LPA', 'predicted_LPA', 'actual_SBLPA', 'predicted_SBLPA', 'actual_MVPA', 'predicted_MVPA']
    output_results = []
    num_classes = 3
    for key, data in tqdm(data_dict.items(), desc='Evaluating () - ():'.format(TIME_PERIODS, group)):

        # Test data placeholder
        X_test, y_test = data['X_data'], data['Y_data_classif']

        # Calculate actual PA
        actual_SB, actual_LPA, actual_MVPA = get_time_in(y_test, TIME_PERIODS)

        # Data -> Model ready
        num_time_periods, num_sensors = X_test.shape[1], X_test.shape[2]
        input_shape = (num_time_periods * num_sensors)
        X_test = X_test.reshape(X_test.shape[0], input_shape)

        # Convert type for Keras otherwise Keras cannot process the data
        X_test = X_test.astype("float32")
        y_test = y_test.astype("float32")

        # One-hot encoding of y_train labels (only execute once!)
        y_test = np_utils.to_categorical(y_test, num_classes)

        # Evaluate against test data
        y_pred_test = model_b.predict(X_test)

        # Take the class with the highest probability from the test predictions
        max_y_pred_test = np.argmax(y_pred_test, axis=1)
        max_y_test = np.argmax(y_test, axis=1)

        # Predicted values for time
        predicted_SB, predicted_LPA, predicted_MVPA = get_time_in(max_y_pred_test, TIME_PERIODS)

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

        max_y_test_sb, max_y_pred_test_sb = [max_y_test[i] for i, c in enumerate(max_y_test) if c == 0], [max_y_pred_test[i] for i, c in enumerate(max_y_test) if c == 0]
        spearman_corr_sb, _ = spearmanr(max_y_test_sb, max_y_pred_test_sb)

        max_y_test_lpa, max_y_pred_test_lpa = [max_y_test[i] for i, c in enumerate(max_y_test) if c == 1], [max_y_pred_test[i] for i, c in enumerate(max_y_test) if c == 1]
        spearman_corr_lpa, _ = spearmanr(max_y_test_lpa, max_y_pred_test_lpa)

        max_y_test_mvpa, max_y_pred_test_mvpa = [max_y_test[i] for i, c in enumerate(max_y_test) if c == 2], [max_y_pred_test[i] for i, c in enumerate(max_y_test) if c == 2]
        spearman_corr_mvpa, _ = spearmanr(max_y_test_mvpa, max_y_pred_test_mvpa)

        """Evaluation matrices (2 CLASS)"""
        max_y_test_2cls = np.where(max_y_test == 0, 1, max_y_test)
        max_y_pred_test_2cls = np.where(max_y_pred_test == 0, 1, max_y_test)

        cnf_matrix_2cls = confusion_matrix(max_y_test_2cls, max_y_pred_test_2cls)
        stats_2cls = SE.GeneralStats.evaluation_statistics(cnf_matrix_2cls)

        accuracy_2cls, accuracy_ci_2cls = stats_2cls['accuracy'], stats_2cls['accuracy_ci']
        sensitivity_sblpa, sensitivity_mvpa2 = stats_2cls['sensitivity']
        sensitivity_ci_sblpa, sensitivity_ci_mvpa2 = stats_2cls['sensitivity_ci']
        specificity_sblpa, specificity_mvpa2 = stats_2cls['specificity']
        specificity_ci_sblpa, specificity_ci_mvpa2 = stats_2cls['specificity_ci']

        # Spearman's correlation
        spearman_corr_2cls, _ = spearmanr(max_y_test_2cls, max_y_pred_test_2cls)

        max_y_test_sblpa, max_y_pred_test_sblpa = [max_y_test_2cls[i] for i, c in enumerate(max_y_test_2cls) if c == 0], [
            max_y_pred_test_2cls[i] for i, c in enumerate(max_y_test_2cls) if c == 1]
        spearman_corr_sblpa, _ = spearmanr(max_y_test_sblpa, max_y_pred_test_sblpa)
        """END 2 CLASS"""

        result_row = [key, stats['accuracy'], stats['accuracy_ci'], accuracy_2cls, accuracy_ci_2cls,
                      spearman_corr, spearman_corr_2cls, spearman_corr_sb, spearman_corr_lpa, spearman_corr_sblpa, spearman_corr_mvpa,
                      sensitivity_sb, sensitivity_lpa, sensitivity_sblpa, sensitivity_mvpa, sensitivity_mvpa2,
                      sensitivity_ci_sb, sensitivity_ci_lpa, sensitivity_ci_sblpa, sensitivity_ci_mvpa, sensitivity_ci_mvpa2,
                      specificity_sb, specificity_lpa, specificity_sblpa, specificity_mvpa, specificity_mvpa2,
                      specificity_ci_sb, specificity_ci_lpa, specificity_ci_sblpa, specificity_ci_mvpa, specificity_ci_mvpa2,
                      actual_SB, predicted_SB, actual_LPA, predicted_LPA, (actual_LPA+actual_LPA), (predicted_SB+predicted_LPA), actual_MVPA, predicted_MVPA]
        output_results.append(result_row)

    # test completed
    pd.DataFrame(output_results, columns=out_col_names).to_csv(join(CLASSIFICATION_RESULTS_FOLDER, 'results_classif_{}.csv'.format(group)), index=None)


def run(FOLDER_NAME, training_version, trial_id, group, data_root, demo=False):

    TEST_DATA_FOLDER = data_root + '/{}/{}/'.format(FOLDER_NAME, group)
    CLASSIF_MODEL_ROOT_FOLDER = '../output/v{}/classification/{}/model_out/'.format(training_version, FOLDER_NAME)
    OUTPUT_FOLDER_ROOT = '../output/v{}/classification/{}/individual_results/{}/'.format(trial_id, FOLDER_NAME, group)

    # Create output folders
    for f in [OUTPUT_FOLDER_ROOT]:
        if not exists(f):
            makedirs(f)

    # The number of steps within one time segment
    TIME_PERIODS = int(FOLDER_NAME.split('-')[1])

    # Load data
    all_files_test = [join(TEST_DATA_FOLDER, f) for f in listdir(TEST_DATA_FOLDER) if isfile(join(TEST_DATA_FOLDER, f))]

    print('Loading data...')
    data_dictionary = load_data(all_files_test, demo=demo)

    evaluate_classification_modal(CLASSIF_MODEL_ROOT_FOLDER, OUTPUT_FOLDER_ROOT, data_dictionary, TIME_PERIODS, group)


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1_Combined\model_ready/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f)) and (f.split('-')[1] != f.split('-')[3])]

    trial_num = 1
    training_version = 1
    allowed_list = [3000, 6000]
    groups = ['test', 'train_test']

    for f, grp in itertools.product(all_files, groups):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\n\nProcessing {} for {}'.format(f, grp))
        run(f, training_version, trial_num, grp, temp_folder, demo=False)

    print('Completed.')
