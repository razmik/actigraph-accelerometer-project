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
from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score

print('Keras version ', keras.__version__)


def load_data(filenames):

    # Single row for each
    data_dict = {}
    # ccc = 1
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

        # ccc += 1
        #
        # if ccc > 20:
        #     break

    return data_dict


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

    return person_corr, float(r_2_error)


def evaluate_regression_modal(REG_MODEL_ROOT_FOLDER, REG_RESULTS_FOLDER, data_dict):

    # Select best model
    model_files = [join(REG_MODEL_ROOT_FOLDER, f) for f in listdir(REG_MODEL_ROOT_FOLDER) if
                   isfile(join(REG_MODEL_ROOT_FOLDER, f)) and '.h5' in f]
    model_b = load_model(model_files[0])

    out_col_names = ['participant', 'R2_Overall', 'R2_SB', 'R2_LPA', 'R2_MVPA', 'Pearson_Overall', 'Pearson_SB', 'Pearson_LPA', 'Pearson_MVPA']
    output_results = []
    for key, data in tqdm(data_dict.items(), desc='Predicting'):

        # Test data placeholder
        X_test, y_test, y_class = data['X_data'], data['Y_data_regress'], data['Y_data_classif']

        # Overall
        p_overall, r2_overall = calculate_error(X_test, y_test, model_b)

        # For each PA category
        X_test_sb, y_test_sb = [X_test[i] for i, c in enumerate(y_class) if c == 1], [y_test[i] for i, c in enumerate(y_class) if c == 1]
        if len(y_test_sb) > 0:
            p_sb, r2_sb = calculate_error(X_test_sb, y_test_sb, model_b)
        else:
            p_sb, r2_sb = 0, 0

        X_test_lpa, y_test_lpa = [X_test[i] for i, c in enumerate(y_class) if c == 2], [y_test[i] for i, c in enumerate(y_class) if c == 2]
        if len(y_test_lpa) > 0:
            p_lpa, r2_lpa = calculate_error(X_test_lpa, y_test_lpa, model_b)
        else:
            p_lpa, r2_lpa = 0, 0

        X_test_mvpa, y_test_mvpa = [X_test[i] for i, c in enumerate(y_class) if c > 3], [y_test[i] for i, c in enumerate(y_class) if c > 3]
        if len(y_test_mvpa) > 0:
            p_mvpa, r2_mvpa = calculate_error(X_test_mvpa, y_test_mvpa, model_b)
        else:
            p_mvpa, r2_mvpa = 0, 0

        result_row = [key, r2_overall, r2_sb, r2_lpa, r2_mvpa, p_overall, p_sb, p_lpa, p_mvpa]
        output_results.append(result_row)

    # test completed
    pd.DataFrame(output_results, columns=out_col_names).to_csv(join(REG_RESULTS_FOLDER, 'results_reg_updated.csv'), index=None)


def run(FOLDER_NAME, training_version, trial_id, group, data_root):

    model_folder_name = FOLDER_NAME.split('-')
    model_folder_name[-1] = str(int(int(model_folder_name[-1])/2))
    model_folder_name = '-'.join(model_folder_name)

    TEST_DATA_FOLDER = data_root + '/{}/{}/'.format(FOLDER_NAME, group)
    REGRESSION_MODEL_ROOT_FOLDER = '../output/regression/v{}/{}/model_out/'.format(training_version, model_folder_name)
    OUTPUT_FOLDER_ROOT = '../output/regression/v{}/{}/individual_results'.format(trial_id, FOLDER_NAME)

    # Create output folders
    for f in [OUTPUT_FOLDER_ROOT]:
        if not exists(f):
            makedirs(f)

    # Load data
    all_files_test = [join(TEST_DATA_FOLDER, f) for f in listdir(TEST_DATA_FOLDER) if isfile(join(TEST_DATA_FOLDER, f))]

    data_dictionary = load_data(all_files_test)

    print('Evaluating Classification')
    evaluate_regression_modal(REGRESSION_MODEL_ROOT_FOLDER, OUTPUT_FOLDER_ROOT, data_dictionary)

    print('Completed {}.'.format(FOLDER_NAME))


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/Week 2/test_data/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f))]

    trial_num = 8
    training_version = 3
    allowed_list = [3000, 6000]
    groups = ['test', 'train_test']

    for f, grp in itertools.product(all_files, groups):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\n\nProcessing {}'.format(f))
        run(f, training_version, trial_num, grp, temp_folder)

