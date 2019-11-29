from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import listdir, makedirs
from os.path import join, isfile, exists
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import keras
from keras.models import load_model
from keras.utils import np_utils
from math import sqrt
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score, confusion_matrix
import supervised_learning.modeling.statistical_extensions as SE

pd.options.display.float_format = '{:.1f}'.format
sns.set()
plt.style.use('ggplot')
print('Keras version ', keras.__version__)


def load_data(filenames):

    # Single row for each
    data_dict = {}
    for filename in tqdm(filenames, desc='Loading data'):

        npy = np.load(filename, allow_pickle=True)
        user_id = filename.split('/')[-1][:6]

        if user_id in data_dict.keys():
            data_dict[user_id]['X_data'] = np.concatenate([data_dict[user_id]['X_data'], npy.item().get('segments')], axis=0)
            data_dict[user_id]['Y_data_regress'] = np.concatenate([data_dict[user_id]['Y_data_regress'], npy.item().get('energy_e')], axis=0)
        else:
            data_dict[user_id] = {}
            data_dict[user_id]['X_data'] = npy.item().get('segments')
            data_dict[user_id]['Y_data_regress'] = npy.item().get('energy_e')

    return data_dict


def evaluate_regression_modal(REG_MODEL_ROOT_FOLDER, REG_RESULTS_FOLDER, data_dict):

    # Select best model
    model_files = [join(REG_MODEL_ROOT_FOLDER, f) for f in listdir(REG_MODEL_ROOT_FOLDER) if
                   isfile(join(REG_MODEL_ROOT_FOLDER, f)) and '.h5' in f]
    model_b = load_model(model_files[0])

    out_col_names = ['participant', 'RMSE', 'R2_Error', 'Pearson_Corr', 'Pearson_PVal']
    output_results = []
    for key, data in tqdm(data_dict.items(), desc='Predicting'):

        # Test data placeholder
        X_test, y_test = data['X_data'], data['Y_data_regress']

        # Data -> Model ready
        num_time_periods, num_sensors = X_test.shape[1], X_test.shape[2]
        input_shape = (num_time_periods * num_sensors)
        X_test = X_test.reshape(X_test.shape[0], input_shape)

        # Convert type for Keras otherwise Keras cannot process the data
        X_test = X_test.astype("float32")
        y_test = y_test.astype("float32")

        # Evaluate against test data
        y_pred_test = model_b.predict(X_test)

        y_pred_test_1d_list = [list(i)[0] for i in list(y_pred_test)]
        person_corr, pearson_pval = pearsonr(list(y_test), y_pred_test_1d_list)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r_2_error = r2_score(y_test, y_pred_test)

        result_row = [key, rmse, r_2_error, person_corr, pearson_pval]
        output_results.append(result_row)

    # test completed
    pd.DataFrame(output_results, columns=out_col_names).to_csv(join(REG_RESULTS_FOLDER, 'results_reg.csv'), index=None)


def run(FOLDER_NAME, training_version, trial_id, unique_participants=True):

    model_folder_name = FOLDER_NAME.split('-')
    model_folder_name[-1] = str(int(int(model_folder_name[-1])/2))
    model_folder_name = '-'.join(model_folder_name)

    DATA_ROOT = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/'
    TRAIN_TEST_SUBJECT_PICKLE = 'participant_split/train_test_split.pickle'
    REGRESSION_MODEL_ROOT_FOLDER = '../output/regression/v{}/{}/model_out/'.format(training_version, model_folder_name)
    TEST_DATA_FOLDER = DATA_ROOT + 'Week 2/test_data/{}/'.format(FOLDER_NAME)
    OUTPUT_FOLDER_ROOT = '../output/test_results_individual/v{}/{}'.format(trial_id, FOLDER_NAME)

    # Create output folders
    for f in [OUTPUT_FOLDER_ROOT]:
        if not exists(f):
            makedirs(f)

    # Test Train Split
    if unique_participants:
        with open(TRAIN_TEST_SUBJECT_PICKLE, 'rb') as handle:
            split_dict = pickle.load(handle)
        test_subjects = split_dict['test']

        # Load all data
        all_files_test = [join(TEST_DATA_FOLDER, f) for f in listdir(TEST_DATA_FOLDER) if isfile(join(TEST_DATA_FOLDER, f))
                          and f.split(' ')[0] in test_subjects]
    else:

        # Load all data
        all_files_test = [join(TEST_DATA_FOLDER, f) for f in listdir(TEST_DATA_FOLDER) if isfile(join(TEST_DATA_FOLDER, f))]

    data_dictionary = load_data(all_files_test)

    print('Evaluating Classification')
    evaluate_regression_modal(REGRESSION_MODEL_ROOT_FOLDER, OUTPUT_FOLDER_ROOT, data_dictionary)

    print('Completed {}.'.format(FOLDER_NAME))


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/Week 2/test_data/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f))]

    trial_num = 7
    training_version = 3
    allowed_list = [3000, 6000]

    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\n\nProcessing {}'.format(f))
        run(f, training_version, trial_num, unique_participants=True)

