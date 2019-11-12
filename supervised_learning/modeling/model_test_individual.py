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
from scipy.stats.stats import pearsonr
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score, confusion_matrix
import supervised_learning.modeling.statistical_extensions as SE

pd.options.display.float_format = '{:.1f}'.format
sns.set()
plt.style.use('ggplot')
print('Keras version ', keras.__version__)


def get_time_in(labels, time_epoch):

    unique, counts = np.unique(labels, return_counts=True)
    outcomes = {}
    for i, j in zip(unique, counts):
        outcomes[i] = j

    SB = outcomes[0] * (6000 / time_epoch) if 0 in outcomes else 0
    LPA = outcomes[1] * (6000 / time_epoch) if 1 in outcomes else 0
    MVPA = outcomes[2] * (6000 / time_epoch) if 2 in outcomes else 0

    return SB, LPA, MVPA


def load_data(filenames):

    # Single row for each
    data_dict = {}
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

    # Data relabeling from index 0 (use only 3 classes)
    for key in tqdm(data_dict.keys(), desc='Relabeling'):

        data_dict[key]['Y_data_classif'] = np.where(data_dict[key]['Y_data_classif'] == 1, 0, data_dict[key]['Y_data_classif'])
        data_dict[key]['Y_data_classif'] = np.where(data_dict[key]['Y_data_classif'] == 2, 1, data_dict[key]['Y_data_classif'])
        data_dict[key]['Y_data_classif'] = np.where(data_dict[key]['Y_data_classif'] == 3, 2, data_dict[key]['Y_data_classif'])
        data_dict[key]['Y_data_classif'] = np.where(data_dict[key]['Y_data_classif'] == 4, 2, data_dict[key]['Y_data_classif'])

    return data_dict


def evaluate_classification_modal(CLASSIF_MODEL_ROOT_FOLDER, CLASSIFICATION_RESULTS_FOLDER, data_dict, TIME_PERIODS):

    # Select best model
    model_files = [join(CLASSIF_MODEL_ROOT_FOLDER, f) for f in listdir(CLASSIF_MODEL_ROOT_FOLDER) if
                   isfile(join(CLASSIF_MODEL_ROOT_FOLDER, f)) and '.h5' in f]
    model_b = load_model(model_files[0])

    out_col_names = ['participant',
                     'accuracy', 'accuracy_ci', 'sensitivity SB', 'sensitivity LPA', 'sensitivity MVPA',
                     'sensitivity_ci_SB', 'sensitivity_ci_LPA', 'sensitivity_ci_MVPA', 'specificity_SB',
                     'specificity_LPA', 'specificity_MVPA', 'specificity_ci_SB', 'specificity_ci_LPA', 'specificity_ci_MVPA',
                     'actual_SB', 'predicted_SB', 'actual_LPA', 'predicted_LPA', 'actual_MVPA', 'predicted_MVPA']
    output_results = []
    num_classes = 3
    for key, data in tqdm(data_dict.items(), desc='Predicting'):

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

        # Evaluation matrices
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

        result_row = [key, stats['accuracy'], stats['accuracy_ci'], sensitivity_sb, sensitivity_lpa, sensitivity_mvpa,
                      sensitivity_ci_sb, sensitivity_ci_lpa, sensitivity_ci_mvpa, specificity_sb, specificity_lpa,
                      specificity_mvpa, specificity_ci_sb, specificity_ci_lpa, specificity_ci_mvpa, actual_SB,
                      predicted_SB, actual_LPA, predicted_LPA, actual_MVPA, predicted_MVPA]
        output_results.append(result_row)

    # test completed
    pd.DataFrame(output_results, columns=out_col_names).to_csv(join(CLASSIFICATION_RESULTS_FOLDER, 'results.csv'), index=None)


def run(FOLDER_NAME, training_version, trial_id, unique_participants=True):

    model_folder_name = FOLDER_NAME.split('-')
    model_folder_name[-1] = str(int(int(model_folder_name[-1])/2))
    model_folder_name = '-'.join(model_folder_name)

    DATA_ROOT = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/'
    TRAIN_TEST_SUBJECT_PICKLE = 'participant_split/train_test_split.pickle'
    CLASSIF_MODEL_ROOT_FOLDER = '../output/classification/v{}/{}/model_out/'.format(training_version, model_folder_name)
    TEST_DATA_FOLDER = DATA_ROOT + 'Week 2/test_data/{}/'.format(FOLDER_NAME)
    OUTPUT_FOLDER_ROOT = '../output/test_results_individual/v{}/{}'.format(trial_id, FOLDER_NAME)

    # Create output folders
    for f in [OUTPUT_FOLDER_ROOT]:
        if not exists(f):
            makedirs(f)

    # The number of steps within one time segment
    TIME_PERIODS = int(FOLDER_NAME.split('-')[1])

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
    evaluate_classification_modal(CLASSIF_MODEL_ROOT_FOLDER, OUTPUT_FOLDER_ROOT, data_dictionary, TIME_PERIODS)

    print('Completed {}.'.format(FOLDER_NAME))


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/Week 2/test_data/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f))]

    trial_num = 5
    training_version = 3
    allowed_list = [3000, 6000]

    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\n\nProcessing {}'.format(f))
        run(f, training_version, trial_num, unique_participants=False)

