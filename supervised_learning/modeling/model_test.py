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


def load_data(filenames):

    X_data = []
    Y_data_classif = []
    Y_data_regress = []
    ID_user = []
    for filename in tqdm(filenames, desc='Loading data'):
        npy = np.load(filename, allow_pickle=True)
        X_data.append(npy.item().get('segments'))
        Y_data_classif.append(npy.item().get('activity_classes'))
        Y_data_regress.append(npy.item().get('energy_e'))

        user_id = filename.split('/')[-1][:6]
        data_length = npy.item().get('activity_classes').shape[0]
        ID_user.extend([user_id for _ in range(data_length)])

    X_data = np.concatenate(X_data, axis=0)
    Y_data_classif = np.concatenate(Y_data_classif, axis=0)
    Y_data_regress = np.concatenate(Y_data_regress, axis=0)

    # Data relabeling from index 0 (use only 3 classes)
    Y_data_classif = np.where(Y_data_classif == 1, 0, Y_data_classif)
    Y_data_classif = np.where(Y_data_classif == 2, 1, Y_data_classif)
    Y_data_classif = np.where(Y_data_classif == 3, 2, Y_data_classif)
    Y_data_classif = np.where(Y_data_classif == 4, 2, Y_data_classif)

    return X_data, Y_data_classif, Y_data_regress, ID_user


def evaluate_classification_modal(CLASSIF_MODEL_ROOT_FOLDER, CLASSIFICATION_RESULTS_FOLDER, test_X_data, test_Y_data, test_ID_user,
                                 TIME_PERIODS, STEP_DISTANCE):

    assert test_X_data.shape[0] == test_Y_data.shape[0] == len(test_ID_user)

    # Test data placeholder
    X_test, y_test = test_X_data, test_Y_data

    # Data -> Model ready
    print('Data -> Model ready.')
    num_time_periods, num_sensors = X_test.shape[1], X_test.shape[2]
    num_classes = len(np.unique(y_test))
    input_shape = (num_time_periods * num_sensors)
    X_test = X_test.reshape(X_test.shape[0], input_shape)

    # Convert type for Keras otherwise Keras cannot process the data
    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    # One-hot encoding of y_train labels (only execute once!)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # Select best model
    model_files = [join(CLASSIF_MODEL_ROOT_FOLDER, f) for f in listdir(CLASSIF_MODEL_ROOT_FOLDER) if
                   isfile(join(CLASSIF_MODEL_ROOT_FOLDER, f)) and '.h5' in f]
    model_b = load_model(model_files[0])

    # Evaluate against test data

    results_descriptions = []

    LABEL = 'activity_classes'
    results_descriptions.append(
        'Time Period = {}, Step Distance = {}, Label = {}'.format(TIME_PERIODS, STEP_DISTANCE, LABEL))

    print('Model Prediction')
    y_pred_test = model_b.predict(X_test)

    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(y_test, axis=1)

    assert y_test.shape[0] == y_pred_test.shape[0]

    # Evaluation matrices
    print('Model Evaluation - Classification stats')

    class_names = ['SED', 'LPA', 'MVPA']
    cnf_matrix = confusion_matrix(max_y_test, max_y_pred_test)

    stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)

    assessment_result = 'Classes' + '\t' + str(class_names) + '\t' + '\n'
    assessment_result += 'Accuracy' + '\t' + str(stats['accuracy']) + '\t' + str(stats['accuracy_ci']) + '\n'
    assessment_result += 'Sensitivity' + '\t' + str(stats['sensitivity']) + '\n'
    assessment_result += 'Sensitivity CI' + '\t' + str(stats['sensitivity_ci']) + '\n'
    assessment_result += 'Specificity' + '\t' + str(stats['specificity']) + '\n'
    assessment_result += 'Specificity CI' + '\t' + str(stats['specificity_ci']) + '\n'

    results_descriptions.append(assessment_result)

    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title='CM',
                                          output_filename=join(CLASSIFICATION_RESULTS_FOLDER, 'confusion_matrix.png'))

    result_string = '\n'.join(results_descriptions)
    with open(join(CLASSIFICATION_RESULTS_FOLDER, 'result_report.txt'), "w") as text_file:
        text_file.write(result_string)


def evaluate_regression_modal(REGRESS_MODEL_ROOT_FOLDER, REGRESSION_RESULTS_FOLDER, test_X_data, test_Y_data, test_ID_user,
                                 TIME_PERIODS, STEP_DISTANCE):

    assert test_X_data.shape[0] == test_Y_data.shape[0] == len(test_ID_user)

    # Test data placeholder
    X_test, y_test = test_X_data, test_Y_data
    ID_test = test_ID_user

    # Data -> Model ready
    print('Data -> Model ready.')
    num_time_periods, num_sensors = X_test.shape[1], X_test.shape[2]
    input_shape = (num_time_periods * num_sensors)
    X_test = X_test.reshape(X_test.shape[0], input_shape)

    # Convert type for Keras otherwise Keras cannot process the data
    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    # Select best model
    model_files = [join(REGRESS_MODEL_ROOT_FOLDER, f) for f in listdir(REGRESS_MODEL_ROOT_FOLDER) if
                   isfile(join(REGRESS_MODEL_ROOT_FOLDER, f)) and '.h5' in f]
    model_b = load_model(model_files[0])

    # Evaluate against test data

    results_descriptions = []

    LABEL = 'activity_classes'
    results_descriptions.append(
        'Time Period = {}, Step Distance = {}, Label = {}'.format(TIME_PERIODS, STEP_DISTANCE, LABEL))

    print('Model Prediction')
    y_pred_test = model_b.predict(X_test)

    assert y_test.shape[0] == y_pred_test.shape[0]

    print('Model Evaluation - plot regression plot')
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test)
    plt.xlabel('Actual EE')
    plt.ylabel('Predicted EE')
    plt.savefig(join(REGRESSION_RESULTS_FOLDER, 'actual_vs_predicted_met.png'))
    plt.clf()
    plt.close()

    print('Model Evaluation - Regression stats')
    y_pred_test_1d_list = [list(i)[0] for i in list(y_pred_test)]
    corr = pearsonr(list(y_test), y_pred_test_1d_list)

    results_descriptions.append('\n\n -------RESULTS-------\n\n')
    results_descriptions.append('Pearsons Correlation = {}'.format(corr))
    results_descriptions.append('MSE - {}'.format(mean_squared_error(y_test, y_pred_test)))
    results_descriptions.append('RMSE - {}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
    results_descriptions.append('RMSE - {}'.format(sqrt(mean_squared_error(y_test, y_pred_test))))
    results_descriptions.append('R2 Error - {}'.format(r2_score(y_test, y_pred_test)))
    results_descriptions.append('Explained Variance Score - {}'.format(explained_variance_score(y_test, y_pred_test)))

    print('Model Evaluation - Regression to Intensity')
    class_names = ['SED', 'LPA', 'MVPA']
    y_test_ai = SE.EnergyTransform.met_to_intensity(y_test)
    y_pred_test_ai = SE.EnergyTransform.met_to_intensity(y_pred_test)

    cnf_matrix = confusion_matrix(y_test_ai, y_pred_test_ai)

    stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)

    assessment_result = 'Classes' + '\t' + str(class_names) + '\t' + '\n'
    assessment_result += 'Accuracy' + '\t' + str(stats['accuracy']) + '\t' + str(stats['accuracy_ci']) + '\n'
    assessment_result += 'Sensitivity' + '\t' + str(stats['sensitivity']) + '\n'
    assessment_result += 'Sensitivity CI' + '\t' + str(stats['sensitivity_ci']) + '\n'
    assessment_result += 'Specificity' + '\t' + str(stats['specificity']) + '\n'
    assessment_result += 'Specificity CI' + '\t' + str(stats['specificity_ci']) + '\n'

    results_descriptions.append(assessment_result)

    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title='CM',
                                          output_filename=join(REGRESSION_RESULTS_FOLDER, 'confusion_matrix.png'))

    results_df = pd.DataFrame(
        {'subject': ID_test,
         'waist_ee': list(y_test),
         'predicted_ee': [list(i)[0] for i in list(y_pred_test)]
         })

    def clean_data_points(data):
        data = data.assign(waist_ee_cleaned=data['waist_ee'])
        data = data.assign(predicted_ee_cleaned=data['predicted_ee'])
        data.loc[(data['predicted_ee'] < 1), 'predicted_ee_cleaned'] = 1
        return data

    results_df = clean_data_points(results_df)

    print('Model Evaluation - BlandAltman')
    SE.BlandAltman.bland_altman_paired_plot_tested(results_df, '{}'.format(REGRESSION_RESULTS_FOLDER), 1, log_transformed=True,
                                                   min_count_regularise=False,
                                                   output_filename=join(REGRESSION_RESULTS_FOLDER, 'bland_altman'))

    result_string = '\n'.join(results_descriptions)
    with open(join(REGRESSION_RESULTS_FOLDER, 'result_report.txt'), "w") as text_file:
        text_file.write(result_string)


def run(FOLDER_NAME, training_version, trial_id):

    model_folder_name = FOLDER_NAME.split('-')
    model_folder_name[-1] = str(int(int(model_folder_name[-1])/2))
    model_folder_name = '-'.join(model_folder_name)

    DATA_ROOT = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/'
    TRAIN_TEST_SUBJECT_PICKLE = 'participant_split/train_test_split.pickle'
    CLASSIF_MODEL_ROOT_FOLDER = '../output/classification/v{}/{}/model_out/'.format(training_version, model_folder_name)
    REGRESS_MODEL_ROOT_FOLDER = '../output/regression/v{}/{}/model_out/'.format(training_version, model_folder_name)
    TEST_DATA_FOLDER = DATA_ROOT + 'Week 2/test_data/{}/'.format(FOLDER_NAME)
    OUTPUT_FOLDER_ROOT = '../output/test_results/v{}/{}'.format(trial_id, FOLDER_NAME)

    CLASSIFICATION_RESULTS_FOLDER = OUTPUT_FOLDER_ROOT + '/classif_results/'
    REGRESSION_RESULTS_FOLDER = OUTPUT_FOLDER_ROOT + '/regress_results/'

    # Create output folders
    for f in [OUTPUT_FOLDER_ROOT, CLASSIFICATION_RESULTS_FOLDER, REGRESSION_RESULTS_FOLDER]:
        if not exists(f):
            makedirs(f)

    # The number of steps within one time segment
    TIME_PERIODS = int(FOLDER_NAME.split('-')[1])
    STEP_DISTANCE = int(FOLDER_NAME.split('-')[3])

    # Test Train Split
    with open(TRAIN_TEST_SUBJECT_PICKLE, 'rb') as handle:
        split_dict = pickle.load(handle)
    test_subjects = split_dict['test']

    # Load all data
    all_files_test = [join(TEST_DATA_FOLDER, f) for f in listdir(TEST_DATA_FOLDER) if isfile(join(TEST_DATA_FOLDER, f))
                      and f.split(' ')[0] in test_subjects]
    X_data, Y_data_classif, Y_data_regress, ID_user = load_data(all_files_test)

    print('Evaluating Classification')
    evaluate_classification_modal(CLASSIF_MODEL_ROOT_FOLDER, CLASSIFICATION_RESULTS_FOLDER, X_data, Y_data_classif,
                                  ID_user, TIME_PERIODS, STEP_DISTANCE)

    print('Evaluating Regression')
    evaluate_regression_modal(REGRESS_MODEL_ROOT_FOLDER, REGRESSION_RESULTS_FOLDER, X_data, Y_data_regress, ID_user,
                              TIME_PERIODS, STEP_DISTANCE)

    print('Completed {}.'.format(FOLDER_NAME))


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/Week 2/test_data/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f))]

    trial_num = 5
    training_version = 4
    allowed_list = [6000]

    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\n\nProcessing {}'.format(f))
        run(f, training_version, trial_num)

