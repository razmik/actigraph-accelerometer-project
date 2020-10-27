from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from os import listdir, makedirs
from os.path import join, isfile, exists
import numpy as np
import pandas as pd
from tqdm import tqdm
import keras
from keras.models import load_model
import itertools
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, confusion_matrix, mean_squared_error
import supervised_learning.modeling.statistical_extensions as SE


def plot_confusion_matrix(cm, classes, normalize=False, title='', output_filename=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title + " Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(output_filename, dpi=600)
    plt.clf()
    plt.close()


def load_data_overall(filenames, demo=False):

    X_data = []
    Y_data = []
    ID_user = []
    counter = 0
    for filename in tqdm(filenames, desc='Loading data.'):
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


def load_data_individual(filenames, demo=False, user_list=None):

    # Single row for each
    data_dict = {}
    ccc = 1
    for filename in tqdm(filenames, desc='Loading data'):

        user_id = filename.split('/')[-1].split()[0]

        if user_list is not None and user_id not in user_list:
            continue

        npy = np.load(filename, allow_pickle=True)

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
            if ccc > 15:
                break

    return data_dict


def get_time_in(labels, time_epoch):

    unique, counts = np.unique(labels, return_counts=True)
    outcomes = {}
    for i, j in zip(unique, counts):
        outcomes[i] = j

    SB = outcomes[1] / (6000 / time_epoch) if 1 in outcomes else 0
    LPA = outcomes[2] / (6000 / time_epoch) if 2 in outcomes else 0
    MVPA = outcomes[3] / (6000 / time_epoch) if 3 in outcomes else 0

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

    return y_pred_test_1d_list


def evaluate_regression_modal(model_b, REG_RESULTS_FOLDER, data_dict, TIME_PERIODS, group):


    output_predicted = []
    output_truth = []
    for key, data in tqdm(data_dict.items(), desc='Predicting'):

        # Test data placeholder
        X_test, y_test, y_class = data['X_data'], data['Y_data_regress'], data['Y_data_classif']

        """Regression"""
        # Overall
        y_pred_test = calculate_error(X_test, y_test, model_b)

        """Classification"""
        y_pred_test = np.asarray(y_pred_test)
        max_y_test = SE.EnergyTransform.met_to_intensity(y_test)
        max_y_pred_test = SE.EnergyTransform.met_to_intensity(y_pred_test)

        output_truth.append(max_y_test)
        output_predicted.append(max_y_pred_test)

        # """Evaluation matrices (3 CLASS)"""
        # cnf_matrix = confusion_matrix(max_y_test, max_y_pred_test)
        # class_names = ['SED', 'LPA', 'MVPA']
        #
        # plt.figure()
        # conf_mat_output_filename = join(REG_RESULTS_FOLDER, 'conf_mat_{}.png'.format(key))
        # SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title=key, normalize=False,
        #                                       output_filename=conf_mat_output_filename)

    # test completed
    cnf_matrix = confusion_matrix(np.concatenate(output_truth), np.concatenate(output_predicted))
    class_names = ['SED', 'LPA', 'MVPA']

    plt.figure()
    conf_mat_output_filename = join(REG_RESULTS_FOLDER, 'conf_mat_overall.png')
    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, normalize=True, classes=class_names, title='overall', output_filename=conf_mat_output_filename)



def bland_altman_eval(model_b, REG_RESULTS_FOLDER, X_data, Y_data, ID_user, group):

    X, y = np.asarray(X_data), np.asarray(Y_data)

    num_time_periods, num_sensors = X.shape[1], X.shape[2]
    input_shape = (num_time_periods * num_sensors)
    X = X.reshape(X.shape[0], input_shape)

    # Convert type for Keras otherwise Keras cannot process the data
    X = X.astype("float32")
    y = y.astype("float32")

    # Evaluate against test data
    print('Predicting...')
    y_pred_test = model_b.predict(X)
    y_pred_test_1d_list = [list(i)[0] for i in list(y_pred_test)]

    # Overall analysis
    print('Overall analysis calculations.')
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

    ba_outfolder = join(REG_RESULTS_FOLDER, 'bland_altman')
    if not exists(ba_outfolder):
        makedirs(ba_outfolder)

    print('Plotting bland-altman.')
    SE.BlandAltman.bland_altman_paired_plot_tested(results_df, 'Bland Altman Analysis', 1, log_transformed=True,
                                                   min_count_regularise=False,
                                                   output_filename=ba_outfolder)


def run(FOLDER_NAME, training_version, group, data_root, demo=False, user_list=None):

    TEST_DATA_FOLDER = data_root + '/{}/{}/'.format(FOLDER_NAME, group)
    REGRESSION_MODEL_ROOT_FOLDER = 'E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}/regression/{}/model_out/'.format(training_version, FOLDER_NAME)
    OUTPUT_FOLDER_ROOT = 'E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}/regression/{}/individual_results_JSS_V2/{}/'.format(training_version, FOLDER_NAME, group)
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
    # test_X_data, test_Y_data, test_ID_user = load_data_overall(all_files_test, demo)
    # bland_altman_eval(model_b, OUTPUT_FOLDER_ROOT_OVERALL, test_X_data, test_Y_data, test_ID_user, group)
    # del test_X_data
    # del test_Y_data
    # del test_ID_user

    # Load data for individual assessment
    data_dictionary = load_data_individual(all_files_test, demo=demo, user_list=user_list)
    evaluate_regression_modal(model_b, OUTPUT_FOLDER_ROOT, data_dictionary, TIME_PERIODS, group)


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1_Combined\model_ready/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f)) and (f.split('-')[1] != f.split('-')[3])]

    training_version = '1-12_Dec_Trail'
    allowed_list = [6000]
    groups = ['test', 'train_test']

    req_user_list = []

    for f, grpd in itertools.product(all_files, groups):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\n\nProcessing {} for {}'.format(f, grpd))
        run(f, training_version, grpd, temp_folder, demo=False, user_list=None)

