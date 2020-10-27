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
from keras.utils import np_utils
import itertools
import matplotlib.pyplot as plt
from scipy.stats.stats import spearmanr
from sklearn.metrics import confusion_matrix
import supervised_learning.modeling.statistical_extensions as SE

print('Keras version ', keras.__version__)


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
        print("Normalized confusion matrix")
    else:
        print(title + ' confusion matrix')

    print(cm)

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


def get_time_in(labels, time_epoch):

    unique, counts = np.unique(labels, return_counts=True)
    outcomes = {}
    for i, j in zip(unique, counts):
        outcomes[i] = j

    SB = outcomes[0] / (6000 / time_epoch) if 0 in outcomes else 0
    LPA = outcomes[1] / (6000 / time_epoch) if 1 in outcomes else 0
    MVPA = outcomes[2] / (6000 / time_epoch) if 2 in outcomes else 0

    return SB, LPA, MVPA


def load_data(filenames, demo=False, user_list=None):

    # Single row for each
    data_dict = {}
    cc = 1
    for filename in tqdm(filenames, desc='Loading data'):

        user_id = filename.split('/')[-1].split()[0]

        if user_list is not None and user_id not in user_list:
            continue

        npy = np.load(filename, allow_pickle=True)

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
            if cc > 10:
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

    output_predicted = []
    output_truth = []
    num_classes = 3
    for key, data in tqdm(data_dict.items(), desc='Evaluating {} - {}:'.format(TIME_PERIODS, group)):

        # Test data placeholder
        X_test, y_test = data['X_data'], data['Y_data_classif']

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

        output_truth.append(max_y_test)
        output_predicted.append(max_y_pred_test)

        # Predicted values for time
        # predicted_SB, predicted_LPA, predicted_MVPA = get_time_in(max_y_pred_test, TIME_PERIODS)

        """Evaluation matrices (3 CLASS)"""
        # cnf_matrix = confusion_matrix(max_y_test, max_y_pred_test)
        # class_names = ['SED', 'LPA', 'MVPA']

        # plt.figure()
        # conf_mat_output_filename = join(CLASSIFICATION_RESULTS_FOLDER, 'conf_mat_{}.png'.format(key))
        # SE.GeneralStats.plot_confusion_matrix(cnf_matrix, normalize=False, classes=class_names, title=key, output_filename=conf_mat_output_filename)

    # test completed
    cnf_matrix = confusion_matrix(np.concatenate(output_truth), np.concatenate(output_predicted))
    class_names = ['SED', 'LPA', 'MVPA']

    plt.figure()
    conf_mat_output_filename = join(CLASSIFICATION_RESULTS_FOLDER, 'conf_mat_overall.png')
    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, normalize=True, classes=class_names, title='overall', output_filename=conf_mat_output_filename)


def run(FOLDER_NAME, training_version, group, data_root, demo=False, user_list=None):

    TEST_DATA_FOLDER = data_root + '/{}/{}/'.format(FOLDER_NAME, group)
    CLASSIF_MODEL_ROOT_FOLDER = 'E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}/classification/{}/model_out/'.format(training_version, FOLDER_NAME)
    OUTPUT_FOLDER_ROOT = 'E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}/classification/{}/individual_results_JSS_V2/{}/'.format(training_version, FOLDER_NAME, group)

    # Create output folders
    for f in [OUTPUT_FOLDER_ROOT]:
        if not exists(f):
            makedirs(f)

    # The number of steps within one time segment
    TIME_PERIODS = int(FOLDER_NAME.split('-')[1])

    # Load data
    all_files_test = [join(TEST_DATA_FOLDER, f) for f in listdir(TEST_DATA_FOLDER) if isfile(join(TEST_DATA_FOLDER, f))]

    print('Loading data...')
    data_dictionary = load_data(all_files_test, demo=demo, user_list=user_list)

    evaluate_classification_modal(CLASSIF_MODEL_ROOT_FOLDER, OUTPUT_FOLDER_ROOT, data_dictionary, TIME_PERIODS, group)


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1_Combined\model_ready/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f)) and (f.split('-')[1] != f.split('-')[3])]

    training_version = '1-12_Dec_Trail'
    allowed_list = [6000]
    groups = ['test', 'train_test']

    req_user_list = ['LSM148', 'LSM148A']

    for f, grp in itertools.product(all_files, groups):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\n\nProcessing {} for {}'.format(f, grp))
        run(f, training_version, grp, temp_folder, demo=False, user_list=None)

    print('Completed.')
