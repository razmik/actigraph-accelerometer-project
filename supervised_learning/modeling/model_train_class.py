from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import listdir, makedirs
from os.path import join, isfile, exists
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import shutil
from tqdm import tqdm
from time import time
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Reshape, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from tensorflow.python.keras.callbacks import TensorBoard

from sklearn.metrics import confusion_matrix
import supervised_learning.modeling.statistical_extensions as SE

# pd.options.display.float_format = '{:.1f}'.format
# sns.set()  # Default seaborn look and feel
# plt.style.use('ggplot')
print('Keras version ', keras.__version__)


def load_data(filenames):

    X_data = []
    Y_data = []
    ID_user = []
    counter = 0
    for filename in tqdm(filenames):
        npy = np.load(filename, allow_pickle=True)
        X_data.append(npy.item().get('segments'))
        Y_data.append(npy.item().get('activity_classes'))

        user_id = filename.split('/')[-1][:6]
        data_length = npy.item().get('activity_classes').shape[0]
        ID_user.extend([user_id for _ in range(data_length)])

        # counter += 1
        # if counter > 30:
        #     break

    X_data = np.concatenate(X_data, axis=0)
    Y_data = np.concatenate(Y_data, axis=0)

    # Data relabeling from index 0 (use only 3 classes)
    Y_data = np.where(Y_data == 1, 0, Y_data)
    Y_data = np.where(Y_data == 2, 1, Y_data)
    Y_data = np.where(Y_data == 3, 2, Y_data)
    Y_data = np.where(Y_data == 4, 2, Y_data)

    return X_data, Y_data, ID_user


def plot_model(history, MODEL_FOLDER):
    # summarize history for accuracy and loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
    plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.savefig(MODEL_FOLDER + 'learning_history.png')
    plt.clf()
    plt.close()


def run(FOLDER_NAME, trial_id, data_root):

    TRAIN_DATA_FOLDER = data_root + '/{}/train/'.format(FOLDER_NAME)
    GROUPS = ['test', 'train_test']
    TEST_DATA_FOLDERS = {
        'test': data_root + '/{}/test/'.format(FOLDER_NAME),
        'train_test': data_root + '/{}/train_test/'.format(FOLDER_NAME),
        }
    OUTPUT_FOLDER_ROOT = '../output/classification/v{}/{}/'.format(trial_id, FOLDER_NAME)
    MODEL_FOLDER = OUTPUT_FOLDER_ROOT + '/model_out/'
    RESULTS_FOLDER = OUTPUT_FOLDER_ROOT + '/results/'
    if not exists(OUTPUT_FOLDER_ROOT):
        makedirs(OUTPUT_FOLDER_ROOT)
        makedirs(MODEL_FOLDER)
        makedirs(join(RESULTS_FOLDER, 'test'))
        makedirs(join(RESULTS_FOLDER, 'train_test'))

    # Create temp folder to save model outputs
    temp_model_out_folder = 'temp_model_out'
    makedirs(temp_model_out_folder)

    results_descriptions = []

    # The number of steps within one time segment
    TIME_PERIODS = int(FOLDER_NAME.split('-')[1])
    # The steps to take from one segment to the next; if this value is equal to
    # TIME_PERIODS, then there is no overlap between the segments
    STEP_DISTANCE = int(FOLDER_NAME.split('-')[3])
    LABEL = 'activity_classes'
    results_descriptions.append('Time Period = {}, Step Distance = {}, Label = {}'.format(TIME_PERIODS, STEP_DISTANCE, LABEL))

    """Load and Setup Train Data"""
    all_files_train = [join(TRAIN_DATA_FOLDER, f) for f in listdir(TRAIN_DATA_FOLDER) if
                       isfile(join(TRAIN_DATA_FOLDER, f))]

    train_X_data, train_Y_data, train_ID_user = load_data(all_files_train)
    X_train, y_train, ID_train = train_X_data, train_Y_data, train_ID_user

    # Data -> Model ready
    num_time_periods, num_sensors = X_train.shape[1], X_train.shape[2]
    num_classes = len(np.unique(y_train))

    # Set input_shape / reshape for Keras
    # Remark: acceleration data is concatenated in one array in order to feed
    # it properly into coreml later, the preferred matrix of shape [40,3]
    input_shape = (num_time_periods * num_sensors)
    X_train = X_train.reshape(X_train.shape[0], input_shape)

    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")

    # One-hot encoding of y_train labels (only execute once!)
    y_train = np_utils.to_categorical(y_train, num_classes)

    """Load and Setup Test Data"""
    test_data_combined = {}
    for grp in GROUPS:

        test_data_combined[grp] = {}

        fs = [join(TEST_DATA_FOLDERS[grp], f) for f in listdir(TEST_DATA_FOLDERS[grp]) if
              isfile(join(TEST_DATA_FOLDERS[grp], f))]

        test_X_data, test_Y_data, test_ID_user = load_data(fs)
        test_X_data = test_X_data.reshape(test_X_data.shape[0], input_shape).astype("float32")
        test_Y_data = test_Y_data.astype("float32")
        test_Y_data = np_utils.to_categorical(test_Y_data, num_classes)

        test_data_combined[grp]['test_X_data'] = test_X_data
        test_data_combined[grp]['test_Y_data'] = test_Y_data
        test_data_combined[grp]['test_ID_user'] = test_ID_user

    """Model architecture"""
    model_m = Sequential()
    model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
    model_m.add(Conv1D(80, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
    model_m.add(Conv1D(100, 10, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(Conv1D(180, 10, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(220, 10, activation='relu'))
    model_m.add(Conv1D(240, 10, activation='relu'))
    model_m.add(GlobalMaxPooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(num_classes, activation='softmax'))

    callbacks_list = [
        ModelCheckpoint(
            filepath='temp_model_out/best_model.{epoch:03d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir='logs/{}'.format(time())),
        EarlyStopping(monitor='val_loss', patience=15)
    ]

    model_m.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    # Hyper-parameters
    BATCH_SIZE = 32
    EPOCHS = 40

    history = model_m.fit(X_train,
                          y_train,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          callbacks=callbacks_list,
                          validation_split=0.2,
                          verbose=2)

    plot_model(history, MODEL_FOLDER)

    print('Selecting best model.')
    model_files = [(join(temp_model_out_folder, f), int(f.split('-')[0].split('.')[1])) for f in listdir(temp_model_out_folder) if
                   isfile(join(temp_model_out_folder, f)) and f.split('-')[0].split('.')[0] != 'final']

    model_b_name = sorted(model_files, key=lambda x: x[1], reverse=True)[0][0]
    results_descriptions.append('Best model name {}'.format(model_b_name))

    model_b = load_model(model_b_name)
    model_b.save(join(MODEL_FOLDER, model_b_name.split('\\')[1]))
    shutil.rmtree(temp_model_out_folder)

    # Evaluate against test data
    for grp in GROUPS:

        grp_results = results_descriptions[:]

        X_test = test_data_combined[grp]['test_X_data']
        y_test = test_data_combined[grp]['test_Y_data']

        print('Model Evaluation for {}'.format(grp))
        y_pred_test = model_b.predict(X_test)

        # Take the class with the highest probability from the test predictions
        max_y_pred_test = np.argmax(y_pred_test, axis=1)
        max_y_test = np.argmax(y_test, axis=1)

        assert y_test.shape[0] == y_pred_test.shape[0]

        # Evaluation matrices

        class_names = ['SED', 'LPA', 'MVPA']
        cnf_matrix = confusion_matrix(max_y_test, max_y_pred_test)

        stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)

        assessment_result = 'Classes' + '\t' + str(class_names) + '\t' + '\n'
        assessment_result += 'Accuracy' + '\t' + str(stats['accuracy']) + '\t' + str(stats['accuracy_ci']) + '\n'
        assessment_result += 'Sensitivity' + '\t' + str(stats['sensitivity']) + '\n'
        assessment_result += 'Sensitivity CI' + '\t' + str(stats['sensitivity_ci']) + '\n'
        assessment_result += 'Specificity' + '\t' + str(stats['specificity']) + '\n'
        assessment_result += 'Specificity CI' + '\t' + str(stats['specificity_ci']) + '\n'

        grp_results.append(assessment_result)

        SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title='CM',
                                              output_filename=join(RESULTS_FOLDER, grp, 'confusion_matrix.png'))

        result_string = '\n'.join(grp_results)
        with open(join(RESULTS_FOLDER, grp, 'result_report.txt'), "w") as text_file:
            text_file.write(result_string)


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1_Combined/model_ready/'
    all_files = [f for f in listdir(temp_folder) if os.path.isdir(join(temp_folder, f))]

    allowed_windows = [3000, 6000]
    trial_num = 1

    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_windows:
            continue

        print('\n\n\n\nProcessing {}'.format(f))
        run(f, trial_num, temp_folder)

