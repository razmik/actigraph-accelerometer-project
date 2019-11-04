"""
Ref: https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
"""
# https://github.com/ni79ls/har-keras-cnn/blob/master/20180903_Keras_HAR_WISDM_CNN_v1.0_for_medium.py
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import listdir, makedirs
from os.path import join, isfile, exists

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import random
from tqdm import tqdm

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import supervised_learning.modeling.utils as utils
import supervised_learning.modeling.statistical_extensions as SE

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from talos import Scan, Evaluate

import datetime
from time import time
from tensorflow.python.keras.callbacks import TensorBoard


pd.options.display.float_format = '{:.1f}'.format
sns.set()  # Default seaborn look and feel
plt.style.use('ggplot')
print('Keras version ', keras.__version__)


def predictive_model(X_train, y_train, x_val, y_val, params):

    input_shape = (params['time_periods'] * params['num_sensors'])

    model = Sequential()
    model.add(Reshape((params['time_periods'], params['num_sensors']), input_shape=(input_shape,)))
    model.add(Conv1D(params['conv_1_filters'], params['conv_1_kernal_size'], activation='relu', input_shape=(params['time_periods'], params['num_sensors']), name='conv_1'))
    model.add(Conv1D(params['conv_2_filters'], params['conv_2_kernal_size'], activation='relu', name='conv_2'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(params['conv_3_filters'], params['conv_3_kernal_size'], activation='relu', name='conv_3'))
    model.add(Conv1D(params['conv_4_filters'], params['conv_4_kernal_size'], activation='relu', name='conv_4'))
    model.add(params['global_pooling'])
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error',
                    optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train,
                          y_train,
                          batch_size=params['batch_size'],
                          epochs=params['epochs'],
                          callbacks=[
                              EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=int(params['epochs'] / 4),
                                            verbose=0,
                                            mode='auto')
                          ],
                          validation_data=[x_val, y_val],
                          verbose=0)

    return history, model


if __name__ == '__main__':

    SAMPLE_SIZE = 40
    DATA_FOLDER = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/supervised_data/window-300-overlap-150/'
    MODEL_FOLDER = 'model_out/'
    if not exists(MODEL_FOLDER):
        makedirs(MODEL_FOLDER)

    # The number of steps within one time segment
    TIME_PERIODS = int(DATA_FOLDER.split('/')[-2].split('-')[1])
    # The steps to take from one segment to the next; if this value is equal to
    # TIME_PERIODS, then there is no overlap between the segments
    STEP_DISTANCE = int(DATA_FOLDER.split('/')[-2].split('-')[3])
    LABEL = 'energy_expenditure'
    print('Time Period = {}, Step Distance = {}, Label = {}'.format(TIME_PERIODS, STEP_DISTANCE, LABEL))

    # Load all data
    all_files = [join(DATA_FOLDER, f) for f in listdir(DATA_FOLDER) if isfile(join(DATA_FOLDER, f))]

    # Sample data files to reduce the size
    all_files = random.sample(all_files, SAMPLE_SIZE)

    # Ready up the data
    X_data = []
    Y_data = []
    ID_user = []
    for filename in tqdm(all_files):
        npy = np.load(filename, allow_pickle=True)
        X_data.append(npy.item().get('segments'))
        Y_data.append(npy.item().get('energy_e'))

        user_id = filename.split('/')[-1][:6]
        data_length = npy.item().get('energy_e').shape[0]
        ID_user.extend([user_id for _ in range(data_length)])

    X_data = np.concatenate(X_data, axis=0)
    Y_data = np.concatenate(Y_data, axis=0)

    assert X_data.shape[0] == Y_data.shape[0] == len(ID_user)
    print('Length of dataset = {}'.format(Y_data.shape[0]))
    print('Number of data points used = {}'.format(Y_data.shape[0] * TIME_PERIODS / 2))
    print('Unique subjects = {}'.format(len(set(ID_user))))

    # Set training/ testing data
    split_point = int(len(ID_user) * .6)
    print('Train subjects = {}'.format(len(set(ID_user[:split_point]))))
    print('Test subjects = {}'.format(len(set(ID_user[split_point:]))))
    print('Overlapping subjects = {}'.format(
        len(set(ID_user[:split_point])) + len(set(ID_user[split_point:])) - len(set(ID_user))))

    X_train, X_test = X_data[:split_point], X_data[split_point:]
    y_train, y_test = Y_data[:split_point], Y_data[split_point:]
    ID_train, ID_test = ID_user[:split_point], ID_user[split_point:]

    # Inspect x data
    print('x_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('id_train shape: ', len(ID_train))

    # Data Shaping
    num_time_periods, num_sensors = X_train.shape[1], X_train.shape[2]
    input_shape = (num_time_periods * num_sensors)
    X_train = X_train.reshape(X_train.shape[0], input_shape)
    X_test = X_test.reshape(X_test.shape[0], input_shape)

    # Convert type for Keras otherwise Keras cannot process the data
    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")
    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    # Hyper-parameter tuning
    hyperparameters = {'batch_size': [32],
         'epochs': [80],
         'conv_1_filters': [80],
         'conv_1_kernal_size': [10],
         'conv_2_filters': [100],
         'conv_2_kernal_size': [10],
         'conv_3_filters': [160],
         'conv_3_kernal_size': [10],
         'conv_4_filters': [180],
         'conv_4_kernal_size': [10],
         'global_pooling': [GlobalAveragePooling1D(), GlobalMaxPooling1D()],
         'dropout': [0.3, 0.5, 0.8],
         'time_periods': [num_time_periods],
         'num_sensors': [num_sensors]
         }

    # Best hyper-parameter search
    h = Scan(X_train, y_train,
             model=predictive_model,
             params=hyperparameters,
             print_params=True,
             experiment_name='exp_2',
             reduction_metric="val_loss")

    # Evaluate
    e = Evaluate(h)
    evaluation = e.evaluate(X_test,
                            y_test,
                            model_id=None,
                            folds=3,
                            shuffle=True,
                            task='continuous',
                            metric='val_loss',
                            print_out=True,
                            asc=True)

    print('Completed at {}'.format(datetime.datetime.now()))







