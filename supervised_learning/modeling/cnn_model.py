"""
Ref: https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
"""
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from os import makedirs, listdir
from os.path import join, exists, isfile

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils


# DATA_ROOT = "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/supervised_data/window-500/"
DATA_ROOT = "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/testss/"
MODEL_OUTPUT_PATH = '../output/CNN/'
if not exists(MODEL_OUTPUT_PATH):
    makedirs(MODEL_OUTPUT_PATH)


def plot_results(history):

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
    plt.show()


# def get_model(input_shape):
#
#     model_m = Sequential()
#     model_m.add(Conv1D(100, 10, activation='relu', input_shape=(input_shape, )))
#     model_m.add(Conv1D(100, 10, activation='relu'))
#     model_m.add(MaxPooling1D(3))
#     model_m.add(Conv1D(160, 10, activation='relu'))
#     model_m.add(Conv1D(160, 10, activation='relu'))
#     model_m.add(GlobalAveragePooling1D())
#     model_m.add(Dropout(0.5))
#     model_m.add(Dense(1, activation='linear'))
#
#     print(model_m.summary())
#     return model_m


# def get_callbacks():
#
#     # The EarlyStopping callback monitors training accuracy:
#     # if it fails to improve for two consecutive epochs,
#     # training stops early
#     callbacks_list = [
#         keras.callbacks.ModelCheckpoint(
#             filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
#             monitor='val_loss', save_best_only=True),
#         keras.callbacks.EarlyStopping(monitor='acc', patience=1)
#     ]
#
#     return callbacks_list


if __name__ == '__main__':

    print('test')

    # data_files = [f for f in listdir(DATA_ROOT) if isfile(join(DATA_ROOT, f))]
    #
    # df = pd.read_csv(data_files[0])
    #
    # X, Y = df[:, 0], df[:, 1:]
    #
    # # Hyper-parameters
    # BATCH_SIZE = 256
    # EPOCHS = 100
    #
    # # CNN model
    # model = get_model(X.shape[0])
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    #
    # # Train the model
    # history = model.fit(X, Y,
    #                     batch_size=BATCH_SIZE,
    #                     epochs=EPOCHS,
    #                     callbacks=get_callbacks(),
    #                     validation_split=0.2,
    #                     verbose=1)
    #
    # plot_results(history)

