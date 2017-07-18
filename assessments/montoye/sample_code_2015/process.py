import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join


# filename = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\LSM203 Waist (2016-11-02)_FE.csv'.replace('\\', '/')
# data = pd.read_csv(filename)
# data.to_csv('sample_code_2015/participant_LSM203.txt', sep='\t')

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title + "Confusion matrix")
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print(title + ' confusion matrix')
#
#     print(cm)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# filename = 'sample_code_2015/lsm203_actual_pred.csv'.replace('\\', '/')
# data = pd.read_csv(filename, skiprows=1)
#
# predicted = data['predicted_intensity']
# target = data['waist_intensity_moderated']
# class_names = ['light', 'moderate', 'vigorous']
#
# print("LR Mean squared error: %.2f" % np.mean((predicted - target) ** 2))
#
# # The R squared score
# print("LR R squared score: %.2f" % r2_score(target, predicted))
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(target, predicted)
# np.set_printoptions(precision=2)
#
# # Plot non-normalized confusion matrix
# plt.figure(1)
# plot_confusion_matrix(cnf_matrix, classes=class_names, title=' Montoye 2015 ')
#
# plt.show()

input_filepath = 'E:\Projects/accelerometer-project/assessments\montoye\sample_code_2015/Participant 1 example data_MSSE 2015 study.txt'.replace('\\', '/')
input_data = pd.read_csv(input_filepath, sep="\t", usecols=[1])

results_filepath = 'E:\Projects/accelerometer-project/assessments\montoye\sample_code_2015/prediction_participant1_1.csv'.replace('\\', '/')
result_data = pd.read_csv(results_filepath, usecols=[1])

input_data['ee'] = result_data['V1']

output_filepath = 'E:\Projects/accelerometer-project/assessments\montoye\sample_code_2015/output.csv'.replace('\\', '/')

input_data.to_csv(output_filepath)