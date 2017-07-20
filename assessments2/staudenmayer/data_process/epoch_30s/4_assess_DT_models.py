import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + "Confusion matrix")
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


def evaluate_models(data, status):

    # convert activity intensity to 3 levels
    data.loc[data['actilife_waist_intensity'] == 1, 'target_met_category'] = 1
    data.loc[data['actilife_waist_intensity'] == 2, 'target_met_category'] = 2
    data.loc[data['actilife_waist_intensity'] == 3, 'target_met_category'] = 3
    data.loc[data['actilife_waist_intensity'] == 4, 'target_met_category'] = 3

    """
    Decision Tree
    """

    data.loc[(data['raw_wrist_sdvm'] <= 0.26) & (data['raw_wrist_mangle'] < -52), 'dt_estimated_intensity_category'] = 1
    data.loc[(data['raw_wrist_sdvm'] <= 0.26) & (data['raw_wrist_mangle'] >= -52), 'dt_estimated_intensity_category'] = 2
    data.loc[(0.26 < data['raw_wrist_sdvm']) & (data['raw_wrist_sdvm'] <= 0.79) & (
        data['raw_wrist_mangle'] > -53), 'dt_estimated_intensity_category'] = 2
    data.loc[(0.26 < data['raw_wrist_sdvm']) & (data['raw_wrist_sdvm'] <= 0.79) & (
        data['raw_wrist_mangle'] <= -53), 'dt_estimated_intensity_category'] = 3
    data.loc[data['raw_wrist_sdvm'] > 0.79, 'dt_estimated_intensity_category'] = 3

    target_category = data['target_met_category']
    dt_estimated_category = data['dt_estimated_intensity_category']

    class_names = ['light', 'moderate', 'vigorous']


    """
    Model evaluation statistics
    """

    print("\nDecision Tree")

    # The mean squared error
    print("DT Mean squared error: %.2f" % np.mean((dt_estimated_category - target_category) ** 2))

    # The R squared score
    print("DT R squared score: %.2f" % r2_score(target_category, dt_estimated_category))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target_category, dt_estimated_category)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=status + ' Decision Tree')

    # Precision and Recall for Decision Tree
    precision, recall, fscore, support = precision_recall_fscore_support(target_category, dt_estimated_category, average='macro')
    print('DT overall precision: {}'.format(precision))
    print('DT overall recall: {}'.format(recall))
    print('DT overall fscore: {}'.format(fscore))
    print('DT overall support: {}'.format(support))

    precision, recall, fscore, support = precision_recall_fscore_support(target_category, dt_estimated_category)
    print('DT precision: {}'.format(precision))
    print('DT recall: {}'.format(recall))
    print('DT fscore: {}'.format(fscore))
    print('DT support: {}'.format(support))

    plt.show()


print('Evaluation for non-filtered raw readings.')

experiment = 'LSM2'
week = 'Week 1'
day = 'Wednesday'

input_file_path = ("D:/Accelerometer Data\Assessment\staudenmayer/"+experiment+"/"+week+"/"+day+"/"+"Epoch30/").replace('\\', '/')

data = pd.DataFrame()
input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]
for file in input_filenames:
    data = data.append(pd.read_csv(input_file_path + file), ignore_index=True)

del data['Unnamed: 0']

print('Completed reading data.')

evaluate_models(data, 'Staudenmayer DT - 30 sec')