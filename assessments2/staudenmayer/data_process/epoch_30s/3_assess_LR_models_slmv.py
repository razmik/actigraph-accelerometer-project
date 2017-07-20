import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
import math
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

    def met_to_intensity(row):
        ee = data['waist_ee'][row.name]
        return 1 if ee < 1.5 else 2 if ee < 3 else 3

    # convert activity intensity to 3 levels - SB, LPA, MVPA
    data['target_met_category'] = data.apply(met_to_intensity, axis=1)

    """
    Linear Regression
    """

    data['lr_estimated_met'] = 1.89378 + (5.50821 * data['raw_wrist_sdvm']) - (0.02705 * data['raw_wrist_mangle'])
    data.loc[data['lr_estimated_met'] < 3, 'lr_estimated_met_category'] = 1
    data.loc[(3 <= data['lr_estimated_met']) & (data['lr_estimated_met'] < 6), 'lr_estimated_met_category'] = 2
    data.loc[data['lr_estimated_met'] >= 6, 'lr_estimated_met_category'] = 3

    target_category = data['target_met_category']
    lr_estimated_category = data['lr_estimated_met_category']

    target_met = data['waist_ee']
    lr_estimated_met = data['lr_estimated_met']

    class_names = ['SB', 'LPA', 'MVPA']


    """
    Model evaluation statistics
    """
    print("Linear Regression Equation")

    # The mean squared error
    print("LR Mean squared error [MET]: %.2f" % np.mean((lr_estimated_met - target_met) ** 2))
    print("LR Mean squared error [Category]: %.2f" % np.mean((lr_estimated_category - target_category) ** 2))

    # The R squared score
    print("LR R squared score [MET]: %.2f" % r2_score(target_met, lr_estimated_met))
    print("LR R squared score [Category]: %.2f" % r2_score(target_category, lr_estimated_category))

    # Precision and Recall for Linear Regression model
    precision, recall, fscore, support = precision_recall_fscore_support(target_category, lr_estimated_category, average='macro')
    print('LR overall precision: {}'.format(precision))
    print('LR overall recall: {}'.format(recall))
    print('LR overall fscore: {}'.format(fscore))
    print('LR overall support: {}'.format(support))

    precision, recall, fscore, support = precision_recall_fscore_support(target_category, lr_estimated_category)
    print('LR precision: {}'.format(precision))
    print('LR recall: {}'.format(recall))
    print('LR fscore: {}'.format(fscore))
    print('LR support: {}'.format(support))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target_category, lr_estimated_category)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=status + ' Linear Regression')

    plt.show()


print('Evaluation for non-filtered raw readings.')

experiment = 'LSM2'
week = 'Week 1'
day = 'Wednesday'

input_file_path = ("D:/Accelerometer Data\Assessment\staudenmayer/"+experiment+"/"+week+"/"+day+"/Epoch30/").replace('\\', '/')

data = pd.DataFrame()
input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]
for file in input_filenames:
    data = data.append(pd.read_csv(input_file_path + file), ignore_index=True)

    # # Verify if all the values are NON NAN
    # temp = pd.read_csv(input_file_path + file)
    # count = 0
    # for val in temp['raw_wrist_mangle']:
    #     if math.isnan(val):
    #         print(file, count)
    #     count += 1

del data['Unnamed: 0']

print('Completed reading data.')

evaluate_models(data, 'Staudenmayer LR - 30 sec')