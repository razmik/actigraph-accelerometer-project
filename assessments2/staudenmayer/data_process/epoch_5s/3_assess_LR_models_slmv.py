import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
import math, time
from os import listdir
from os.path import isfile, join
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')
import statistical_extensions

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


def predict(data):
    data['predicted_ee'] = 1.89378 + (5.50821 * data['raw_wrist_sdvm']) - (0.02705 * data['raw_wrist_mangle'])
    return data


def evaluate_models(data, status):

    def met_to_intensity_waist_ee(row):
        ee = data['waist_ee'][row.name]
        return 1 if ee <= 1.5 else 2 if ee < 3 else 3

    def met_to_intensity_lr_estimated_ee(row):
        ee = data['predicted_ee'][row.name]
        return 1 if ee <= 1.5 else 2 if ee < 3 else 3

    # convert activity intensity to 3 levels - SB, LPA, MVPA
    data['target_met_category'] = data.apply(met_to_intensity_waist_ee, axis=1)
    data['lr_predicted_met_category'] = data.apply(met_to_intensity_lr_estimated_ee, axis=1)

    target_category = data['target_met_category']
    lr_estimated_category = data['lr_predicted_met_category']

    target_met = data['waist_ee']
    lr_estimated_met = data['predicted_ee']

    class_names = ['SB', 'LPA', 'MVPA']

    """
    Model evaluation statistics
    """
    print("Linear Regression Equation")

    # The mean squared error
    print("LR Mean squared error [MET]: %.2f" % np.mean((lr_estimated_met - target_met) ** 2))
    print("LR Mean squared error [Category]: %.2f" % np.mean((lr_estimated_category - target_category) ** 2))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target_category, lr_estimated_category)
    np.set_printoptions(precision=2)

    stats = statistical_extensions.GeneralStats.evaluation_statistics(cnf_matrix)
    print('Accuracy', stats['accuracy'])
    print('Sensitivity', stats['sensitivity'])
    print('Specificity', stats['specificity'])

    # Plot non-normalized confusion matrix
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=status)


print('Start - Reading')

experiments = ['LSM1', 'LSM2']
week = 'Week 1'
days = ['Wednesday', 'Thursday']
epoch = 'Epoch5'
model_title = 'Staudenmayer Linear Regression (5 sec epoch)'

start_reading = time.time()

count = 0
for experiment in experiments:
    if count != 0:
        break
    for day in days:
        if count != 0:
            break
        input_file_path = ("E:/Data/Accelerometer_Processed_Raw_Epoch_Data/"+experiment+"/"+week+"/"+day+"/"+epoch+"/").replace('\\', '/')
        input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]

        for file in input_filenames:
            dataframe = pd.read_csv(input_file_path + file)
            dataframe['subject'] = file.split('_(2016')[0]

            if count != 0:
                results = results.append(dataframe, ignore_index=True)
            else:
                results = dataframe
                count += 1

        print('completed', experiment, day)

end_reading = time.time()

print('Reading time', round(end_reading-start_reading, 2), '(s)\nStart - Assessment')


"""Prediction"""
results = predict(results)


"""General Assessment"""
evaluate_models(results, model_title)


"""Bland Altman Plot"""
results = statistical_extensions.BlandAltman.clean_data_points(results)
statistical_extensions.BlandAltman.bland_altman_paired_plot_tested(results, model_title, 2, log_transformed=True, min_count_regularise=False)

print('Assessment completed.')

plt.show()
