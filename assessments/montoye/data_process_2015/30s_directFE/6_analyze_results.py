import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import scipy.stats as stats
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

result_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Montoye_2015_predictions/30sec/combined/'.replace('\\', '/')
result_data_files = [f for f in listdir(result_folder) if isfile(join(result_folder, f))]

"""
Evaluate users individually
"""
for file in result_data_files:

    if file.split(' ')[0] == 'LSM204':

        results = pd.read_csv(result_folder+file)

        target_ee = results['waist_ee']
        predicted_ee = results['predicted_ee']
        target_intensity = results['moderated_freedson_intensity']
        predicted_intensity = results['predicted_intensity_by_ee']

        class_names = ['light', 'moderate', 'vigorous']
        timeline = np.arange(len(results))

        # The mean squared error
        print("Montoye 2015 ANN (EE) Mean squared error: %.2f" % np.mean((predicted_ee - target_ee) ** 2))
        print("Montoye 2015 ANN (Intensity) Mean squared error: %.2f" % np.mean(
            (predicted_intensity - target_intensity) ** 2))

        # The R squared score
        print("Montoye 2015 ANN (EE) R squared score: %.2f" % r2_score(target_ee, predicted_ee))
        print("Montoye 2015 ANN (Intensity) R squared score: %.2f" % r2_score(target_intensity, predicted_intensity))

        # Precision and Recall
        precision, recall, fscore, support = precision_recall_fscore_support(target_intensity, predicted_intensity, average='macro')
        print('overall precision: {}'.format(precision))
        print('overall recall: {}'.format(recall))
        print('overall fscore: {}'.format(fscore))
        print('overall support: {}'.format(support))

        precision, recall, fscore, support = precision_recall_fscore_support(target_intensity, predicted_intensity)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))

        # Compute confusion matrix for intensity
        cnf_matrix = confusion_matrix(target_intensity, predicted_intensity)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure(1)
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Montoye 2015 ANN')

        plt.figure(2)
        plt.title('Energy Expenditure')
        plt.xlabel('Red - Freedson VM3 EE   |   Blue - EE Predicted by Montoye 2015 ANN')
        plt.plot(timeline, target_ee, 'r', timeline, predicted_ee, 'b')

        plt.figure(3)
        plt.title('Activity Intensity')
        plt.xlabel('Red  -Freedson VM3 Intensity   |   Blue - Intensity Calculated from Montoye 2015 ANN Predictions')
        plt.plot(timeline, target_intensity, 'r', timeline, predicted_intensity, 'b')

        plt.show()
        sys.exit(0)

"""
Evaluate the users as a whole
"""
count = 0
for file in result_data_files:

    if count == 0:
        results = pd.read_csv(result_folder+file)
    else:
        results = results.append(pd.read_csv(result_folder+file), ignore_index=True)
    count += 1

target_ee = results['waist_ee']
predicted_ee = results['predicted_ee']
target_intensity = results['moderated_freedson_intensity']
predicted_intensity = results['predicted_intensity_by_ee']

class_names = ['light', 'moderate', 'vigorous']

# The mean squared error
print("Montoye 2015 ANN (EE) Mean squared error: %.2f" % np.mean((predicted_ee - target_ee) ** 2))
print("Montoye 2015 ANN (Instensity) Mean squared error: %.2f" % np.mean((predicted_intensity - target_intensity) ** 2))

# The R squared score
print("Montoye 2015 ANN (EE) R squared score: %.2f" % r2_score(target_ee, predicted_ee))
print("Montoye 2015 ANN (Instensity) R squared score: %.2f" % r2_score(target_intensity, predicted_intensity))

# Precision and Recall
precision, recall, fscore, support = precision_recall_fscore_support(target_intensity, predicted_intensity, average='macro')
print('overall precision: {}'.format(precision))
print('overall recall: {}'.format(recall))
print('overall fscore: {}'.format(fscore))
print('overall support: {}'.format(support))

precision, recall, fscore, support = precision_recall_fscore_support(target_intensity, predicted_intensity)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

# Compute confusion matrix for intensity
cnf_matrix = confusion_matrix(target_intensity, predicted_intensity)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(1)
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Montoye 2015 ANN')
plt.show()
sys.exit(0)

"""
Visalize the prediction and actual values
"""
count = 0
for file in result_data_files:

    if count == 0:
        results = pd.read_csv(result_folder+file)
    else:
        break
        results = results.append(pd.read_csv(result_folder+file), ignore_index=True)
    count += 1

target_ee = results['waist_ee']
predicted_ee = results['predicted_ee']

target_intensity = results['moderated_freedson_intensity']
predicted_intensity = results['predicted_intensity_by_ee']

timeline = np.arange(len(results))

plt.figure(1)
plt.title('Energy Expenditure')
plt.xlabel('Red-Freedson VM3 EE   |   Blue- EE Predicted by Montoye 2015 ANN')
plt.plot(timeline, target_ee, 'r', timeline, predicted_ee, 'b')

plt.figure(2)
plt.title('Activity Intensity')
plt.xlabel('Red-Freedson VM3 Intensity   |   Blue- Intensity Calculated from Montoye 2015 ANN Predictions')
plt.plot(timeline, target_intensity, 'r', timeline, predicted_intensity, 'b')

plt.show()
