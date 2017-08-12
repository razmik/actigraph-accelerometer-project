import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')
import bland_altman_extension as BA


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


wrist = 'left_wrist'
# model = 'v1v2'
model = 'v2'
print('Montoye 2017 ANN', wrist)

result_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Assessment2\Montoye_2017_predictions/' + wrist + '/60sec/' + model + '/combined/'.replace(
    '\\', '/')
result_data_files = [f for f in listdir(result_folder) if isfile(join(result_folder, f))]

"""
Evaluate the users as a whole
"""

count = 0
for file in result_data_files:

    dataframe = pd.read_csv(result_folder + file)
    dataframe['subject'] = file.split('_(2016')[0]

    if count == 0:
        results = dataframe
    else:
        results = results.append(dataframe, ignore_index=True)

    count += 1

results = BA.BlandAltman.clean_data_points(results)
BA.BlandAltman.bland_altman_paired_plot_tested(results, 'Montoye 2017 ANN left_wrist', 2, log_transformed=True, min_count_regularise=True)

target_intensity = results['actual_category']
predicted_intensity = results['predicted_category']

target_ee = results['waist_ee']
predicted_ee = results['predicted_ee']

class_names = ['SED', 'LPA', 'MVPA']

# The mean squared error
print("Montoye 2017 ANN (MET) Mean squared error: %.2f" % np.mean((predicted_ee - target_ee) ** 2))
print("Montoye 2017 ANN (Instensity) Mean squared error: %.2f" % np.mean((predicted_intensity - target_intensity) ** 2))

# The R squared score
print("Montoye 2017 ANN (MET) R squared score: %.2f" % r2_score(target_ee, predicted_ee))
print("Montoye 2017 ANN (Instensity) R squared score: %.2f" % r2_score(target_intensity, predicted_intensity))

# Precision and Recall
precision, recall, fscore, support = precision_recall_fscore_support(target_intensity, predicted_intensity,
                                                                     average='macro')
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
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Montoye 2017 ' + wrist + ' ANN')
plt.show()
