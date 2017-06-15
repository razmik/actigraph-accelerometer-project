import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join

path = "D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/".replace('\\', '/')

files = [f for f in listdir(path) if isfile(join(path, f))]

data = pd.DataFrame()
for file in files:
    data = data.append(pd.read_csv(path + file), ignore_index=True)

# filename_only = 'LSM255_(2016-11-01)_row_0_to_1920'
# epoch_filename = "D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/" + filename_only + ".csv".replace('\\', '/')

"""
If only a single file needs to be assessed.
"""
data = pd.read_csv('D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/LSM204_(2016-11-02)_row_16560_to_18440.csv')


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


# data = pd.read_csv(epoch_filename)
del data['Unnamed: 0']

print("bandpass wrist vm vs. wrist_epoch_15:", round(stats.pearsonr(data['band_vm'], data['wrist_vm_15'])[0], 2))
print("wrist processed vm vs. waist process:", round(stats.pearsonr(data['waist_vm_60'], data['wrist_vm_60'])[0], 2))
print("bandpass wrist vm vs. waist processed:", round(stats.pearsonr(data['band_vm'], data['waist_vm_60'])[0], 2))

data.loc[data['waist_intensity'] == 1, 'target_met_category'] = 1
data.loc[data['waist_intensity'] == 2, 'target_met_category'] = 2
data.loc[data['waist_intensity'] == 3, 'target_met_category'] = 3
data.loc[data['waist_intensity'] == 4, 'target_met_category'] = 3

"""
Linear Regression
"""

target = data['target_met_category']
lr_estimated_filtered = data['lr_estimated_met_bandpass_filtered']
lr_estimated_not_filtered = data['lr_estimated_met_notfiltered']

class_names = ['light', 'moderate', 'vigorous']

"""
Model evaluation statistics
"""
print("Linear Regression - Filtered")

# The mean squared error
print("LR Mean squared error: %.2f"
      % np.mean((lr_estimated_filtered - target) ** 2))

# The R squared score
# r2_score(y_true, y_pred, sample_weight=None, multioutput=None)
print("LR R squared score: %.2f"
      % r2_score(target, lr_estimated_filtered))

# Compute confusion matrix
cnf_matrix = confusion_matrix(target, lr_estimated_filtered)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(1)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Linear Regression - Filtered')



print("Linear Regression - Not Filtered")

# The mean squared error
print("LR Mean squared error: %.2f"
      % np.mean((lr_estimated_not_filtered - target) ** 2))

# The R squared score
# r2_score(y_true, y_pred, sample_weight=None, multioutput=None)
print("LR R squared score: %.2f"
      % r2_score(target, lr_estimated_not_filtered))

# Compute confusion matrix
cnf_matrix = confusion_matrix(target, lr_estimated_not_filtered)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(2)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Linear Regression - Not Filtered')


plt.show()
