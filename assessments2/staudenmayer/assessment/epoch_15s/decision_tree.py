import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import itertools


epoch_filename = "D:\Accelerometer Data\Processed\LSM2\Week 1\Wednesday\LSM255_(2016-11-01)_row_0_to_1920.csv".replace('\\', '/')

data = pd.read_csv(epoch_filename)
del data['Unnamed: 0']

print("bandpass wrist vm vs. wrist_epoch_15:", round(stats.pearsonr(data['band_vm'], data['wrist_vm_15'])[0], 2))
print("wrist processed vm vs. waist process:", round(stats.pearsonr(data['waist_vm_60'], data['wrist_vm_60'])[0], 2))
print("bandpass wrist vm vs. waist processed:", round(stats.pearsonr(data['band_vm'], data['waist_vm_60'])[0], 2))

data.loc[data['waist_intensity'] == 1, 'target_met_category'] = 1
data.loc[data['waist_intensity'] == 2, 'target_met_category'] = 2
data.loc[data['waist_intensity'] == 3, 'target_met_category'] = 3
data.loc[data['waist_intensity'] == 4, 'target_met_category'] = 3

data.loc[(data['wrist_sdvm'] <= 0.26) & (data['wrist_mangle'] < -52), 'estimated_met_category'] = 1
data.loc[(data['wrist_sdvm'] <= 0.26) & (data['wrist_mangle'] >= -52), 'estimated_met_category'] = 2
data.loc[(0.26 < data['wrist_sdvm']) & (data['wrist_sdvm'] <= 0.79) & (data['wrist_mangle'] > -53), 'estimated_met_category'] = 2
data.loc[(0.26 < data['wrist_sdvm']) & (data['wrist_sdvm'] <= 0.79) & (data['wrist_mangle'] <= -53), 'estimated_met_category'] = 3
data.loc[data['wrist_sdvm'] > 0.79, 'estimated_met_category'] = 3

target = data['target_met_category']
estimated = data['estimated_met_category']

"""
Model evaluation statistics
"""

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((estimated - target) ** 2))

# The R squared score
# r2_score(y_true, y_pred, sample_weight=None, multioutput=None)
print("R squared score: %.2f"
      % r2_score(target, estimated))

"""
CF matrix: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""

class_names = ['light', 'moderate', 'vigorous']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(target, estimated)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.show()
