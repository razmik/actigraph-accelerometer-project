import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join

experiment = 'LSM2'
week = 'Week 1'
day1 = 'Wednesday'
day2 = 'Thursday'

non_filtered_path1 = "D:/Accelerometer Data/Processed/"+experiment+"/"+week+"/"+day1+"/not_filtered/epoch_15/".replace('\\', '/')
non_filtered_path2 = "D:/Accelerometer Data/Processed/"+experiment+"/"+week+"/"+day2+"/not_filtered/epoch_15/".replace('\\', '/')
filtered_path1 = "D:/Accelerometer Data/Processed/"+experiment+"/"+week+"/"+day1+"/filtered/".replace('\\', '/')
filtered_path2 = "D:/Accelerometer Data/Processed/"+experiment+"/"+week+"/"+day2+"/filtered/".replace('\\', '/')

# filename_only = 'LSM255_(2016-11-01)_row_0_to_1920'
# epoch_filename = "D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/" + filename_only + ".csv".replace('\\', '/')

"""
If only a single file needs to be assessed.
"""
# data = pd.read_csv('D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/LSM204_(2016-11-02)_row_16560_to_18440.csv')


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

    del data['Unnamed: 0']

    # convert activity intensity to 3 levels
    data.loc[data['actilife_waist_intensity'] == 1, 'target_met_category'] = 1
    data.loc[data['actilife_waist_intensity'] == 2, 'target_met_category'] = 2
    data.loc[data['actilife_waist_intensity'] == 3, 'target_met_category'] = 3
    data.loc[data['actilife_waist_intensity'] == 4, 'target_met_category'] = 3

    """
    Linear Regression
    """
    data['lr_estimated_met'] = 1.89378 + (5.50821 * data['raw_wrist_sdvm']) - (0.02705 * data['raw_wrist_mangle'])
    data.loc[data['lr_estimated_met'] < 3, 'lr_estimated_met_category'] = 1
    data.loc[(3 <= data['lr_estimated_met']) & (data['lr_estimated_met'] < 6), 'lr_estimated_met_category'] = 2
    data.loc[6 <= data['lr_estimated_met'], 'lr_estimated_met_category'] = 3

    target = data['target_met_category'].fillna(1)  #float64
    lr_estimated = data['lr_estimated_met_category'].fillna(1)  #.astype(np.int64)  #float64

    """
    Decision Tree
    """

    data.loc[(data['raw_wrist_sdvm'] <= 0.26) & (data['raw_wrist_mangle'] < -52), 'dt_estimated_met_category'] = 1
    data.loc[(data['raw_wrist_sdvm'] <= 0.26) & (data['raw_wrist_mangle'] >= -52), 'dt_estimated_met_category'] = 2
    data.loc[(0.26 < data['raw_wrist_sdvm']) & (data['raw_wrist_sdvm'] <= 0.79) & (
        data['raw_wrist_mangle'] > -53), 'dt_estimated_met_category'] = 2
    data.loc[(0.26 < data['raw_wrist_sdvm']) & (data['raw_wrist_sdvm'] <= 0.79) & (
        data['raw_wrist_mangle'] <= -53), 'dt_estimated_met_category'] = 3
    data.loc[data['raw_wrist_sdvm'] > 0.79, 'dt_estimated_met_category'] = 3

    dt_estimated = data['dt_estimated_met_category'].fillna(1)

    class_names = ['light', 'moderate', 'vigorous']

    # print(dt_estimated)
    # sys.exit(0)

    """
    Model evaluation statistics
    """
    print("Linear Regression Equation")

    # The mean squared error
    print("LR Mean squared error: %.2f"
          % np.mean((lr_estimated - target) ** 2))

    # The R squared score
    print("LR R squared score: %.2f" % r2_score(target, lr_estimated))

    # Precision and Recall for Linear Regression model
    precision, recall, fscore, support = precision_recall_fscore_support(target, lr_estimated, average='macro')
    print('LR overall precision: {}'.format(precision))
    print('LR overall recall: {}'.format(recall))
    print('LR overall fscore: {}'.format(fscore))
    print('LR overall support: {}'.format(support))

    precision, recall, fscore, support = precision_recall_fscore_support(target, lr_estimated)
    print('LR precision: {}'.format(precision))
    print('LR recall: {}'.format(recall))
    print('LR fscore: {}'.format(fscore))
    print('LR support: {}'.format(support))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target, lr_estimated)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=status + ' Linear Regression')

    print("\nDecision Tree")

    # The mean squared error
    print("DT Mean squared error: %.2f" % np.mean((dt_estimated - target) ** 2))

    # The R squared score
    print("DT R squared score: %.2f" % r2_score(target, dt_estimated))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target, dt_estimated)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(2)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=status + ' Decision Tree')

    # Precision and Recall for Decision Tree
    precision, recall, fscore, support = precision_recall_fscore_support(target, dt_estimated, average='macro')
    print('DT overall precision: {}'.format(precision))
    print('DT overall recall: {}'.format(recall))
    print('DT overall fscore: {}'.format(fscore))
    print('DT overall support: {}'.format(support))

    precision, recall, fscore, support = precision_recall_fscore_support(target, dt_estimated)
    print('DT precision: {}'.format(precision))
    print('DT recall: {}'.format(recall))
    print('DT fscore: {}'.format(fscore))
    print('DT support: {}'.format(support))

    plt.show()


print('Evaluation for non-filtered raw readings.')

data_nonf = pd.DataFrame()
non_filtered_files1 = [f for f in listdir(non_filtered_path1) if isfile(join(non_filtered_path1, f))]
for file in non_filtered_files1:
    data_nonf = data_nonf.append(pd.read_csv(non_filtered_path1 + file), ignore_index=True)

# non_filtered_files2 = [f for f in listdir(non_filtered_path2) if isfile(join(non_filtered_path2, f))]
# for file in non_filtered_files2:
#     data_nonf = data_nonf.append(pd.read_csv(non_filtered_path2 + file), ignore_index=True)

print('Completed reading data.')

evaluate_models(data_nonf, 'Non Filtered')

# print('Evaluation for filtered raw readings.')
#
# data_f = pd.DataFrame()
# filtered_files1 = [f for f in listdir(filtered_path1) if isfile(join(filtered_path1, f))]
# for file in filtered_files1:
#     data_f = data_f.append(pd.read_csv(filtered_path1 + file), ignore_index=True)
#
# filtered_files2 = [f for f in listdir(filtered_path2) if isfile(join(filtered_path2, f))]
# for file in filtered_files2:
#     data_f = data_f.append(pd.read_csv(filtered_path2 + file), ignore_index=True)
#
# evaluate_models(data_f, 'Filtered')
