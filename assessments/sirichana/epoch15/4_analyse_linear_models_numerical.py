import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools, sys
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
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

    print('Assessment for', status)

    def update_met_based_on_mv(row):
        met_value = data['lr_A_estimated_met'][row.name]
        if met_value > 6:
            met_value = (data['svm'][row.name] - 1708.1) / 373.4
        return met_value

    """
    Linear Regression A
    """
    data['lr_A_estimated_met'] = (data['svm'] - 32.5) / 83.3
    data['lr_A_estimated_met'] = data.apply(update_met_based_on_mv, axis=1)

    """
    Linear Regression B
    """
    data['lr_B_estimated_met'] = (data['svm'] + 12.7) / 105.3

    """
    Assessment data
    """
    target_ee = data['actilife_waist_ee']
    lr_A_esitmated = data['lr_A_estimated_met']
    lr_B_esitmated = data['lr_B_estimated_met']

    """
    Model evaluation statistics
    """
    print("Linear Regression A")

    # The mean squared error
    print("LR A Mean squared error: %.2f" % np.mean((lr_A_esitmated - target_ee) ** 2))
    print("LR A R squared score: %.2f" % r2_score(target_ee, lr_A_esitmated))

    print("Linear Regression B")

    print("LR B Mean squared error: %.2f" % np.mean((lr_B_esitmated - target_ee) ** 2))
    print("LR B R squared score: %.2f" % r2_score(target_ee, lr_B_esitmated))


if __name__ == '__main__':

    experiment = 'LSM2'
    week = 'Week 1'
    day1 = 'Wednesday'
    epoch = 'Epoch15'

    prediction_path = ("D:/Accelerometer Data/Processed/" + experiment + "/" + week + "/" + day1 + "/filtered/"+epoch+"/").replace(
        '\\', '/')

    prediction_files = [f for f in listdir(prediction_path) if isfile(join(prediction_path, f))]

    predictions = pd.DataFrame()
    for file in prediction_files:
        predictions = predictions.append(pd.read_csv(prediction_path + file))

    predictions.index = np.arange(0, len(predictions))

    evaluate_models(predictions, 'Sirichana - '+epoch)
