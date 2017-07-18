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


def evaluate_models(data, status='Sirichana - 15 epochs'):

    def update_met_based_on_mv(row):
        met_value = data['lr_A_estimated_met'][row.name]
        if met_value > 6:
            met_value = (data['svm'][row.name] - 1708.1) / 373.4
        return met_value

    # Freedson EE MET values -> convert activity intensity to 3 levels
    data.loc[data['actilife_waist_ee'] < 1.5, 'target_met_category_freedson_intensity'] = 1
    data.loc[((1.5 <= data['actilife_waist_ee']) & (data['actilife_waist_ee'] < 3)), 'target_met_category_freedson_intensity'] = 2
    data.loc[3 <= data['actilife_waist_ee'], 'target_met_category_freedson_intensity'] = 3

    """
    Linear Regression A
    """
    data['lr_A_estimated_met'] = (data['svm'] - 32.5) / 83.3
    data['lr_A_estimated_met'] = data.apply(update_met_based_on_mv, axis=1)

    data['lr_B_estimated_met'] = (data['svm'] + 12.7) / 105.3

    data.loc[data['lr_A_estimated_met'] < 1.5, 'lr_A_estimated_met_category'] = 1
    data.loc[(1.5 <= data['lr_A_estimated_met']) & (data['lr_A_estimated_met'] < 3), 'lr_A_estimated_met_category'] = 2
    data.loc[3 <= data['lr_A_estimated_met'], 'lr_A_estimated_met_category'] = 3

    data.loc[data['lr_B_estimated_met'] < 1.5, 'lr_B_estimated_met_category'] = 1
    data.loc[(1.5 <= data['lr_B_estimated_met']) & (data['lr_B_estimated_met'] < 3), 'lr_B_estimated_met_category'] = 2
    data.loc[3 <= data['lr_B_estimated_met'], 'lr_B_estimated_met_category'] = 3

    target_met_category = data['target_met_category_freedson_intensity']
    lr_A_esitmated = data['lr_A_estimated_met_category']
    lr_B_esitmated = data['lr_B_estimated_met_category']

    class_names = ['SB', 'LPA', 'MVPA']

    """
    Model evaluation statistics
    """
    print("Linear Regression A")

    # The mean squared error
    print("LR A Mean squared error: %.2f" % np.mean((lr_A_esitmated - target_met_category) ** 2))

    # The R squared score
    print("LR A R squared score: %.2f" % r2_score(target_met_category, lr_A_esitmated))

    # Precision and Recall for Linear Regression model
    precision, recall, fscore, support = precision_recall_fscore_support(target_met_category, lr_A_esitmated, average='macro')
    print('LR A overall precision: {}'.format(precision))
    print('LR A overall recall: {}'.format(recall))
    print('LR A overall fscore: {}'.format(fscore))
    print('LR A overall support: {}'.format(support))

    precision, recall, fscore, support = precision_recall_fscore_support(target_met_category, lr_A_esitmated)
    print('LR A precision: {}'.format(precision))
    print('LR A recall: {}'.format(recall))
    print('LR A fscore: {}'.format(fscore))
    print('LR A support: {}'.format(support))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target_met_category, lr_A_esitmated)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=status + ' Linear Regression - A')


    print("Linear Regression B")

    print("LR B Mean squared error: %.2f" % np.mean((lr_B_esitmated - target_met_category) ** 2))
    print("LR B R squared score: %.2f" % r2_score(target_met_category, lr_B_esitmated))

    # Precision and Recall for Linear Regression model
    precision, recall, fscore, support = precision_recall_fscore_support(target_met_category, lr_B_esitmated, average='macro')
    print('LR B overall precision: {}'.format(precision))
    print('LR B overall recall: {}'.format(recall))
    print('LR B overall fscore: {}'.format(fscore))
    print('LR B overall support: {}'.format(support))

    precision, recall, fscore, support = precision_recall_fscore_support(target_met_category, lr_B_esitmated)
    print('LR B precision: {}'.format(precision))
    print('LR B recall: {}'.format(recall))
    print('LR B fscore: {}'.format(fscore))
    print('LR B support: {}'.format(support))

    # Compute confusion matrix
    cnf_matrixB = confusion_matrix(target_met_category, lr_B_esitmated)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(2)
    plot_confusion_matrix(cnf_matrixB, classes=class_names, title=status + ' Linear Regression - B')

    plt.show()


if __name__ == '__main__':

    experiment = 'LSM2'
    week = 'Week 1'
    day1 = 'Wednesday'

    prediction_path = "D:/Accelerometer Data/Processed/" + experiment + "/" + week + "/" + day1 + "/filtered\Epoch15/".replace(
        '\\', '/')

    prediction_files = [f for f in listdir(prediction_path) if isfile(join(prediction_path, f))]

    predictions = pd.DataFrame()
    for file in prediction_files:
        predictions = predictions.append(pd.read_csv(prediction_path + file))

    predictions.index = np.arange(0, len(predictions))

    evaluate_models(predictions)
