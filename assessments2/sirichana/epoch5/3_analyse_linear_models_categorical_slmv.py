import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import itertools, sys
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
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


def predict_ee_A(data):

    def update_met_based_on_mv(row):
        met_value = data['predicted_ee'][row.name]
        if met_value > 6:
            met_value = (data['svm'][row.name] - 1708.1) / 373.4
        return met_value

    data['predicted_ee'] = (data['svm'] - 32.5) / 83.3
    data['predicted_ee'] = data.apply(update_met_based_on_mv, axis=1)

    return data


def predict_ee_B(data):

    data['predicted_ee'] = (data['svm'] + 12.7) / 105.3
    return data


def evaluate_models(data, status, plot_title):

    def transform_to_met_category(column, new_column):
        data.loc[data[column] < 1.5, new_column] = 1
        data.loc[(1.5 <= data[column]) & (data[column] < 3), new_column] = 2
        data.loc[3 <= data[column], new_column] = 3

    # Freedson EE MET values -> convert activity intensity to 3 levels
    transform_to_met_category('waist_ee', 'target_met_category_freedson_intensity')

    transform_to_met_category('predicted_ee', 'lr_estimated_met_category')

    target_met_category = data['target_met_category_freedson_intensity']
    lr_esitmated = data['lr_estimated_met_category']

    class_names = ['SB', 'LPA', 'MVPA']

    """
    Model evaluation statistics
    """
    print("Evaluation of", status)

    # The mean squared error
    print("LR Mean squared error: %.2f" % np.mean((lr_esitmated - target_met_category) ** 2))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target_met_category, lr_esitmated)
    np.set_printoptions(precision=2)

    stats = statistical_extensions.GeneralStats.evaluation_statistics(cnf_matrix)
    print('Accuracy', stats['accuracy'])
    print('Sensitivity', stats['sensitivity'])
    print('Specificity', stats['specificity'])

    # Plot non-normalized confusion matrix
    plt.figure(plot_title)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=status)


if __name__ == '__main__':

    print('Start - Reading')

    experiments = ['LSM1', 'LSM2']
    week = 'Week 1'
    days = ['Wednesday', 'Thursday']
    epoch = 'Epoch5'
    model_title = 'Sirichana Linear Regression (5 sec epoch)'

    start_reading = time.time()

    count = 0
    for experiment in experiments:
        if count != 0:
            break
        for day in days:
            if count != 0:
                break
            input_file_path = (
            "E:/Data/Accelerometer_Processed_Raw_Epoch_Data/" + experiment + "/" + week + "/" + day + "/" + epoch + "/").replace(
                '\\', '/')
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

    print('Reading time', round(end_reading - start_reading, 2), '(s)\nStart - Assessment')

    print('\n\n')
    """LR A"""
    model_title = 'Sirichana Linear Regression A (5 sec epoch)'
    results = predict_ee_A(results)
    evaluate_models(results, model_title, 1)
    results = statistical_extensions.BlandAltman.clean_data_points(results)
    statistical_extensions.BlandAltman.bland_altman_paired_plot_tested(results, model_title, 2, log_transformed=True, min_count_regularise=False)

    results.drop('predicted_ee', axis=1, inplace=True)

    """LR B"""
    print('\n\n')
    model_title = 'Sirichana Linear Regression B (5 sec epoch)'
    results = predict_ee_A(results)
    evaluate_models(results, model_title, 4)
    results = statistical_extensions.BlandAltman.clean_data_points(results)
    statistical_extensions.BlandAltman.bland_altman_paired_plot_tested(results, model_title, 5, log_transformed=True, min_count_regularise=False)

    plt.show()

    print('Completed.')
