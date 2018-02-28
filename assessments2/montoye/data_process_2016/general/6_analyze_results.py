import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import time
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')
import statistical_extensions as SE


def evaluate_models(data, status, plot_number, output_folder_path, output_title, correlation_only=False):

    assessment_result = 'Assessment of ' + output_title + '\n\n'

    target_intensity = data['actual_category']
    predicted_intensity = data['predicted_category']

    # Pearson Correlation
    if correlation_only:
        corr, p_val = SE.GeneralStats.pearson_correlation(target_intensity, predicted_intensity)
        print('\n', output_folder_path, output_title)
        print('Categorical', corr, p_val)

        return

    class_names = ['SB', 'LPA', 'MVPA']

    # The mean squared error
    int_mse = "Montoye 2016 ANN (Instensity) Mean squared error: %.2f" % np.mean(
        (predicted_intensity - target_intensity) ** 2)
    # print(int_mse)
    assessment_result += int_mse + '\n\n'

    # Compute confusion matrix for intensity
    cnf_matrix = confusion_matrix(target_intensity, predicted_intensity)
    np.set_printoptions(precision=2)

    stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)
    # print('Accuracy', stats['accuracy'])
    # print('Sensitivity', stats['sensitivity'])
    # print('Specificity', stats['specificity'])

    assessment_result += 'Classes' + '\t' + str(class_names) + '\t' + '\n'
    assessment_result += 'Accuracy' + '\t' + str(stats['accuracy']) + '\t' + str(stats['accuracy_ci']) + '\n'
    assessment_result += 'Sensitivity' + '\t' + str(stats['sensitivity']) + '\n'
    assessment_result += 'Sensitivity CI' + '\t' + str(stats['sensitivity_ci']) + '\n'
    assessment_result += 'Specificity' + '\t' + str(stats['specificity']) + '\n'
    assessment_result += 'Specificity CI' + '\t' + str(stats['specificity_ci']) + '\n'

    results_output_filename = output_folder_path + output_title + '_stat_assessment.txt'
    SE.Utils.print_assessment_results(results_output_filename, assessment_result)

    # Plot non-normalized confusion matrix
    plt.figure(plot_number)
    conf_mat_output_filename = output_folder_path + output_title + '_confusion_matrix.png'
    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title=status, output_filename=conf_mat_output_filename)


def evaluate_average_measures(data, epoch, output_title):
    SE.Average_Stats.evaluate_average_measures_for_categorical(data, epoch, output_title, output_folder_path)


if __name__ == '__main__':

    print('Start - Reading')

    experiments = ['LSM1', 'LSM2']
    week = 'Week 1'
    days = ['Wednesday', 'Thursday']
    wrists = ['left_wrist', 'right_wrist']
    # epochs = ['Epoch5', 'Epoch15', 'Epoch30', 'Epoch60']
    epochs = ['Epoch30']
    model_title = 'Montoye ANN 2016'
    plot_number = 1

    for epoch in epochs:
        for wrist in wrists:

            start_reading = time.time()

            output_title = model_title + '_' + wrist + '_' + epoch
            output_folder_path = ('E:\Data\Accelerometer_Results\Montoye/2016/').replace('\\', '/')

            input_file_path = ("E:\Data\Accelerometer_Montoye_ANN/2016/"+wrist+"/"+epoch+"/combined/").replace('\\', '/')
            input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]

            count = 0
            for file in input_filenames:
                dataframe = pd.read_csv(input_file_path + file)
                dataframe['subject'] = file.split('_(2016')[0]

                if count != 0:
                    results = results.append(dataframe, ignore_index=True)
                else:
                    results = dataframe

                count += 1

            """Evaluate Average Measures"""
            evaluate_average_measures(results, epoch, output_title)

            """General Assessment"""
            evaluate_models(results, model_title, plot_number, output_folder_path, output_title, correlation_only=False)

            plot_number += 1

            end_reading = time.time()
            print('Completed', output_title, 'in', round(end_reading-start_reading, 2), '(s)')

    print('Assessment completed.')
