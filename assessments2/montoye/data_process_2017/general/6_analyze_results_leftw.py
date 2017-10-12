import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join
import time
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')
import statistical_extensions as SE


def evaluate_models(data, status, plot_number, output_folder_path, output_title, correlation_only=False):

    assessment_result = 'Assessment of ' + output_title + '\n\n'

    target_intensity = data['actual_category']
    predicted_intensity = data['predicted_category']

    target_ee = data['waist_ee']
    predicted_ee = data['predicted_ee']

    # Pearson Correlation
    if correlation_only:
        # Pearson Correlation
        corr, p_val = SE.GeneralStats.pearson_correlation(target_intensity, predicted_intensity)
        print('\n', output_folder_path, output_title)
        print('Categorical', corr, p_val)

        corr, p_val = SE.GeneralStats.pearson_correlation(target_ee, predicted_ee)
        print('MET', corr, p_val)

        return

    class_names = ['SED', 'LPA', 'MVPA']

    # The mean squared error
    met_mse = "Montoye 2017 ANN (MET) Mean squared error: %.2f" % np.mean((predicted_ee - target_ee) ** 2)
    int_mse = "Montoye 2017 ANN (Instensity) Mean squared error: %.2f" % np.mean(
        (predicted_intensity - target_intensity) ** 2)
    # print(met_mse)
    # print(int_mse)
    assessment_result += met_mse + '\n\n'
    assessment_result += int_mse + '\n\n'

    # Compute confusion matrix for intensity
    cnf_matrix = confusion_matrix(target_intensity, predicted_intensity)
    np.set_printoptions(precision=2)

    stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)
    # print('Accuracy', stats['accuracy'])
    # print('Sensitivity', stats['sensitivity'])
    # print('Specificity', stats['specificity'])

    assessment_result += 'Classes' + '\t' + str(class_names) + '\t' + '\n'
    assessment_result += 'Accuracy' + '\t' + str(stats['accuracy']) + '\n'
    assessment_result += 'Sensitivity' + '\t' + str(stats['sensitivity']) + '\n'
    assessment_result += 'Specificity' + '\t' + str(stats['specificity']) + '\n'

    results_output_filename = output_folder_path + output_title + '_stat_assessment.txt'
    SE.Utils.print_assessment_results(results_output_filename, assessment_result)

    # Plot non-normalized confusion matrix
    plt.figure(plot_number)
    conf_mat_output_filename = output_folder_path + output_title + '_confusion_matrix.png'
    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title=status, output_filename=conf_mat_output_filename)


if __name__ == '__main__':

    print('Start - Reading')

    experiments = ['LSM1', 'LSM2']
    week = 'Week 1'
    days = ['Wednesday', 'Thursday']
    wrists = ['left_wrist', 'right_wrist']
    epochs = ['Epoch5', 'Epoch15', 'Epoch30', 'Epoch60']
    models = ['v2', 'v1v2']
    model_title = 'Montoye ANN 2017 '
    plot_number = 1

    for model in models:
        for epoch in epochs:
            for wrist in wrists:

                start_reading = time.time()

                output_title = model_title + '_' + wrist + '_' + epoch + '_' + model
                output_folder_path = ('E:\Data\Accelerometer_Results\Montoye/2017/').replace('\\', '/')
                input_file_path = ("E:\Data\Accelerometer_Montoye_ANN/2017/"+wrist+"/"+epoch+"/"+model+"/combined/").replace('\\', '/')
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

                """General Assessment"""
                evaluate_models(results, model_title, plot_number, output_folder_path, output_title, correlation_only=True)

                """Bland Altman Plot"""
                # results = SE.BlandAltman.clean_data_points(results)
                # SE.BlandAltman.bland_altman_paired_plot_tested(results, model_title, plot_number+1, log_transformed=True,
                #                                                min_count_regularise=True, output_filename=output_folder_path+output_title)

                plot_number += 4

                end_reading = time.time()
                print('Completed', output_title, 'in', round(end_reading-start_reading, 2), '(s)')

    # plt.show()
    print('Assessment completed.')
