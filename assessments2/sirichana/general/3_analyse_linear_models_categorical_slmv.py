import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import itertools, sys
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
from os import listdir
from os.path import isfile, join
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')
import statistical_extensions as SE


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


def evaluate_models(data, status, plot_title, output_folder_path, output_title, correlation_only=False):

    assessment_result = 'Assessment of ' + output_title + '\n\n'

    def transform_to_met_category(column, new_column):
        data.loc[data[column] <= 1.5, new_column] = 1
        data.loc[(1.5 < data[column]) & (data[column] < 3), new_column] = 2
        data.loc[3 <= data[column], new_column] = 3

    # Freedson EE MET values -> convert activity intensity to 3 levels
    transform_to_met_category('waist_ee', 'target_met_category_freedson_intensity')

    transform_to_met_category('predicted_ee', 'lr_estimated_met_category')

    target_met_category = data['target_met_category_freedson_intensity']
    lr_esitmated = data['lr_estimated_met_category']

    target_ee = data['waist_ee']
    predicted_ee = data['predicted_ee']

    # Pearson Correlation
    if correlation_only:
        corr, p_val = SE.GeneralStats.pearson_correlation(target_met_category, lr_esitmated)
        print('\n', output_folder_path, output_title)
        print('Categorical', corr, p_val)

        corr, p_val = SE.GeneralStats.pearson_correlation(target_ee, predicted_ee)
        print('MET', corr, p_val)
        return

    class_names = ['SB', 'LPA', 'MVPA']

    """
    Model evaluation statistics
    """
    # print("Evaluation of", status)

    # The mean squared error
    int_mse = "LR Mean squared error: %.2f" % np.mean((lr_esitmated - target_met_category) ** 2)
    # print(int_mse)
    assessment_result += int_mse + '\n\n'

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target_met_category, lr_esitmated)
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
    plt.figure(plot_title)
    conf_mat_output_filename = output_folder_path + output_title + '_confusion_matrix.png'
    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title=status, output_filename=conf_mat_output_filename)


if __name__ == '__main__':

    print('Start - Reading')

    experiments = ['LSM1', 'LSM2']
    week = 'Week 1'
    days = ['Wednesday', 'Thursday']
    # epochs = ['Epoch5', 'Epoch15', 'Epoch30', 'Epoch60']
    epochs = ['Epoch60']
    model_title = 'Sirichana Linear Regression'
    plot_number = 1

    # experiments = ['LSM1']
    # week = 'Week 1'
    # days = ['Wednesday']
    # epochs = ['Epoch60']
    # model_title = 'Sirichana Linear Regression'
    # plot_number = 1

    for epoch in epochs:

        output_title = model_title + '_' + epoch
        output_folder_path = ('E:\Data\Accelerometer_Results/Sirichana/').replace('\\', '/')

        start_reading = time.time()

        count = 0
        for experiment in experiments:
            for day in days:

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

                print('Completed', experiment, day)

        # result_B = results.copy()

        # print('\n\n')
        """LR A"""
        model_titleA = output_title + '_A'
        results = predict_ee_A(results)
        evaluate_models(results, model_titleA, plot_number+1, output_folder_path, model_titleA, correlation_only=False)
        results = SE.BlandAltman.clean_data_points(results)
        SE.BlandAltman.bland_altman_paired_plot_tested(results, model_titleA, plot_number+2, log_transformed=True,
                                                       min_count_regularise=False, output_filename=output_folder_path+model_titleA)

        results.drop('predicted_ee', axis=1, inplace=True)

        """LR B"""
        # # print('\n\n')
        # model_titleB = output_title + '_B'
        # result_B = predict_ee_B(result_B)
        # evaluate_models(result_B, model_titleB, plot_number+8, output_folder_path, model_titleB, correlation_only=False)
        # result_B = SE.BlandAltman.clean_data_points(result_B)
        # SE.BlandAltman.bland_altman_paired_plot_tested(result_B, model_titleB, plot_number+9, log_transformed=True,
        #                                                min_count_regularise=False, output_filename=output_folder_path+model_titleB)

        plot_number += 14

        end_reading = time.time()
        print('Completed', output_title, 'in', round(end_reading-start_reading, 2), '(s)')

    # plt.show()

    print('Completed.')
