import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import os
from os import listdir
from os.path import isfile, join
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')
import statistical_extensions as SE


def predict(data):
    # ENMO is acceleration in milig's
    data['predicted_ee'] = ((0.0320 * data['enmo'] * 1000) + 7.28) / 3.5
    return data


def evaluate_models(data, status, plot_title, output_folder_path, output_title, correlation_only=False):

    assessment_result = 'Assessment of ' + output_title + '\n\n'

    def met_to_intensity_waist_ee(row):
        ee = data['waist_ee'][row.name]
        return 1 if ee < 3 else 2
        # return 1 if ee <= 1.5 else 2 if ee < 3 else 3

    def met_to_intensity_lr_estimated_ee(row):
        ee = data['predicted_ee'][row.name]
        return 1 if ee < 3 else 2
        # return 1 if ee <= 1.5 else 2 if ee < 3 else 3

    # convert activity intensity to 3 levels - SB, LPA, MVPA
    data['target_met_category'] = data.apply(met_to_intensity_waist_ee, axis=1)
    data['lr_predicted_met_category'] = data.apply(met_to_intensity_lr_estimated_ee, axis=1)

    target_category = data['target_met_category']
    lr_estimated_category = data['lr_predicted_met_category']

    target_met = data['waist_ee']
    lr_estimated_met = data['predicted_ee']

    # Pearson Correlation
    if correlation_only:
        corr, p_val = SE.GeneralStats.pearson_correlation(target_category, lr_estimated_category)
        print('\n', output_folder_path, output_title)
        print('Categorical', corr, p_val)

        corr, p_val = SE.GeneralStats.pearson_correlation(target_met, lr_estimated_met)
        print('MET', corr, p_val)

        return

    class_names = ['SED_LPA', 'MVPA']
    # class_names = ['SED', 'LPA', 'MVPA']

    """
    Model evaluation statistics
    """
    # The mean squared error
    met_mse = ("LR Mean squared error [MET]: %.2f" % np.mean((lr_estimated_met - target_met) ** 2))
    int_mse = ("LR Mean squared error [Category]: %.2f" % np.mean((lr_estimated_category - target_category) ** 2))
    assessment_result += met_mse + '\n\n'
    assessment_result += int_mse + '\n\n'

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target_category, lr_estimated_category)
    np.set_printoptions(precision=2)

    stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)

    assessment_result += 'Classes' + '\t' + str(class_names) + '\t' + '\n'
    assessment_result += 'Accuracy' + '\t' + str(stats['accuracy']) + '\t' + str(stats['accuracy_ci']) + '\n'
    assessment_result += 'Sensitivity' + '\t' + str(stats['sensitivity']) + '\n'
    assessment_result += 'Sensitivity CI' + '\t' + str(stats['sensitivity_ci']) + '\n'
    assessment_result += 'Specificity' + '\t' + str(stats['specificity']) + '\n'
    assessment_result += 'Specificity CI' + '\t' + str(stats['specificity_ci']) + '\n'

    results_output_filename = output_folder_path + output_title + '_stat_assessment.txt'
    SE.Utils.print_assessment_results(results_output_filename, assessment_result)

    # Plot non-normalized confusion matrix
    plt.figure(plot_title)
    conf_mat_output_filename = output_folder_path + output_title + '_confusion_matrix.png'
    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title=status, output_filename=conf_mat_output_filename)


def evaluate_average_measures(data, epoch, output_title, output_folder_path):

    SE.Average_Stats.evaluate_average_measures_controller(data, epoch, output_title, output_folder_path, is_categorical=False)


if __name__ == '__main__':

    print('Start - Reading')

    experiments = ['LSM1', 'LSM2']
    week = 'Week 1'
    days = ['Wednesday', 'Thursday']
    epochs = ['Epoch60']
    model_title = 'Hilderbrand Linear Regression'
    plot_number = 1

    for epoch in epochs:

        output_title = model_title + '_' + epoch
        output_folder_path = ('E:\Data\Accelerometer_Results/Hilderbrand/').replace('\\', '/')

        start_reading = time.time()

        count = 0
        for experiment in experiments:
            for day in days:

                input_file_path = ("E:/Data/Accelerometer_Processed_Raw_Epoch_Data/only_hilderbrand/"+experiment+"/"+week+"/"+day+"/"+epoch+"/").replace('\\', '/')
                input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]

                for file in input_filenames:

                    # if file.split('_(2016')[0] != 'LSM219':
                    #     continue

                    dataframe = pd.read_csv(input_file_path + file)
                    dataframe['subject'] = file.split('_(2016')[0]

                    if count != 0:
                        results = results.append(dataframe, ignore_index=True)
                    else:
                        results = dataframe

                    count += 1

                    # if count > 20:
                    #     break

                print('Completed', experiment, day)

        """Prediction"""
        results = predict(results)

        results['incorrect_waist_ee'] = results['waist_ee']
        del results['waist_ee']

        """Update reference value"""
        results = SE.ReferenceMethod.update_reference_ee(results)

        # """Evaluate Average Measures"""
        evaluate_average_measures(results, epoch, output_title, output_folder_path)
        #
        # """General Assessment"""
        evaluate_models(results, output_title, plot_number+1, output_folder_path, output_title, correlation_only=False)

        """Bland Altman Plot"""
        results = SE.BlandAltman.clean_data_points(results)
        SE.BlandAltman.bland_altman_paired_plot_for_two_catagories(results, model_title, plot_number+2, log_transformed=True,
                                                       min_count_regularise=False, output_filename=output_folder_path+output_title)
        # SE.BlandAltman.bland_altman_paired_plot_tested(results, model_title, plot_number+2, log_transformed=True,
        #                                                min_count_regularise=False, output_filename=output_folder_path+output_title)

        # sys.exit(0)

        plot_number += 10

        end_reading = time.time()
        print('Completed', output_title, 'in', round(end_reading-start_reading, 2), '(s)')

    print('Assessment completed.')
