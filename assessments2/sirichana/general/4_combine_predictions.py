import sys
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')


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


def transform_to_met_category(data, column, new_column):
    data.loc[data[column] <= 1.5, new_column] = 1
    data.loc[(1.5 < data[column]) & (data[column] < 3), new_column] = 2
    data.loc[3 <= data[column], new_column] = 3


if __name__ == '__main__':

    print('Start - Reading')

    experiments = ['LSM1', 'LSM2']
    week = 'Week 1'
    days = ['Wednesday', 'Thursday']
    epochs = ['Epoch5', 'Epoch15', 'Epoch30', 'Epoch60']
    model_title = 'Sirichana Linear Regression'

    for epoch in epochs:

        start_reading = time.time()

        for experiment in experiments:
            for day in days:

                output_folder = ("E:\Data\Accelerometer_LR\sirichana/").replace('\\', '/')
                input_file_path = ("E:/Data/Accelerometer_Processed_Raw_Epoch_Data/"+experiment+"/"+week+"/"+day+"/"+epoch+"/").replace('\\', '/')
                input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]

                print('processing', input_file_path)

                for file in input_filenames:
                    results = pd.read_csv(input_file_path + file)

                    results_A = predict_ee_A(results.copy())
                    transform_to_met_category(results_A, 'waist_ee', 'actual_category')
                    transform_to_met_category(results_A, 'predicted_ee', 'predicted_category')

                    # save output file
                    output_filename = output_folder + 'LRA/' + epoch + '/' + file
                    results_A.to_csv(output_filename, sep=',', index=None)

                    results_B = predict_ee_B(results.copy())
                    transform_to_met_category(results_B, 'waist_ee', 'actual_category')
                    transform_to_met_category(results_B, 'predicted_ee', 'predicted_category')

                    # save output file
                    output_filename = output_folder + 'LRB/' + epoch + '/' + file
                    results_B.to_csv(output_filename, sep=',', index=None)

                print('Completed', experiment, day)

        end_reading = time.time()
        print('Completed', epoch, 'in', round(end_reading-start_reading, 2), '(s)')

    print('Assessment completed.')
