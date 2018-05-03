import sys
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')


def predict(data):
    data['predicted_ee'] = 1.89378 + (5.50821 * data['raw_wrist_sdvm']) - (0.02705 * data['raw_wrist_mangle'])
    return data


def met_to_intensity_waist_ee(row):
    ee = results['waist_ee'][row.name]
    return 1 if ee <= 1.5 else 2 if ee < 3 else 3


def met_to_intensity_lr_estimated_ee(row):
    ee = results['predicted_ee'][row.name]
    return 1 if ee <= 1.5 else 2 if ee < 3 else 3


if __name__ == '__main__':

    print('Start - Reading')

    experiments = ['LSM1', 'LSM2']
    week = 'Week 1'
    days = ['Wednesday', 'Thursday']
    epochs = ['Epoch15', 'Epoch60']
    model_title = 'Staudenmayer Linear Regression'

    for epoch in epochs:

        start_reading = time.time()

        for experiment in experiments:
            for day in days:

                output_folder = ("E:\Data\Accelerometer_LR\staudenmayer/").replace('\\', '/')
                input_file_path = ("E:/Data/Accelerometer_Processed_Raw_Epoch_Data/"+experiment+"/"+week+"/"+day+"/"+epoch+"/").replace('\\', '/')
                input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]

                for file in input_filenames:
                    results = pd.read_csv(input_file_path + file)
                    results = predict(results)

                    # convert activity intensity to 3 levels - SB, LPA, MVPA
                    results['actual_category'] = results.apply(met_to_intensity_waist_ee, axis=1)
                    results['predicted_category'] = results.apply(met_to_intensity_lr_estimated_ee, axis=1)

                    # save output file
                    output_filename = output_folder + epoch + '/' + file
                    results.to_csv(output_filename, sep=',', index=None)

                print('Completed', experiment, day)

        end_reading = time.time()
        print('Completed', epoch, 'in', round(end_reading-start_reading, 2), '(s)')

    print('Assessment completed.')
