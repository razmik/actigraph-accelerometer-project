import sys
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')
import statistical_extensions as SE


if __name__ == '__main__':

    print('Start - Reading')

    experiments = ['LSM1', 'LSM2']
    week = 'Week 1'
    days = ['Wednesday', 'Thursday']
    # epochs = ['Epoch5', 'Epoch15', 'Epoch30', 'Epoch60']
    epochs = ['Epoch15']
    output_folder_root = ('E:/Data/Accelerometer_Processed_Raw_Epoch_Data_updated/').replace('\\', '/')

    for epoch in epochs:

        count = 0
        for experiment in experiments:
            for day in days:

                # Staudenmayer and Sirichana
                # input_file_path = ("E:/Data/Accelerometer_Processed_Raw_Epoch_Data/"+experiment+"/"+week+"/"+day+"/"+epoch+"/").replace('\\', '/')
                # output_folder_path = os.path.join(output_folder_root, experiment + "/" + week + "/" + day + "/" + epoch + "/")


                # Hilderbrand
                input_file_path = ("E:/Data/Accelerometer_Processed_Raw_Epoch_Data/only_hilderbrand/" + experiment + "/" + week + "/" + day + "/" + epoch + "/").replace('\\', '/')
                output_folder_path = os.path.join(output_folder_root, "only_hilderbrand/" + experiment + "/" + week + "/" + day + "/" + epoch + "/")

                input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]

                for file in input_filenames:

                    if file.split('_(2016')[0] != 'LSM219':
                        continue

                    dataframe = pd.read_csv(os.path.join(input_file_path, file))

                    dataframe = SE.ReferenceMethod.update_reference_ee(dataframe)

                    dataframe.to_csv(os.path.join(output_folder_path, file), index=None)

                    count += 1

                    # if count > 10:
                    #     break

                print('Completed', experiment, day)

    print('Correction completed.')
