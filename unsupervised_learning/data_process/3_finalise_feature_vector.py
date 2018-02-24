import importlib
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


experiments = ['LSM1', 'LSM2']
weeks = ['Week 1', 'Week 2']
days = ['Wednesday', 'Thursday']
time_epoch_dictionary = {
    'Epoch5': 500,
    'Epoch15': 1500,
    'Epoch30': 3000,
    'Epoch60': 6000
}


if __name__ == '__main__':

    root_folder_name = 'E:\Data\Accelerometer_Processed_Raw_Epoch_Data_Unsupervised/'.replace('\\', '/')
    output_folder_name = 'E:\Data\Accelerometer_Processed_Raw_Epoch_Data_Unsupervised\outputs/'.replace('\\', '/')

    for experiment in experiments:
        for week in weeks:
            for day in days:
                for epoch, value in time_epoch_dictionary.items():

                    folder_name = root_folder_name + experiment + '/' + week + '/' + day + '/' + epoch + '/'

                    files = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]

                    print(folder_name)
                    result_df = pd.DataFrame()
                    temp_filename = ''
                    for file in files:

                        if file.split('_LSM110')[0] != 'Wrist':
                            continue

                        df = pd.read_csv(folder_name + file)

                        if len(result_df) == 0:
                            temp_filename = file
                            result_df = df
                        else:
                            result_df = result_df.append(df, ignore_index=True)

                    if len(result_df) != 0:
                        folder_name = output_folder_name #+ '/' + week + '/' + day + '/' + epoch + '/'
                        result_df.to_csv(folder_name + temp_filename)