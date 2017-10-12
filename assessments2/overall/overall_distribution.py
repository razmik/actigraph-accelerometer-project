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


if __name__ == '__main__':

    print('Start - Reading')

    experiments = ['LSM1', 'LSM2']
    week = 'Week 1'
    days = ['Wednesday', 'Thursday']
    epochs = ['Epoch5', 'Epoch15', 'Epoch30', 'Epoch60']
    model_title = 'Overall distribution'

    for epoch in epochs:

        output_title = model_title + '_' + epoch
        output_folder_path = ('E:\Data\Accelerometer_Results/Montoye/2017/').replace('\\', '/')

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

                    if count == 20: break

                    if count != 0:
                        results = results.append(dataframe, ignore_index=True)
                    else:
                        results = dataframe

                    count += 1

                print('Completed', experiment, day)

        print('Epoch', epoch)
        print('SB', len(dataframe.loc[dataframe['waist_ee'] <= 1.5]))
        print('LPA', len(dataframe.loc[(dataframe['waist_ee'] > 1.5) & (dataframe['waist_ee'] < 3)]))
        print('MVPA', len(dataframe.loc[dataframe['waist_ee'] > 3]))
        print('\n')

    print('Completed.')
