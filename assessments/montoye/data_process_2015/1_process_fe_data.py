from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time
import matplotlib.pyplot as plt

summarise_duration = 3000

input_detail_filename = "D:\Accelerometer Data\Processed/LSM2_ActiveTimeline_Details_v1.csv".replace('\\', '/')
input_details = pd.read_csv(input_detail_filename, usecols=[0, 1, 2, 3, 4, 7, 8, 9, 10])
input_details.columns = ['experiment', 'week', 'day', 'date', 'subject', 'epoch_start', 'epoch_end', 'row_start', 'row_end']


def convert_user(row):
    return input_details['subject'][row.name].split(' ')[0]
input_details['subject'] = input_details.apply(convert_user, axis=1)

output_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\FE_30_processed/'.replace('\\', '/')
epoch_files_path = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\FE_30/'.replace('\\', '/')
epoch_files = [f for f in listdir(epoch_files_path) if isfile(join(epoch_files_path, f))]

for file in epoch_files:

    start_time = time.time()

    """
    Process Files
    """
    fe_data = pd.read_csv(epoch_files_path+file, skiprows=2)
    fe_data.columns = ['Time', 'XMean', 'XSD', 'X10Perc', 'X25Perc', 'X50Perc', 'X75Perc', 'X90Perc',
                          'YMean', 'YSD', 'Y10Perc', 'Y25Perc', 'Y50Perc', 'Y75Perc', 'Y90Perc',
                          'ZMean', 'ZSD', 'Z10Perc', 'Z25Perc', 'Z50Perc', 'Z75Perc', 'Z90Perc', 'Unnamed']
    del fe_data['Unnamed']
    fe_data['XVar'] = fe_data['XSD'] ** 2
    fe_data['YVar'] = fe_data['YSD'] ** 2
    fe_data['ZVar'] = fe_data['ZSD'] ** 2

    """
    Clean Files
    """
    active_fe_data = pd.DataFrame()
    # Filename example: LSM203 Waist (2016-11-02)_FE.csv
    rows = input_details[input_details['subject'].isin([file.split(' ')[0]]) & input_details['week'].isin(['Week 1'])]
    for index, row in rows.iterrows():

        starting_row = row['row_start']
        end_row = row['row_end']
        if end_row > starting_row > -1 and end_row > -1:
            start_epoch = int(starting_row / summarise_duration)
            end_epoch = int(end_row / summarise_duration)
            active_fe_data = active_fe_data.append(fe_data.iloc[start_epoch:end_epoch], ignore_index=True)

    active_fe_data.to_csv((output_folder+file), sep=',')
    print('Processed', file.split(' ')[0], 'in', round(time.time()-start_time, 2), 'seconds')

print('Completed.')
