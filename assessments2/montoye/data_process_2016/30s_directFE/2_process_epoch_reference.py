from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time

summarise_duration = 3000

time_epoch = 30
multiplication_factor = int(60 / time_epoch)

input_detail_filename = "D:\Accelerometer Data\Processed/LSM2_ActiveTimeline_Details_v1.csv".replace('\\', '/')
input_details = pd.read_csv(input_detail_filename, usecols=[0, 1, 2, 3, 4, 7, 8, 9, 10])
input_details.columns = ['experiment', 'week', 'day', 'date', 'subject', 'epoch_start', 'epoch_end', 'row_start', 'row_end']


def convert_user(row):
    return input_details['subject'][row.name].split(' ')[0]
input_details['subject'] = input_details.apply(convert_user, axis=1)

output_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Epoch_30_processed/'.replace('\\', '/')
epoch_files_path = "D:\Accelerometer Data\ActilifeProcessedEpochs\Epoch60/LSM2\Week 1\Wednesday/".replace('\\', '/')
# epoch_files_path = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Epoch_30/'.replace('\\', '/')
epoch_files = [f for f in listdir(epoch_files_path) if isfile(join(epoch_files_path, f))]


def get_freedson_adult_vm3_intensity(row):
    cpm = epoch_data['waist_vm_60'][row.name]
    return 1 if cpm <= 2690 else 2 if cpm <= 6166 else 3 if cpm <= 9642 else 4


def get_freedson_vm3_combination_11_energy_expenditure(row):
    #  calculate Energy Expenditure using VM3 and a constant body mass (e.g., 80 kg)
    if epoch_data['waist_vm_cpm'][row.name] > 2453:
        met_value = (0.001064 * epoch_data['waist_vm_60'][row.name]) + (0.087512 * 80) - 5.500229
    else:
        met_value = epoch_data['waist_cpm'][row.name] * 0.0000191 * 80

    #  convert Energy Expenditure from Kcal/min to kJ/min
    met_value *= 4.184

    #  assuming that you use 80 kg as body mass, divide value by ((3.5/1000)*80*20.9)
    met_value /= ((3.5 / 1000) * 80 * 20.9)

    return met_value

for file in epoch_files:

    start_time = time.time()

    """
    Process Files
    """
    epoch_data = pd.read_csv(epoch_files_path+file, skiprows=10, usecols=[0, 1, 2], header=None)
    epoch_data.columns = ['AxisX', 'AxisY', 'AxisZ']
    epoch_data['waist_vm_60'] = np.sqrt([(epoch_data.AxisX ** 2) + (epoch_data.AxisY ** 2) + (epoch_data.AxisZ ** 2)])[0]
    epoch_data['waist_vm_cpm'] = epoch_data['waist_vm_60']
    epoch_data['waist_cpm'] = epoch_data.AxisY

    epoch_data['waist_ee'] = epoch_data.apply(get_freedson_vm3_combination_11_energy_expenditure, axis=1)
    epoch_data['waist_intensity'] = epoch_data.apply(get_freedson_adult_vm3_intensity, axis=1)

    epoch_data = epoch_data.reindex(np.repeat(epoch_data.index.values, multiplication_factor), method='ffill')

    """
    Clean Files
    """
    active_data = pd.DataFrame()

    rows = input_details[input_details['subject'].isin([file.split(' ')[0]]) & input_details['week'].isin(['Week 1'])]

    for index, row in rows.iterrows():
        start_epoch = int(row['row_start'] / summarise_duration)
        end_epoch = int(row['row_end'] / summarise_duration)
        active_data = active_data.append(epoch_data.iloc[start_epoch:end_epoch], ignore_index=True)

    active_data.to_csv((output_folder+file), sep=',')
    print('Processed', file.split(' ')[0], 'in', round(time.time()-start_time, 2), 'seconds')

print('Completed.')
