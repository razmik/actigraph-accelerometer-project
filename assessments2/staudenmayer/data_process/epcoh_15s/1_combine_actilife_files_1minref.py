from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time

process_start_time = time.time()

time_epoch = 15
multiplication_factor = 60 / time_epoch

experiment = 'LSM2'
week = 'Week 1'
day = 'Wednesday'

hip_key = 'Waist'
epoch_start_rows = 10
path = ("D:\Accelerometer Data\ActilifeProcessedEpochs\Epoch60/" + experiment + "/" + week + "/" + day + "").replace('\\', '/')
output_folder = ("D:\Accelerometer Data\ActilifeProcessedEpochs\Epoch15/" + experiment + "/" + week + "/" + day + "/processed_1min_ref/").replace('\\', '/')

path_components = path.split('/')
output_prefix = (path_components[4] + '_' + path_components[5] + '_' + path_components[6]).replace(' ', '_')


def get_freedson_adult_vm3_intensity(row):
    cpm = hip_epoch_data['waist_vm_60'][row.name]
    return 1 if cpm <= 2690 else 2 if cpm <= 6166 else 3 if cpm <= 9642 else 4


def get_freedson_vm3_combination_11_energy_expenditure(row):
    # Step 1: calculate Energy Expenditure using VM3 and a constant body mass (e.g., 80 kg)
    if hip_epoch_data['waist_vm_cpm'][row.name] > 2453:
        met_value = (0.001064 * hip_epoch_data['waist_vm_60'][row.name]) + (0.087512 * 80) - 5.500229
    else:
        met_value = hip_epoch_data['waist_cpm'][row.name] * 0.0000191 * 80

    # Step 2: convert Energy Expenditure from Kcal/min to kJ/min
    met_value *= 4.184

    # Step 2: assuming that you use 80 kg as body mass, divide value by ((3.5/1000)*80*20.9)
    met_value /= ((3.5 / 1000) * 80 * 20.9)

    return met_value


print("Reading files and creating dictionary.")
files = [f for f in listdir(path) if isfile(join(path, f))]
file_dictionary = {}
for file in files:

    # Filename example: LSM203 Waist (2016-11-02)60sec.csv
    file_components = file.split(' ')
    key = file_components[0]

    if key not in file_dictionary:
        file_dictionary[key] = {file_components[1]: file}
    else:
        file_dictionary[key][file_components[1]] = file

print("Processing data of total", len(file_dictionary))
i = 1
for participant in file_dictionary:
    itr_start_time = time.time()
    hip_file = path + '/' + file_dictionary[participant][hip_key]

    """
    Calculate Waist (hip) epoch values and reference parameters
    """
    hip_epoch_data = pd.read_csv(hip_file, skiprows=epoch_start_rows, usecols=[0, 1, 2])
    hip_epoch_data.columns = ['AxisX', 'AxisY', 'AxisZ']
    hip_epoch_data['waist_vm_60'] = np.sqrt([(hip_epoch_data.AxisX ** 2) + (hip_epoch_data.AxisY ** 2) + (hip_epoch_data.AxisZ ** 2)])[0]
    hip_epoch_data['waist_vm_cpm'] = hip_epoch_data['waist_vm_60']
    hip_epoch_data['waist_cpm'] = hip_epoch_data.AxisY

    hip_epoch_data['waist_ee'] = hip_epoch_data.apply(get_freedson_vm3_combination_11_energy_expenditure, axis=1)
    hip_epoch_data['waist_intensity'] = hip_epoch_data.apply(get_freedson_adult_vm3_intensity, axis=1)

    hip_epoch_data = hip_epoch_data.reindex(np.repeat(hip_epoch_data.index.values, multiplication_factor), method='ffill')

    itr_end_time = time.time()

    # save output file
    # Filename example: LSM203 Waist (2016-11-02)15sec.csv
    output_filename = file_dictionary[participant][hip_key].split(' ')
    file_date = output_filename[2].split('60sec')
    output_filename = output_folder + '/' + participant + '_' + output_prefix + '_' + file_date[0] + str(time_epoch) + 's' + '.csv'
    hip_epoch_data.to_csv(output_filename, sep=',', index=None)
    print("Process completion " + str(i) + " / " + str(len(file_dictionary)) + " with duration " + str(
        round(itr_end_time - itr_start_time, 2)) + " seconds.")
    i += 1

process_end_time = time.time()
print("Completed with duration " + str(round((process_end_time - process_start_time) / 60, 2)) + " minutes.")