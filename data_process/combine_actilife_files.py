"""
1. Read all files in the folder
2. Get participant IDs into an array
3. For each array,
    a. Get the waist file,
        calculate VM 15s,
        VM 60a (CPM),
        cut point intensity with Freedson Adult VM3 (2011), https://actigraph.desk.com/customer/en/portal/articles/2515803-what-s-the-difference-among-the-cut-points-available-in-actilife-
        energy expenditure (EE) with Freedson Treadmill Adult (1998), https://actigraph.desk.com/customer/en/portal/articles/2515804-what-is-the-difference-among-the-met-algorithms-
    b. Get the wrist file,
        calculate VM 15s,
        VM 60a (CPM),
        combine the waist cut point intensity as reference,
        combine the waist EE MET level as reference
4. Save the file as csv in a new folder (processed)
"""
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time

process_start_time = time.time()

wrist_key = 'Wrist'
hip_key = 'Waist'
epoch_start_rows = 10
path = "D:/Accelerometer Data/ActilifeProcessedEpochs/LSM2/Week 1/Thursday".replace('\\', '/')
output_folder = "D:/Accelerometer Data/ActilifeProcessedEpochs/LSM2/Week 1/Thursday/processed".replace('\\', '/')

path_components = path.split('/')
output_prefix = (path_components[3] + '_' + path_components[4] + '_' + path_components[5]).replace(' ', '_')


def get_freedson_adult_vm3_intensity(row):
    cpm = hip_epoch_data['waist_vm_60'][row.name]
    return 1 if cpm <= 2690 else 2 if cpm <= 6166 else 3 if cpm <= 9642 else 4


def get_freedson_vm3_combination_11_energy_expenditure(row):
    # https://actigraph.desk.com/customer/en/portal/articles/2515835-what-is-the-difference-among-the-energy-expenditure-algorithms-
    cpm = hip_epoch_data['waist_vm_60'][row.name]
    return 1 if cpm <= 2690 else 2 if cpm <= 6166 else 3 if cpm <= 9642 else 4


def get_waist_equivalent_wrist_counts(row):
    # https://actigraph.desk.com/customer/en/portal/articles/2515826-what-does-the-%22worn-on-wrist%22-option-do-in-the-data-scoring-tab-
    wrist_cpm = wrist_epoch_data['wrist_vm_60'][row.name]

    if wrist_cpm <= 644:
        waist_eq_cpm = 0.5341614 * wrist_cpm
    elif wrist_cpm <= 1272:
        waist_eq_cpm = 1.7133758 * wrist_cpm - 759.414013
    elif wrist_cpm <= 3806:
        waist_eq_cpm = 0.3997632 * wrist_cpm + 911.501184
    else:
        waist_eq_cpm = 0.0128995 * wrist_cpm + 2383.904505

    return waist_eq_cpm

"""
Read all files and add them into dictionary
{
    'LSM203': {
                'Waist': 'LSM203 Waist (2016-11-02)15sec.csv', 
                'Wrist': 'LSM203 Wrist (2016-11-02)15sec.csv'
           }, 
    'LSM204': {
                'Waist': 'LSM204 Waist (2016-11-02)15sec.csv', 
                'Wrist': 'LSM204 Wrist (2016-11-02)15sec.csv'
            }
}
"""
print("Reading files and creating dictionary.")
files = [f for f in listdir(path) if isfile(join(path, f))]
file_dictionary = {}
for file in files:

    # Filename example: LSM203 Waist (2016-11-02)15sec.csv
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
    wrist_file = path + '/' + file_dictionary[participant][wrist_key]
    hip_file = path + '/' + file_dictionary[participant][hip_key]

    hip_epoch_data = pd.read_csv(hip_file, skiprows=epoch_start_rows, usecols=[0, 1, 2])
    hip_epoch_data.columns = ['waistX', 'waistY', 'waistZ']
    hip_epoch_data['waist_vm_15'] = np.sqrt([(hip_epoch_data.waistX ** 2) + (hip_epoch_data.waistY ** 2) + (hip_epoch_data.waistZ ** 2)])[0]
    hip_epoch_data['waist_vm_60'] = hip_epoch_data['waist_vm_15'] * 4
    hip_epoch_data['waist_intensity'] = hip_epoch_data.apply(get_freedson_adult_vm3_intensity, axis=1)
    # hip_epoch_data['waist_ee'] = hip_epoch_data.apply(get_freedson_vm3_combination_11_energy_expenditure, axis=1)

    wrist_epoch_data = pd.read_csv(wrist_file, skiprows=epoch_start_rows, usecols=[0, 1, 2])
    wrist_epoch_data.columns = ['wristX', 'wristY', 'wristZ']
    wrist_epoch_data['wrist_vm_15'] = np.sqrt([(wrist_epoch_data.wristX ** 2) + (wrist_epoch_data.wristY ** 2) + (wrist_epoch_data.wristZ ** 2)])[0]
    wrist_epoch_data['wrist_vm_60'] = wrist_epoch_data['wrist_vm_15'] * 4
    wrist_epoch_data['waist_eq_wrist_vm_60'] = wrist_epoch_data.apply(get_waist_equivalent_wrist_counts, axis=1)

    del wrist_epoch_data['wristX']
    del wrist_epoch_data['wristY']
    del wrist_epoch_data['wristZ']

    wrist_epoch_data['waist_vm_60'] = hip_epoch_data['waist_vm_60']
    wrist_epoch_data['waist_intensity'] = hip_epoch_data['waist_intensity']

    itr_end_time = time.time()

    # save output file
    # Filename example: LSM203 Waist (2016-11-02)15sec.csv
    output_filename = file_dictionary[participant][wrist_key].split(' ')
    file_date = output_filename[2].split('15sec')
    output_filename = output_folder + '/' + participant + '_' + output_prefix + '_' + file_date[0] + '.csv'
    wrist_epoch_data.to_csv(output_filename, sep=',')
    print("Process completion " + str(i) + " / " + str(len(file_dictionary)) + " with duration " + str(itr_end_time-itr_start_time) + " seconds.")
    i += 1

process_end_time = time.time()
print("Completed with duration " + str(process_end_time-process_start_time) + " seconds.")


