"""
1. Read all files in the folder
2. Get participant IDs into an array
3. For each array,
    a. Get the waist file,
        calculate VM 15s,
        VM 60a (CPM),
        cut point intensity with Freedson Adult VM3 (2011), https://actigraph.desk.com/customer/en/portal/articles/2515803-what-s-the-difference-among-the-cut-points-available-in-actilife-
        energy expenditure (EE) with Freedson Treadmill Adult (1998), https://actigraph.desk.com/customer/en/portal/articles/2515804-what-is-the-difference-among-the-met-algorithms-

        VM = Vector Magnitude Combination (per minute) of all 3 axes (sqrt((Axis 1)^2+(Axis 2)^2+(Axis 3)^2]) - vm * 4
        VMCPM = Vector Magnitude Counts per Minute - Non overlapping 1 minute's sum of VMs - sum (vm for non-overlapping window)
        CPM = Counts per Minute - count of Axis 1
        BM = Body Mass in kg

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

experiment = 'LSM2'
week = 'Week 2'
day = 'Wednesday'

wrist_key = 'Wrist'
hip_key = 'Waist'
epoch_start_rows = 10
path = "D:/Accelerometer Data/ActilifeProcessedEpochs/" + experiment + "/" + week + "/" + day + "".replace('\\', '/')
output_folder = "D:/Accelerometer Data/ActilifeProcessedEpochs/" + experiment + "/" + week + "/" + day + "/processed".replace(
    '\\', '/')

path_components = path.split('/')
output_prefix = (path_components[3] + '_' + path_components[4] + '_' + path_components[5]).replace(' ', '_')


def get_freedson_adult_vm3_intensity(row):
    cpm = hip_epoch_data['waist_vm_60'][row.name]
    return 1 if cpm <= 2690 else 2 if cpm <= 6166 else 3 if cpm <= 9642 else 4


def get_freedson_vm3_combination_11_energy_expenditure(row):
    """
    https://actigraph.desk.com/customer/en/portal/articles/2515835-what-is-the-difference-among-the-energy-expenditure-algorithms-
    if VMCPM > 2453
        Kcals/min= 0.001064×VM + 0.087512(BM) - 5.500229
    else
        Kcals/min=CPM×0.0000191×BM
    where
        VM = Vector Magnitude Combination (per minute) of all 3 axes (sqrt((Axis 1)^2+(Axis 2)^2+(Axis 3)^2])
        VMCPM = Vector Magnitude Counts per Minute
        CPM = Counts per Minute
        BM = Body Mass in kg
    """
    #  calculate Energy Expenditure using VM3 and a constant body mass (e.g., 80 kg)
    if hip_epoch_data['waist_vm_cpm'][row.name] > 2453:
        met_value = (0.001064 * hip_epoch_data['waist_vm_60'][row.name]) + (0.087512 * 80) - 5.500229
    else:
        met_value = hip_epoch_data['waist_cpm'][row.name] * 0.0000191 * 80

        #  convert Energy Expenditure from Kcal/min to kJ/min
        met_value *= 4.184

        #  assuming that you use 80 kg as body mass, divide value by ((3.5/1000)*80*20.9)
        met_value /= ((3.5 / 1000) * 80 * 20.9)

    return met_value


def get_equivalent_count_for_axis1(row):
    # https://actigraph.desk.com/customer/en/portal/articles/2515826-what-does-the-%22worn-on-wrist%22-option-do-in-the-data-scoring-tab-
    count = wrist_epoch_data['wrist_Axis1'][row.name]
    if count <= 644:
        result = 0.5341614 * count
    elif count <= 1272:
        result = 1.7133758 * count - 759.414013
    elif count <= 3806:
        result = 0.3997632 * count + 911.501184
    else:
        result = 0.0128995 * count + 2383.904505
    return result


def get_equivalent_count_for_axis2(row):
    count = wrist_epoch_data['wrist_Axis2'][row.name]
    if count <= 644:
        result = 0.5341614 * count
    elif count <= 1272:
        result = 1.7133758 * count - 759.414013
    elif count <= 3806:
        result = 0.3997632 * count + 911.501184
    else:
        result = 0.0128995 * count + 2383.904505
    return result


def get_equivalent_count_for_axis3(row):
    count = wrist_epoch_data['wrist_Axis3'][row.name]
    if count <= 644:
        result = 0.5341614 * count
    elif count <= 1272:
        result = 1.7133758 * count - 759.414013
    elif count <= 3806:
        result = 0.3997632 * count + 911.501184
    else:
        result = 0.0128995 * count + 2383.904505
    return result


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

    """
    Calculate Waist (hip) epoch values and reference parameters
    """
    hip_epoch_data = pd.read_csv(hip_file, skiprows=epoch_start_rows, usecols=[0, 1, 2])
    hip_epoch_data.columns = ['Axis1', 'Axis2', 'Axis3']
    hip_epoch_data['waist_vm_15'] = np.sqrt([(hip_epoch_data.Axis1 ** 2) + (hip_epoch_data.Axis2 ** 2) + (hip_epoch_data.Axis3 ** 2)])[0]
    hip_epoch_data['waist_vm_60'] = hip_epoch_data['waist_vm_15'] * 4
    hip_epoch_data['waist_cpm'] = hip_epoch_data.Axis1 * 4

    """
    Calculate Wrist epoch values and reference parameters
    """
    wrist_epoch_data = pd.read_csv(wrist_file, skiprows=epoch_start_rows, usecols=[0, 1, 2])
    wrist_epoch_data.columns = ['wrist_Axis1', 'wrist_Axis2', 'wrist_Axis3']
    wrist_epoch_data['wrist_vm_15'] = np.sqrt([(wrist_epoch_data.wrist_Axis1 ** 2) + (wrist_epoch_data.wrist_Axis2 ** 2) + (wrist_epoch_data.wrist_Axis3 ** 2)])[0]
    wrist_epoch_data['wrist_vm_60'] = wrist_epoch_data['wrist_vm_15'] * 4
    wrist_epoch_data['wrist_Axis1_waist_eq'] = wrist_epoch_data.apply(get_equivalent_count_for_axis1, axis=1)
    wrist_epoch_data['wrist_Axis2_waist_eq'] = wrist_epoch_data.apply(get_equivalent_count_for_axis2, axis=1)
    wrist_epoch_data['wrist_Axis3_waist_eq'] = wrist_epoch_data.apply(get_equivalent_count_for_axis3, axis=1)
    wrist_epoch_data['wrist_vm_waist_eq'] = np.sqrt([(wrist_epoch_data['wrist_Axis1_waist_eq'] ** 2) + (wrist_epoch_data['wrist_Axis2_waist_eq'] ** 2) + (wrist_epoch_data['wrist_Axis3_waist_eq'] ** 2)])[0]
    wrist_epoch_data['wrist_cpm'] = wrist_epoch_data.wrist_Axis1 * 4

    temp_hip = 0
    temp_wrist = 0
    length_hip = len(hip_epoch_data)
    length_waist = len(wrist_epoch_data)
    hip_epoch_data['waist_vm_cpm'] = 0
    wrist_epoch_data['wrist_vm_cpm'] = 0
    for index, row in hip_epoch_data.iterrows():
        if index % 4 == 0 and index < length_hip - 3 and index < length_waist - 3:
            temp_hip = hip_epoch_data.iloc[index]['waist_vm_15'] + hip_epoch_data.iloc[index + 1]['waist_vm_15'] + \
                       hip_epoch_data.iloc[index + 2]['waist_vm_15'] + hip_epoch_data.iloc[index + 3]['waist_vm_15']
            temp_wrist = wrist_epoch_data.iloc[index]['wrist_vm_15'] + wrist_epoch_data.iloc[index + 1][
                'wrist_vm_15'] + wrist_epoch_data.iloc[index + 2]['wrist_vm_15'] + wrist_epoch_data.iloc[index + 3][
                             'wrist_vm_15']
        hip_epoch_data.set_value(index, 'waist_vm_cpm', temp_hip)
        wrist_epoch_data.set_value(index, 'wrist_vm_cpm', temp_wrist)

    hip_epoch_data['waist_ee'] = hip_epoch_data.apply(get_freedson_vm3_combination_11_energy_expenditure, axis=1)
    hip_epoch_data['waist_intensity'] = hip_epoch_data.apply(get_freedson_adult_vm3_intensity, axis=1)

    """
    Read Wrist Epoch Data to be saved.
    """

    wrist_epoch_data['waist_vm_15'] = hip_epoch_data['waist_vm_15']
    wrist_epoch_data['waist_vm_60'] = hip_epoch_data['waist_vm_60']
    wrist_epoch_data['waist_vm_cpm'] = hip_epoch_data['waist_vm_cpm']
    wrist_epoch_data['waist_cpm'] = hip_epoch_data['waist_cpm']
    wrist_epoch_data['waist_intensity'] = hip_epoch_data['waist_intensity']
    wrist_epoch_data['waist_ee'] = hip_epoch_data['waist_ee']
    wrist_epoch_data['waist_Axis1'] = hip_epoch_data['Axis1']
    wrist_epoch_data['waist_Axis2'] = hip_epoch_data['Axis2']
    wrist_epoch_data['waist_Axis3'] = hip_epoch_data['Axis3']

    itr_end_time = time.time()

    # save output file
    # Filename example: LSM203 Waist (2016-11-02)15sec.csv
    output_filename = file_dictionary[participant][wrist_key].split(' ')
    file_date = output_filename[2].split('15sec')
    output_filename = output_folder + '/' + participant + '_' + output_prefix + '_' + file_date[0] + '.csv'
    wrist_epoch_data.to_csv(output_filename, sep=',')
    print("Process completion " + str(i) + " / " + str(len(file_dictionary)) + " with duration " + str(
        round(itr_end_time - itr_start_time, 2)) + " seconds.")
    i += 1

process_end_time = time.time()
print("Completed with duration " + str(round((process_end_time - process_start_time) / 60, 2)) + " minutes.")
