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
week = 'Week 1'
day = 'Wednesday'

wrist_key = 'Wrist'
hip_key = 'Waist'
epoch_start_rows = 10
# path = "D:/Accelerometer Data/ActilifeProcessedEpochs/" + experiment + "/" + week + "/" + day + "".replace('\\', '/')
# output_folder = "D:/Accelerometer Data/ActilifeProcessedEpochs/" + experiment + "/" + week + "/" + day + "/processed".replace(
#     '\\', '/')
#
# path_components = path.split('/')
# output_prefix = (path_components[3] + '_' + path_components[4] + '_' + path_components[5]).replace(' ', '_')


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


"""
Calculate Waist (hip) epoch values and reference parameters
"""
hip_file = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\LSM203 Waist (2016-11-02)30sec.csv'.replace('\\', '/')
output_filename = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\LSM203 Waist (2016-11-02)30sec_processed.csv'.replace('\\', '/')

hip_epoch_data = pd.read_csv(hip_file, skiprows=epoch_start_rows, usecols=[0, 1, 2])
hip_epoch_data.columns = ['Axis1', 'Axis2', 'Axis3']
hip_epoch_data['waist_vm_15'] = np.sqrt([(hip_epoch_data.Axis1 ** 2) + (hip_epoch_data.Axis2 ** 2) + (hip_epoch_data.Axis3 ** 2)])[0]
hip_epoch_data['waist_vm_60'] = hip_epoch_data['waist_vm_15'] * 2
hip_epoch_data['waist_cpm'] = hip_epoch_data.Axis1 * 2

temp_hip = 0
temp_wrist = 0
length_hip = len(hip_epoch_data)
hip_epoch_data['waist_vm_cpm'] = 0
for index, row in hip_epoch_data.iterrows():
    if index % 2 == 0 and index < length_hip - 1:
        temp_hip = hip_epoch_data.iloc[index]['waist_vm_15'] + hip_epoch_data.iloc[index + 1]['waist_vm_15']
    hip_epoch_data.set_value(index, 'waist_vm_cpm', temp_hip)

hip_epoch_data['waist_ee'] = hip_epoch_data.apply(get_freedson_vm3_combination_11_energy_expenditure, axis=1)
hip_epoch_data['waist_intensity'] = hip_epoch_data.apply(get_freedson_adult_vm3_intensity, axis=1)

"""
Read Wrist Epoch Data to be saved.
"""

hip_epoch_data.to_csv(output_filename, sep=',')
