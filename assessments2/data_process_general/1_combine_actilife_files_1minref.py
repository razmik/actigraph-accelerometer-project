from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time

process_start_time = time.time()

time_epoch = 60
multiplication_factor = int(60 / time_epoch)

experiment = 'LSM2'
week = 'Week 1'
day = 'Wednesday'

hip_key = 'Waist'
epoch_start_rows = 10
input_path = ("D:\Accelerometer Data\ActilifeProcessedEpochs\Epoch60/" + experiment + "/" + week + "/" + day + "").replace('\\', '/')
output_folder = ("D:\Accelerometer Data\ActilifeProcessedEpochs\Epoch"+str(time_epoch)+"/" + experiment + "/" + week + "/" + day + "/processed_1min_ref/").replace('\\', '/')

path_components = input_path.split('/')
output_prefix = (path_components[4] + '_' + path_components[5] + '_' + path_components[6]).replace(' ', '_')


def get_freedson_adult_vm3_intensity(row):
    # https://actigraph.desk.com/customer/en/portal/articles/2515803-what-s-the-difference-among-the-cut-points-available-in-actilife-
    # Unless otherwise noted, cut points are referencing only counts from the vertical axis (Axis 1 - Y).
    cpm = hip_epoch_data['waist_cpm'][row.name]
    return 1 if cpm <= 2690 else 2 if cpm <= 6166 else 3 if cpm <= 9642 else 4


def get_freedson_vm3_combination_11_energy_expenditure(row):

    if hip_epoch_data['waist_vm_cpm'][row.name] <= 2453:
        # METs = 0.001092(VA) + 1.336129  [capped at 2.9999, where VM3 < 2453]
        met_value = (0.001092 * hip_epoch_data['waist_cpm'][row.name]) + 1.336129
        met_value = met_value if met_value < 2.9999 else 2.9999
    else:
        # METs = 0.000863(VM3) + 0.668876 [where VM3 ≥ 2453]
        met_value = 0.000863 * hip_epoch_data['waist_vm_60'][row.name] + 0.668876

    return met_value

# Old model
# def get_freedson_vm3_combination_11_energy_expenditure(row):
#     # https://actigraph.desk.com/customer/en/portal/articles/2515835-what-is-the-difference-among-the-energy-expenditure-algorithms-
#     if hip_epoch_data['waist_vm_cpm'][row.name] > 2453:
#         # Validation and comparison of ActiGraph activity monitors by Jeffer E. Sasaki
#         met_value = (0.000863 * hip_epoch_data['waist_vm_60'][row.name]) + 0.668876
#     else:
#         # http://www.theactigraph.com/research-database/kcal-estimates-from-activity-counts-using-the-potential-energy-method/
#         body_mass = 80
#         # Step 1: calculate the energy expenditure in Kcals per min
#         Kcals_min = hip_epoch_data['waist_cpm'][row.name] * 0.0000191 * body_mass * 9.81
#
#         # Step 2: convert Energy Expenditure from Kcal/min to kJ/min
#         KJ_min = Kcals_min * 4.184
#
#         # Step 3: assuming that you use 80 kg as body mass, divide value by ((3.5/1000)*80*20.9)
#         met_value = KJ_min / ((3.5 / 1000) * body_mass * 20.9)
#
#     return met_value


print("Reading files and creating dictionary.")
files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
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
    hip_file = input_path + '/' + file_dictionary[participant][hip_key]

    """
    Calculate Waist (hip) epoch values and reference parameters
    """
    hip_epoch_data = pd.read_csv(hip_file, skiprows=epoch_start_rows, usecols=[0, 1, 2], header=None)
    # Axis 1 (y) - Goes through head and middle of feet
    # Axis 2 (x) - Goes through 2 hips
    # Axis 3 (z) - Goes through front and back of the stomach
    hip_epoch_data.columns = ['AxisY', 'AxisX', 'AxisZ']

    """
    https://actigraph.desk.com/customer/en/portal/articles/2515803-what-s-the-difference-among-the-cut-points-available-in-actilife-
    * Unless otherwise noted, cut points are referencing only counts from the vertical axis (Axis 1).  
    * Cutpoints with "VM" or "Vector Magnitude" annotation use 
        the Vector Magnitude (SQRT[(Axis 1)^2 + (Axis 2)^2 + (Axis 3)^2 ] ) count value.
    """

    hip_epoch_data['waist_vm_60'] = np.sqrt([(hip_epoch_data.AxisX ** 2) + (hip_epoch_data.AxisY ** 2) + (hip_epoch_data.AxisZ ** 2)])[0]
    hip_epoch_data['waist_vm_cpm'] = hip_epoch_data['waist_vm_60']
    hip_epoch_data['waist_cpm'] = hip_epoch_data.AxisY

    hip_epoch_data['waist_ee'] = hip_epoch_data.apply(get_freedson_vm3_combination_11_energy_expenditure, axis=1)
    hip_epoch_data['waist_intensity'] = hip_epoch_data.apply(get_freedson_adult_vm3_intensity, axis=1)

    hip_epoch_data = hip_epoch_data.reindex(np.repeat(hip_epoch_data.index.values, multiplication_factor), method='ffill')

    itr_end_time = time.time()

    # save output file
    # Filename example: LSM203 Waist (2016-11-02)5sec.csv
    output_filename = file_dictionary[participant][hip_key].split(' ')
    file_date = output_filename[2].split('60sec')
    output_filename = output_folder + '/' + participant + '_' + output_prefix + '_' + file_date[0] + str(time_epoch) + 's' + '.csv'
    hip_epoch_data.to_csv(output_filename, sep=',', index=None)
    print("Process completion " + str(i) + " / " + str(len(file_dictionary)) + " with duration " + str(
        round(itr_end_time - itr_start_time, 2)) + " seconds.")
    i += 1

process_end_time = time.time()
print("Completed with duration " + str(round((process_end_time - process_start_time) / 60, 2)) + " minutes.")
