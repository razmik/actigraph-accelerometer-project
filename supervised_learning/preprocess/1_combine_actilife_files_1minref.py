from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
import time
import itertools
from tqdm import tqdm


"""
MAKE SURETHE EPOCH START ROW NUMBER!!!!
"""


EXPERIMENTS = ['LSM2']
DAYS = ['Wednesday']
WEEKS = ['Week 2']
TIME_EPOCHS = [1, 60]#, 5, 6, 7, 10, 15, 30, 60]

HIP_KEY = 'Waist'
EPOCH_START_ROW = {'Week 1': 10, 'Week 2': 11}
ROOT_FOLDER = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-ActilifeProcessedEpochs/'


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


def get_waist_ee_based_intensity(row_val):
    pa_intensity = 1 if row_val < 1.5 else 2 if row_val < 3 else 3
    return pa_intensity


if __name__ == "__main__":

    process_start_time = time.time()

    for conf in itertools.product(*[EXPERIMENTS, WEEKS, DAYS]):

        experiment, week, day = conf

        input_path = ROOT_FOLDER + "Epoch60-REF-DO-NOT-DELETE/" + experiment + "/" + week + "/" + day

        files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        file_dictionary = {}
        for file in tqdm(files, desc='Reading files for {}'.format(str(conf))):

            # Filename example: LSM203 Waist (2016-11-02)60sec.csv
            file_components = file.split(' ')
            key = file_components[0]

            if key != 'LSM270':
                continue

            if key not in file_dictionary:
                file_dictionary[key] = {file_components[1]: file}
            else:
                file_dictionary[key][file_components[1]] = file

        for participant in tqdm(file_dictionary, desc='Processing files'):
            hip_file = input_path + '/' + file_dictionary[participant][HIP_KEY]

            """
            Calculate Waist (hip) epoch values and reference parameters
            """
            hip_epoch_data = pd.read_csv(hip_file, skiprows=EPOCH_START_ROW[week], usecols=[0, 1, 2], header=None)
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

            hip_epoch_data['waist_vm_60'] = \
            np.sqrt([(hip_epoch_data.AxisX ** 2) + (hip_epoch_data.AxisY ** 2) + (hip_epoch_data.AxisZ ** 2)])[0]
            hip_epoch_data['waist_vm_cpm'] = hip_epoch_data['waist_vm_60']
            hip_epoch_data['waist_cpm'] = hip_epoch_data.AxisY

            hip_epoch_data['waist_ee'] = hip_epoch_data.apply(get_freedson_vm3_combination_11_energy_expenditure, axis=1)
            # hip_epoch_data['waist_intensity'] = hip_epoch_data.apply(get_freedson_adult_vm3_intensity, axis=1)
            hip_epoch_data['waist_intensity_ee_based'] = hip_epoch_data['waist_ee'].apply(get_waist_ee_based_intensity)

            for time_epoch in TIME_EPOCHS:

                epoch_hip_epoch_data = hip_epoch_data.copy(deep=True)

                output_folder = ROOT_FOLDER + "/Epoch" + str(
                    time_epoch) + "/" + experiment + "/" + week + "/" + day + "/processed_1min_ref/"

                if not exists(output_folder):
                    makedirs(output_folder)

                # Number of rows for each epoch
                multiplication_factor = int(60 / time_epoch)

                epoch_hip_epoch_data = epoch_hip_epoch_data.reindex(np.repeat(epoch_hip_epoch_data.index.values, multiplication_factor),
                                                        method='ffill')

                # save output file
                # Filename example: LSM203 Waist (2016-11-02)5sec.csv
                output_filename = file_dictionary[participant][HIP_KEY].split(' ')
                file_date = output_filename[2].split('60sec')
                path_components = input_path.split('/')
                output_prefix = (path_components[6] + '_' + path_components[7] + '_' + path_components[8]).replace(' ', '_')
                output_filename = output_folder + '/' + participant + '_' + output_prefix + '_' + file_date[0] + str(
                    time_epoch) + 's' + '.csv'
                epoch_hip_epoch_data.to_csv(output_filename, sep=',', index=None)
                del epoch_hip_epoch_data

    process_end_time = time.time()
    print("Completed with duration " + str(round((process_end_time - process_start_time) / 60, 2)) + " minutes.")
