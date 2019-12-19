from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
import time
import itertools
from tqdm import tqdm
import sys


EXPERIMENTS = ['LSM1', 'LSM2']
DAYS = ['Thursday', 'Wednesday']
WEEKS = ['Week 2']
TIME_EPOCHS = [60]

HIP_KEY = 'Waist'
EPOCH_START_ROW = 11
ROOT_FOLDER = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-ActilifeProcessedEpochs/'


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


def get_wrist_cutpoint_intensity_montoye(row_vm):
    """
    The cut-points my group has developed are for count-based, 60-second epoch data (with LFE turned off) for an
    ActiGraph accelerometer worn on the non-dominant wrist. The cut-points are
    <2,860 counts/min – sedentary;
    2,860-3,940 counts/min – light;
    2,941-5,612 counts/min – moderate;
    ≥5,613 counts/min – vigorous,
    and we recommend in the paper that moderate and vigorous be combined into MVPA.
    """
    pa_intensity = 1 if row_vm < 2860 else 2 if row_vm < 3940 else 3
    return pa_intensity


if __name__ == "__main__":

    for conf in itertools.product(*[EXPERIMENTS, WEEKS, DAYS]):

        experiment, week, day = conf

        hip_input_path = ROOT_FOLDER + "Epoch60-REF-DO-NOT-DELETE/" + experiment + "/" + week + "/" + day
        wrist_input_path = ROOT_FOLDER + "Epoch60-WRIST-EPOCHS/" + experiment + "/" + week + "/" + day

        hip_files = sorted([f for f in listdir(hip_input_path) if isfile(join(hip_input_path, f))])
        wrist_files = sorted([f for f in listdir(wrist_input_path) if isfile(join(wrist_input_path, f))])

        # Verify sort
        file_dictionary = {}
        for h, w in zip(hip_files, wrist_files):
            hk = h.split('/')[-1].split()[0]
            wk = w.split('/')[-1].split()[0]

            if hk != wk:
                print('ERROR in FILES {} - {}:\n'.format(experiment, day), h, '\n', w)
                continue

            file_dictionary[hk] = {
                'hip': join(hip_input_path, h),
                'wrist': join(wrist_input_path, w)
            }

        for participant in tqdm(file_dictionary, desc='Processing files - {} : {}'.format(experiment, day)):

            """
            Calculate Waist (hip) epoch values and reference parameters
            """
            # Axis 1 (y) - Goes through head and middle of feet
            # Axis 2 (x) - Goes through 2 hips
            # Axis 3 (z) - Goes through front and back of the stomach
            hip_file = file_dictionary[participant]['hip']
            hip_epoch_data = pd.read_csv(hip_file, skiprows=EPOCH_START_ROW, usecols=[0, 1, 2], header=None)
            hip_epoch_data.columns = ['AxisY', 'AxisX', 'AxisZ']

            hip_epoch_data['waist_vm_60'] = \
            np.sqrt([(hip_epoch_data.AxisX ** 2) + (hip_epoch_data.AxisY ** 2) + (hip_epoch_data.AxisZ ** 2)])[0]
            hip_epoch_data['waist_vm_cpm'] = hip_epoch_data['waist_vm_60']
            hip_epoch_data['waist_cpm'] = hip_epoch_data.AxisY
            hip_epoch_data['waist_ee'] = hip_epoch_data.apply(get_freedson_vm3_combination_11_energy_expenditure, axis=1)
            hip_epoch_data['waist_intensity'] = hip_epoch_data['waist_ee'].apply(get_waist_ee_based_intensity)

            """
            Calculate Non-dominent Wrist epoch values and reference parameters
            """
            wrist_file = file_dictionary[participant]['wrist']
            wrist_epoch_data = pd.read_csv(wrist_file, skiprows=EPOCH_START_ROW, usecols=[0, 1, 2], header=None)
            wrist_epoch_data.columns = ['AxisY', 'AxisX', 'AxisZ']

            wrist_epoch_data['wrist_vm_60'] = \
                np.sqrt([(wrist_epoch_data.AxisX ** 2) + (wrist_epoch_data.AxisY ** 2) + (wrist_epoch_data.AxisZ ** 2)])[0]
            wrist_epoch_data['montoye_wrist_intensity'] = wrist_epoch_data['wrist_vm_60'].apply(get_wrist_cutpoint_intensity_montoye)

            """
            Combine
            """
            if hip_epoch_data.shape[0] != wrist_epoch_data.shape[0]:
                print('Error in ', participant, hip_epoch_data.shape[0], wrist_epoch_data.shape[0])
                sys.exit(0)

            hip_epoch_data['wrist_vm_60'] = wrist_epoch_data['wrist_vm_60']
            hip_epoch_data['montoye_wrist_intensity'] = wrist_epoch_data['montoye_wrist_intensity']

            # for time_epoch in TIME_EPOCHS:
            #
            #     epoch_hip_epoch_data = hip_epoch_data.copy(deep=True)
            #
            #     output_folder = ROOT_FOLDER + "/Epoch" + str(
            #         time_epoch) + "/" + experiment + "/" + week + "/" + day + "/processed_1min_ref/"
            #
            #     if not exists(output_folder):
            #         makedirs(output_folder)
            #
            #     # Number of rows for each epoch
            #     multiplication_factor = int(60 / time_epoch)
            #
            #     epoch_hip_epoch_data = epoch_hip_epoch_data.reindex(np.repeat(epoch_hip_epoch_data.index.values, multiplication_factor),
            #                                             method='ffill')
            #
            #     # save output file
            #     # Filename example: LSM203 Waist (2016-11-02)5sec.csv
            #     output_filename = file_dictionary[participant][HIP_KEY].split(' ')
            #     file_date = output_filename[2].split('60sec')
            #     path_components = input_path.split('/')
            #     output_prefix = (path_components[6] + '_' + path_components[7] + '_' + path_components[8]).replace(' ', '_')
            #     output_filename = output_folder + '/' + participant + '_' + output_prefix + '_' + file_date[0] + str(
            #         time_epoch) + 's' + '.csv'
            #     epoch_hip_epoch_data.to_csv(output_filename, sep=',', index=None)
            #     del epoch_hip_epoch_data

    process_end_time = time.time()
    print("Completed with duration " + str(round((process_end_time - process_start_time) / 60, 2)) + " minutes.")
