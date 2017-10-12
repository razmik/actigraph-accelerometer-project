from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time
import matplotlib.pyplot as plt
import winsound


wrists = ['left_wrist']
epochs = ['Epoch15', 'Epoch30', 'Epoch60']
# wrists = ['left_wrist', 'right_wrist']
# epochs = ['Epoch5', 'Epoch15', 'Epoch30', 'Epoch60']
experiments = ['LSM1', 'LSM2']
week = 'Week 1'
days = ['Wednesday', 'Thursday']
models = ['v1v2', 'v2']
for model in models:
    for wrist in wrists:
        for epoch in epochs:

            epoch_data_files = []
            for experiment in experiments:
                for day in days:
                    input_file_path = (
                    "E:/Data/Accelerometer_Processed_Raw_Epoch_Data/" + experiment + "/" + week + "/" + day + "/" + epoch + "/").replace(
                        '\\', '/')
                    input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]
                    for file in input_filenames:
                        epoch_data_files.append(input_file_path + file)

            predictions_folder = ('E:\Data\Accelerometer_Montoye_ANN/2017/'+wrist+'/'+epoch+'/'+model+'/result_files/').replace('\\', '/')
            output_folder = ('E:\Data\Accelerometer_Montoye_ANN/2017/'+wrist+'/'+epoch+'/'+model+'/combined/').replace('\\', '/')

            prediction_data_files = [f for f in listdir(predictions_folder) if isfile(join(predictions_folder, f))]


            def get_matching_epoch_file(result_filename):
                user = result_filename.split('_(')[0]
                duration = result_filename.split(')_')[1].split('_predicted')[0]

                for epoch_data_file in epoch_data_files:
                    if user in epoch_data_file and duration in epoch_data_file:
                        return epoch_data_file

                return None

            def met_to_intensity(row):
                ee = epoch_data['waist_ee'][row.name]
                return 1 if ee <= 1.5 else 2 if ee < 3 else 3


            def get_predicted_activity_intensity_level(row):
                ee = epoch_data['predicted_ee'][row.name]
                return 1 if ee <= 1.5 else 2 if ee < 3 else 3

            if len(epoch_data_files) == len(prediction_data_files):

                for pred_file in prediction_data_files:

                    epoch_filename = get_matching_epoch_file(pred_file)
                    pred_filename = predictions_folder + pred_file

                    epoch_data = pd.read_csv(epoch_filename)
                    predict_data = pd.read_csv(pred_filename)

                    if len(epoch_data) == len(predict_data):

                        epoch_data['predicted_ee'] = predict_data['V1']
                        epoch_data['predicted_category'] = epoch_data.apply(get_predicted_activity_intensity_level, axis=1)
                        epoch_data['actual_category'] = epoch_data.apply(met_to_intensity, axis=1)

                        epoch_data.to_csv((output_folder+pred_file.split('_pred')[0]+'_'+wrist+'_combined.csv'), sep=',')
                        print('Processed', model, wrist, epoch, pred_file.split(' ')[0])

                    else:
                        print('Error in length\n', epoch_filename, '\n', pred_filename)

            else:
                print('Invalid input files')

Freq = 2500 # Set Frequency To 2500 Hertz
Dur = 20000 # Set Duration To 1000 ms == 1 second
winsound.Beep(Freq, Dur)