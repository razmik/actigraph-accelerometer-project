from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time
import matplotlib.pyplot as plt


epoch_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Epoch_30_processed/'.replace('\\', '/')
predictions_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Montoye_2015_predictions/result_files/'.replace('\\', '/')
output_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Montoye_2015_predictions/combined/'.replace('\\', '/')

epoch_data_files = [f for f in listdir(epoch_folder) if isfile(join(epoch_folder, f))]
prediction_data_files = [f for f in listdir(predictions_folder) if isfile(join(predictions_folder, f))]


def moderate_freedson_intensity(row):
    ee = result_data['predicted_ee'][row.name]
    return 1 if ee < 3 else 2 if ee < 6 else 3


def met_to_intensity(row):
    intensity = result_data['waist_intensity'][row.name]
    return 3 if intensity == 4 else intensity

if len(epoch_data_files) == len(prediction_data_files):
    for epoch_file, pred_file in zip(epoch_data_files, prediction_data_files):

        start_time = time.time()

        result_data = pd.read_csv(epoch_folder + epoch_file)
        del result_data['Unnamed: 0']

        predict_data = pd.read_csv(predictions_folder + pred_file)
        del predict_data['Unnamed: 0']

        result_data['predicted_ee'] = predict_data['V1']

        result_data['moderated_freedson_intensity'] = result_data.apply(moderate_freedson_intensity, axis=1)
        result_data['predicted_intensity_by_ee'] = result_data.apply(met_to_intensity, axis=1)

        result_data.to_csv((output_folder+pred_file.split('_pred')[0]+'_combined.csv'), sep=',')
        print('Processed', pred_file.split(' ')[0], 'in', round(time.time()-start_time, 2), 'seconds')
else:
    print('Invalid input files')
