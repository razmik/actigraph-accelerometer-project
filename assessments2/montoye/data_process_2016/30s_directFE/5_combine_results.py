from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time
import matplotlib.pyplot as plt


wrist = 'left_wrist'

epoch_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Epoch_30_processed/'.replace('\\', '/')
predictions_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Montoye_2016_predictions/'+wrist+'/30sec/result_files/'.replace('\\', '/')
output_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Montoye_2016_predictions/'+wrist+'/30sec/combined/'.replace('\\', '/')

epoch_data_files = [f for f in listdir(epoch_folder) if isfile(join(epoch_folder, f))]
prediction_data_files = [f for f in listdir(predictions_folder) if isfile(join(predictions_folder, f))]


# def process_actual(row):
#     intensity = result_data['waist_intensity'][row.name]
#     return 1 if intensity <= 1.5 else 2
#
#
# def process_predicted(row):
#     sed = predict_data['SED'][row.name]
#     lpa = predict_data['LPA'][row.name]
#     mvpa = predict_data['MVPA'][row.name]
#
#     if sed > lpa and sed > mvpa:
#         return 1
#     elif lpa > sed and lpa > mvpa:
#         return 1
#     elif mvpa > sed and mvpa > lpa:
#         return 2


def met_to_intensity(row):
    ee = result_data['waist_ee'][row.name]
    return 1 if ee < 1.5 else 2 if ee < 3 else 3


def get_predicted_activity_intensity_level(row):
    sed = predict_data['SED'][row.name]
    lpa = predict_data['LPA'][row.name]
    mvpa = predict_data['MVPA'][row.name]

    if sed > lpa and sed > mvpa:
        return 1
    elif lpa > sed and lpa > mvpa:
        return 2
    elif mvpa > sed and mvpa > lpa:
        return 3

if len(epoch_data_files) == len(prediction_data_files):
    for epoch_file, pred_file in zip(epoch_data_files, prediction_data_files):

        start_time = time.time()

        result_data = pd.read_csv(epoch_folder + epoch_file)
        del result_data['Unnamed: 0']

        predict_data = pd.read_csv(predictions_folder + pred_file)
        del predict_data['Unnamed: 0']

        result_data['predicted_category'] = result_data.apply(get_predicted_activity_intensity_level, axis=1)

        result_data['actual_category'] = result_data.apply(met_to_intensity, axis=1)

        # result_data['predicted_intensity'] = result_data.apply(process_predicted, axis=1)
        #
        # result_data['actual_intensity'] = result_data.apply(process_actual, axis=1)

        result_data.to_csv((output_folder+pred_file.split('_pred')[0]+'_'+wrist+'_combined.csv'), sep=',')
        print('Processed', pred_file.split(' ')[0], 'in', round(time.time()-start_time, 2), 'seconds')
else:
    print('Invalid input files')
