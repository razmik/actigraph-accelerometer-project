from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import sys, time
import matplotlib.pyplot as plt

output_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Montoye_2015_predictions\input_files/'.replace('\\', '/')
fe_files_path = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\FE_30_processed/'.replace('\\', '/')
fe_files = [f for f in listdir(fe_files_path) if isfile(join(fe_files_path, f))]

for file in fe_files:

    start_time = time.time()

    fe_data = pd.read_csv(fe_files_path + file)
    del fe_data['Unnamed: 0']

    columns = ['Participant', 'Activity', 'OC_METs', 'TimeSec', 'Mean_AGhipX', 'Mean_AGhipY', 'Mean_AGhipZ', 'Var_AGhipX', 'Var_AGhipY', 'Var_AGhipZ']

    result_data = pd.DataFrame(columns=columns)
    result_data['Mean_AGhipX'] = fe_data['XMean']
    result_data['Mean_AGhipY'] = fe_data['YMean']
    result_data['Mean_AGhipZ'] = fe_data['ZMean']
    result_data['Var_AGhipX'] = fe_data['XVar']
    result_data['Var_AGhipY'] = fe_data['YVar']
    result_data['Var_AGhipZ'] = fe_data['ZVar']
    result_data['Participant'] = file.split(' ')[0]
    result_data['Activity'] = 'Freeliving'
    result_data['OC_METs'] = 1
    result_data['TimeSec'] = (result_data.index+1) * 30

    result_data.to_csv((output_folder+file.split('.csv')[0]+'.txt'), sep='\t', index=False)

    print('Processed', file.split(' ')[0], 'in', round(time.time()-start_time, 2), 'seconds')

print('Completed.')
