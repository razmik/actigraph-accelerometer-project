from os import listdir
from os.path import isfile, join
import pandas as pd
import sys, time

wrist = 'right_wrist'

output_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Assessment2/Montoye_2017_predictions/'+wrist+'/60sec/input_files/'.replace('\\', '/')
fe_files_path = 'D:\Accelerometer Data\Assessment\montoye\LSM2\Week 1\Wednesday\Epoch60/'.replace('\\', '/')
fe_files = [f for f in listdir(fe_files_path) if isfile(join(fe_files_path, f))]

for file in fe_files:

    start_time = time.time()

    fe_data = pd.read_csv(fe_files_path + file)
    del fe_data['Unnamed: 0']

    if wrist == 'left_wrist':

        columns = ['VisitTime', 'Activity', 'METS',
                   'AL_LW_X_pTen', 'AL_LW_X_pTwentyFive', 'AL_LW_X_pFifty', 'AL_LW_X_pSeventyFive', 'AL_LW_X_pNinety', 'AL_LW_X_cov',
                   'AL_LW_Y_pTen', 'AL_LW_Y_pTwentyFive', 'AL_LW_Y_pFifty', 'AL_LW_Y_pSeventyFive', 'AL_LW_Y_pNinety', 'AL_LW_Y_cov',
                   'AL_LW_Z_pTen', 'AL_LW_Z_pTwentyFive', 'AL_LW_Z_pFifty', 'AL_LW_Z_pSeventyFive', 'AL_LW_Z_pNinety', 'AL_LW_Z_cov']

        result_data = pd.DataFrame(columns=columns)

        result_data['AL_LW_X_pTen'] = fe_data['raw_wrist_X10perc']
        result_data['AL_LW_Y_pTen'] = fe_data['raw_wrist_Y10perc']
        result_data['AL_LW_Z_pTen'] = fe_data['raw_wrist_Z10perc']
        result_data['AL_LW_X_pTwentyFive'] = fe_data['raw_wrist_X25perc']
        result_data['AL_LW_Y_pTwentyFive'] = fe_data['raw_wrist_Y25perc']
        result_data['AL_LW_Z_pTwentyFive'] = fe_data['raw_wrist_Z25perc']
        result_data['AL_LW_X_pFifty'] = fe_data['raw_wrist_X50perc']
        result_data['AL_LW_Y_pFifty'] = fe_data['raw_wrist_Y50perc']
        result_data['AL_LW_Z_pFifty'] = fe_data['raw_wrist_Z50perc']
        result_data['AL_LW_X_pSeventyFive'] = fe_data['raw_wrist_X75perc']
        result_data['AL_LW_Y_pSeventyFive'] = fe_data['raw_wrist_Y75perc']
        result_data['AL_LW_Z_pSeventyFive'] = fe_data['raw_wrist_Z75perc']
        result_data['AL_LW_X_pNinety'] = fe_data['raw_wrist_X90perc']
        result_data['AL_LW_Y_pNinety'] = fe_data['raw_wrist_Y90perc']
        result_data['AL_LW_Z_pNinety'] = fe_data['raw_wrist_Z90perc']
        result_data['AL_LW_X_cov'] = fe_data['raw_wrist_X_cov']
        result_data['AL_LW_Y_cov'] = fe_data['raw_wrist_Y_cov']
        result_data['AL_LW_Z_cov'] = fe_data['raw_wrist_Z_cov']

        result_data['VisitTime'] = (result_data.index+1) * 60
        result_data['Activity'] = 'Freeliving'
        result_data['METS'] = '-1'

    elif wrist == 'right_wrist':

        columns = ['VisitTime', 'Activity', 'METS',
                   'AL_RW_X_pTen', 'AL_RW_X_pTwentyFive', 'AL_RW_X_pFifty', 'AL_RW_X_pSeventyFive', 'AL_RW_X_pNinety', 'AL_RW_X_cov',
                   'AL_RW_Y_pTen', 'AL_RW_Y_pTwentyFive', 'AL_RW_Y_pFifty', 'AL_RW_Y_pSeventyFive', 'AL_RW_Y_pNinety', 'AL_RW_Y_cov',
                   'AL_RW_Z_pTen', 'AL_RW_Z_pTwentyFive', 'AL_RW_Z_pFifty', 'AL_RW_Z_pSeventyFive', 'AL_RW_Z_pNinety', 'AL_RW_Z_cov']

        result_data = pd.DataFrame(columns=columns)

        result_data['AL_RW_X_pTen'] = fe_data['raw_wrist_X10perc']
        result_data['AL_RW_Y_pTen'] = fe_data['raw_wrist_Y10perc']
        result_data['AL_RW_Z_pTen'] = fe_data['raw_wrist_Z10perc']
        result_data['AL_RW_X_pTwentyFive'] = fe_data['raw_wrist_X25perc']
        result_data['AL_RW_Y_pTwentyFive'] = fe_data['raw_wrist_Y25perc']
        result_data['AL_RW_Z_pTwentyFive'] = fe_data['raw_wrist_Z25perc']
        result_data['AL_RW_X_pFifty'] = fe_data['raw_wrist_X50perc']
        result_data['AL_RW_Y_pFifty'] = fe_data['raw_wrist_Y50perc']
        result_data['AL_RW_Z_pFifty'] = fe_data['raw_wrist_Z50perc']
        result_data['AL_RW_X_pSeventyFive'] = fe_data['raw_wrist_X75perc']
        result_data['AL_RW_Y_pSeventyFive'] = fe_data['raw_wrist_Y75perc']
        result_data['AL_RW_Z_pSeventyFive'] = fe_data['raw_wrist_Z75perc']
        result_data['AL_RW_X_pNinety'] = fe_data['raw_wrist_X90perc']
        result_data['AL_RW_Y_pNinety'] = fe_data['raw_wrist_Y90perc']
        result_data['AL_RW_Z_pNinety'] = fe_data['raw_wrist_Z90perc']
        result_data['AL_RW_X_cov'] = fe_data['raw_wrist_X_cov']
        result_data['AL_RW_Y_cov'] = fe_data['raw_wrist_Y_cov']
        result_data['AL_RW_Z_cov'] = fe_data['raw_wrist_Z_cov']

        result_data['VisitTime'] = (result_data.index+1) * 60
        result_data['Activity'] = 'Freeliving'
        result_data['METS'] = '-1'

    result_data.to_csv((output_folder+file.split('.csv')[0]+'.txt'), sep='\t', index=False)

    print('Processed', file.split(' ')[0], 'in', round(time.time()-start_time, 2), 'seconds')

print('Completed.')
