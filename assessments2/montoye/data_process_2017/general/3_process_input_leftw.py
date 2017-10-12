from os import listdir
from os.path import isfile, join
import pandas as pd
import sys, time

wrists = ['left_wrist', 'right_wrist']
epochs = ['Epoch5', 'Epoch15', 'Epoch30', 'Epoch60']

for wrist in wrists:
    for epoch in epochs:

        output_folder = ('E:\Data\Accelerometer_Montoye_ANN/2017/'+wrist+'/'+epoch+'/input_files/').replace('\\', '/')
        fe_files = []

        experiments = ['LSM1', 'LSM2']
        week = 'Week 1'
        days = ['Wednesday', 'Thursday']

        for experiment in experiments:
            for day in days:
                input_file_path = (
                "E:/Data/Accelerometer_Processed_Raw_Epoch_Data/" + experiment + "/" + week + "/" + day + "/" + epoch + "/").replace(
                    '\\', '/')
                input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]
                for file in input_filenames:
                    fe_files.append(input_file_path + file)

        for file in fe_files:

            start_time = time.time()

            fe_data = pd.read_csv(file)
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

                result_data['VisitTime'] = (result_data.index+1) * 5
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

                result_data['VisitTime'] = (result_data.index+1) * 5
                result_data['Activity'] = 'Freeliving'
                result_data['METS'] = '-1'

            result_data.to_csv((output_folder+file.split('/')[len(file.split('/'))-1].split('.csv')[0]+'.txt'), sep='\t', index=False)

            print('Processed', file.split('_(2016')[0], 'in', round(time.time()-start_time, 2), 'seconds')

print('Completed.')
