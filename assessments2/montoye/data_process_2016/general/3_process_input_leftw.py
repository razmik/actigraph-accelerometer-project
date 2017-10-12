from os import listdir
from os.path import isfile, join
import pandas as pd
import sys, time

wrists = ['left_wrist', 'right_wrist']
epochs = ['Epoch5', 'Epoch15', 'Epoch30', 'Epoch60']

for wrist in wrists:
    for epoch in epochs:

        output_folder = ('E:\Data\Accelerometer_Montoye_ANN/2016/'+wrist+'/'+epoch+'/input_files/').replace('\\', '/')
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

                columns = ['TenPct_GEleftX', 'TenPct_GEleftY', 'TenPct_GEleftZ', 'TwentyFivePct_GEleftY',
                           'TwentyFivePct_GEleftZ',
                           'FiftyPct_GEleftX', 'FiftyPct_GEleftY', 'FiftyPct_GEleftZ', 'SeventyFivePct_GEleftX',
                           'SeventyFivePct_GEleftY',
                           'SeventyFivePct_GEleftZ', 'NinetyPct_GEleftX', 'NinetyPct_GEleftY', 'NinetyPct_GEleftZ']

                result_data = pd.DataFrame(columns=columns)

                result_data['TenPct_GEleftX'] = fe_data['raw_wrist_X10perc']
                result_data['TenPct_GEleftY'] = fe_data['raw_wrist_Y10perc']
                result_data['TenPct_GEleftZ'] = fe_data['raw_wrist_Z10perc']
                result_data['TwentyFivePct_GEleftX'] = fe_data['raw_wrist_X25perc']
                result_data['TwentyFivePct_GEleftY'] = fe_data['raw_wrist_Y25perc']
                result_data['TwentyFivePct_GEleftZ'] = fe_data['raw_wrist_Z25perc']
                result_data['FiftyPct_GEleftX'] = fe_data['raw_wrist_X50perc']
                result_data['FiftyPct_GEleftY'] = fe_data['raw_wrist_Y50perc']
                result_data['FiftyPct_GEleftZ'] = fe_data['raw_wrist_Z50perc']
                result_data['SeventyFivePct_GEleftX'] = fe_data['raw_wrist_X75perc']
                result_data['SeventyFivePct_GEleftY'] = fe_data['raw_wrist_Y75perc']
                result_data['SeventyFivePct_GEleftZ'] = fe_data['raw_wrist_Z75perc']
                result_data['NinetyPct_GEleftX'] = fe_data['raw_wrist_X90perc']
                result_data['NinetyPct_GEleftY'] = fe_data['raw_wrist_Y90perc']
                result_data['NinetyPct_GEleftZ'] = fe_data['raw_wrist_Z90perc']

            elif wrist == 'right_wrist':

                columns = ['TenPct_GErightX', 'TenPct_GErightY', 'TenPct_GErightZ', 'TwentyFivePct_GErightY',
                           'TwentyFivePct_GErightZ',
                           'FiftyPct_GErightX', 'FiftyPct_GErightY', 'FiftyPct_GErightZ', 'SeventyFivePct_GErightX',
                           'SeventyFivePct_GErightY',
                           'SeventyFivePct_GErightZ', 'NinetyPct_GErightX', 'NinetyPct_GErightY', 'NinetyPct_GErightZ']

                result_data = pd.DataFrame(columns=columns)

                result_data['TenPct_GErightX'] = fe_data['raw_wrist_X10perc']
                result_data['TenPct_GErightY'] = fe_data['raw_wrist_Y10perc']
                result_data['TenPct_GErightZ'] = fe_data['raw_wrist_Z10perc']
                result_data['TwentyFivePct_GErightX'] = fe_data['raw_wrist_X25perc']
                result_data['TwentyFivePct_GErightY'] = fe_data['raw_wrist_Y25perc']
                result_data['TwentyFivePct_GErightZ'] = fe_data['raw_wrist_Z25perc']
                result_data['FiftyPct_GErightX'] = fe_data['raw_wrist_X50perc']
                result_data['FiftyPct_GErightY'] = fe_data['raw_wrist_Y50perc']
                result_data['FiftyPct_GErightZ'] = fe_data['raw_wrist_Z50perc']
                result_data['SeventyFivePct_GErightX'] = fe_data['raw_wrist_X75perc']
                result_data['SeventyFivePct_GErightY'] = fe_data['raw_wrist_Y75perc']
                result_data['SeventyFivePct_GErightZ'] = fe_data['raw_wrist_Z75perc']
                result_data['NinetyPct_GErightX'] = fe_data['raw_wrist_X90perc']
                result_data['NinetyPct_GErightY'] = fe_data['raw_wrist_Y90perc']
                result_data['NinetyPct_GErightZ'] = fe_data['raw_wrist_Z90perc']

            result_data.to_csv((output_folder+file.split('/')[len(file.split('/'))-1].split('.csv')[0]+'.txt'), sep='\t', index=False)

            print('Processed', wrist, epoch, file.split('_(2016')[0], 'in', round(time.time()-start_time, 2), 'seconds')

print('Completed.')
