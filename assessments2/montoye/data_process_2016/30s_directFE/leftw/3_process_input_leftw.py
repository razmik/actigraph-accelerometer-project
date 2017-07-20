from os import listdir
from os.path import isfile, join
import pandas as pd
import sys, time

wrist = 'left_wrist'

output_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Assessment2/Montoye_2016_predictions/'+wrist+'/30sec/input_files/'.replace('\\', '/')
fe_files_path = 'D:\Accelerometer Data\Assessment\montoye\LSM2\Week 1\Wednesday\Epoch30/'.replace('\\', '/')
fe_files = [f for f in listdir(fe_files_path) if isfile(join(fe_files_path, f))]

for file in fe_files:

    start_time = time.time()

    fe_data = pd.read_csv(fe_files_path + file)
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

        result_data['TenPct_GErightX'] = fe_data['X10Perc']
        result_data['TenPct_GErightY'] = fe_data['Y10Perc']
        result_data['TenPct_GErightZ'] = fe_data['Z10Perc']
        result_data['TwentyFivePct_GErightX'] = fe_data['X25Perc']
        result_data['TwentyFivePct_GErightY'] = fe_data['Y25Perc']
        result_data['TwentyFivePct_GErightZ'] = fe_data['Z25Perc']
        result_data['FiftyPct_GErightX'] = fe_data['X50Perc']
        result_data['FiftyPct_GErightY'] = fe_data['Y50Perc']
        result_data['FiftyPct_GErightZ'] = fe_data['Z50Perc']
        result_data['SeventyFivePct_GErightX'] = fe_data['X75Perc']
        result_data['SeventyFivePct_GErightY'] = fe_data['Y75Perc']
        result_data['SeventyFivePct_GErightZ'] = fe_data['Z75Perc']
        result_data['NinetyPct_GErightX'] = fe_data['X90Perc']
        result_data['NinetyPct_GErightY'] = fe_data['Y90Perc']
        result_data['NinetyPct_GErightZ'] = fe_data['Z90Perc']

    result_data.to_csv((output_folder+file.split('.csv')[0]+'.txt'), sep='\t', index=False)

    print('Processed', file.split(' ')[0], 'in', round(time.time()-start_time, 2), 'seconds')

print('Completed.')
