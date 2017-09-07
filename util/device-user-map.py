from os import listdir, rename
from os.path import isfile, join
import pandas as pd
import sys, time


file_folder = 'D:\Accelerometer Data\LSM1\Week 1\Sample/'.replace('\\', '/')
code_file = 'D:\Accelerometer Data\LSM1\device-user-map.csv'.replace('\\', '/')

files = [f for f in listdir(file_folder) if isfile(join(file_folder, f))]

"""
Create the local mapping for codes, (device id <-> user)
id	participant_code
MOS2E33154168	LSM131 WAIST
MOS2E33154170	LSM140
MOS2E33154173	LSM149
MOS2E33154174	LSM105 WAIST
"""
codes = {}
codes_df = pd.read_csv(code_file)
for index, row in codes_df.iterrows():

    if 'WAIST' in row['participant_code']:
        codes[row['id']] = row['participant_code'].split(' ')[0] + ' Waist'
    else:
        codes[row['id']] = row['participant_code'] + ' Wrist'

"""
Update filenames
"""
for file in files:

    current_name_split = file.split(' ')
    new_filename = codes[current_name_split[0]] + ' ' + current_name_split[1]
    print('In:', file_folder+file)
    rename(file_folder+file, file_folder+new_filename)
    print('Out:', file_folder+new_filename, '\n')

print('Processing completed.')
