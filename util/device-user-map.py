from os import listdir, rename
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm
from shutil import copyfile



"""
Create the local mapping for codes, (device id <-> user)
id	participant_code
MOS2E33154168	LSM131 WAIST
MOS2E33154170	LSM140
MOS2E33154173	LSM149
MOS2E33154174	LSM105 WAIST
"""
def get_codes(code_filename):
    codes = {}
    codes_df = pd.read_csv(code_filename)
    for index, row in codes_df.iterrows():

        if 'WAIST' in row['participant_code']:
            codes[row['id']] = row['participant_code'].split(' ')[0] + ' Waist'
        else:
            codes[row['id']] = row['participant_code'] + ' Wrist'
    return codes


def update_filenames(file_list, file_folder, codes_dict):
    """
    Update file names
    """
    for file in file_list:
        current_name_split = file.split(' ')
        new_filename = codes_dict[current_name_split[0]] + ' ' + current_name_split[1]
        print('In:', file_folder + file)
        rename(file_folder + file, file_folder + new_filename)
        print('Out:', file_folder + new_filename, '\n')

    print('Processing completed.')


if __name__ == '__main__':

    code_file = 'E:/Data/Accelerometer_Dataset_Rashmika/Staff_Activity_Challenege/LSM1/device-user-map.csv'
    codes = get_codes(code_file)

    raw_files_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/Staff_Activity_Challenege/LSM1/Week 1/Thursday'
    raw_file_names = [f for f in listdir(raw_files_folder) if 'RAW.csv' in f]

    # copy new files
    for filename in tqdm(raw_file_names):

        user_id = filename.split(' ')[0]
        suffix = filename.split(' ')[1]

        updated_user_id = codes[user_id]

        src = join(raw_files_folder, filename)
        dst = join(raw_files_folder, '{} {}'.format(updated_user_id, suffix))

        copyfile(src, dst)

        # print(src, '\n', dst, '\n\n')

    print('Completed.')


#
# """
# Config
# """
# # file_folder_name = 'D:\Accelerometer Data\LSM1\Week 1\Ian/'.replace('\\', '/')
# code_file = 'E:/Data/Accelerometer_Dataset_Rashmika/Staff_Activity_Challenege/LSM1/device-user-map.csv'
# # files = [f for f in listdir(file_folder) if isfile(join(file_folder, f))]
#
# filename = 'D:\Accelerometer Data\LSM1\Week 1\Thursday Outputs/Thursday Outputs_DailyDetailed_testfile.csv'.replace('\\', '/')
# output_filename = 'D:\Accelerometer Data\LSM1\Week 1\Thursday Outputs/Thursday Outputs_DailyDetailed_keys.csv'.replace('\\', '/')
#
# codes = get_codes(code_file)
#
# dataframe = pd.read_csv(filename)
# dataframe['username'] = ''
# for index, row in dataframe.iterrows():
#     row['username'] = codes[row['Subject']]
#
# dataframe.to_csv(output_filename)





