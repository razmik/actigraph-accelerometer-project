from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
import shutil


if __name__ == "__main__":

    groups = ['LSM1', 'LSM2']
    weeks = ['Week 2']
    days = ['Thursday', 'Wednesday']

    INPUT_ROOT = "E:/Data/Accelerometer_Dataset_Rashmika/Staff_Activity_Challenege/"
    OUTPUT_ROOT = "C:/Users/bnawaratne-admin/Desktop/week2_data/"

    filename_suffix1 = 'Waist'
    filename_suffix2 = '60sec.agd'

    for grp in tqdm(groups):
        for wk in weeks:
            for d in days:

                folder_path = join(INPUT_ROOT, grp, wk, d)
                all_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and filename_suffix1 in f and filename_suffix2 in f]

                for filename in all_files:

                    src = join(folder_path, filename)

                    dest_fld = join(OUTPUT_ROOT, grp, wk, d)
                    if not exists(dest_fld):
                        makedirs(dest_fld)

                    # Error files: LSM 146 Waist (2016-10-19)60sec.agd
                    if filename.split(' ')[0] != 'LSM':
                        dest = join(dest_fld, filename)
                    else:
                        dest = join(dest_fld, filename.replace(' ', '', 1))

                    shutil.copy(src, dest)




