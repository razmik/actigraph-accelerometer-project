from os import listdir, makedirs
from os.path import join, isfile, exists, isdir
import supervised_learning.modeling.model_train_class as model_train_class
import supervised_learning.modeling.model_train_reg as model_train_reg


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/Week 1/supervised_data/'
    all_files = [f for f in listdir(temp_folder) if isdir(join(temp_folder, f))]

    allowed_windows = [1500, 3000, 6000]
    trial_num = 4

    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_windows:
            continue

        print('\n\n\n\nProcessing Classification {}'.format(f))
        model_train_class.run(f, trial_num)


    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_windows:
            continue

        print('\n\n\n\nProcessing Regression {}'.format(f))
        model_train_reg.run(f, trial_num)
