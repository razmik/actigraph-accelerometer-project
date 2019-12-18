from os import listdir
from os.path import join, isdir
import supervised_learning.modeling.model_train_class as model_train_class
import supervised_learning.modeling.model_train_reg as model_train_reg


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1_Combined\model_ready/'
    all_files = [f for f in listdir(temp_folder) if isdir(join(temp_folder, f)) and (f.split('-')[1] != f.split('-')[3])]

    allowed_windows = [6000, 3000]
    trial_num = 1

    # Run classifier
    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_windows:
            continue

        print('\n\n\n\nProcessing Classification {}'.format(f))
        model_train_class.run(f, trial_num, temp_folder)

    # Run Regressor
    for f in sorted(all_files, reverse=True):

        if int(f.split('-')[1]) not in allowed_windows:
            continue

        print('\n\n\n\nProcessing Regression {}'.format(f))
        model_train_reg.run(f, trial_num, temp_folder)
