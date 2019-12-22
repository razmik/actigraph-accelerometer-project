from os import listdir
from os.path import join, isdir
import itertools
import time
import supervised_learning.modeling.mod_evaluate.eval_class as model_eval_class
import supervised_learning.modeling.mod_evaluate.eval_regress as model_eval_reg


if __name__ == '__main__':

    # Get folder names
    temp_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1_Combined\model_ready/'
    all_files = [f for f in listdir(temp_folder) if isdir(join(temp_folder, f)) and (f.split('-')[1] != f.split('-')[3])]

    training_version = '2-13_Dec'
    allowed_list = [3000, 6000]
    groups = ['train_test', 'test']

    start_time = time.time()

    # Run classifier
    for f, grp in itertools.product(all_files, groups):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\nProcessing Classification on {} for {}'.format(f, grp))
        cl_begin = time.time()
        model_eval_class.run(f, training_version, grp, temp_folder, demo=False)
        print('\nClassification on {} for {} completed in {:.3f} mins.'.format(f, grp, (time.time() - cl_begin)/60))

    # Run Regressor
    for f, grp in itertools.product(all_files, groups):

        if int(f.split('-')[1]) not in allowed_list:
            continue

        print('\nProcessing Regression {} for {}'.format(f, grp))
        rg_begin = time.time()
        model_eval_reg.run(f, training_version, grp, temp_folder, demo=False)
        print('\nRegression on {} for {} completed in {:.3f} mins.'.format(f, grp, (time.time() - rg_begin)/60))

    print('Completed all in {:.3f} minutes.'.format((time.time() - start_time)/60))
