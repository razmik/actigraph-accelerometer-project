import pandas as pd
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
import pickle


def create_segments_and_labels(dataframe, time_steps, step, n_features, label_class, label_2):

    segments = []
    labels = []
    regression_values = []
    for i in range(0, len(dataframe) - time_steps, step):
        xs = dataframe['X'].values[i: i + time_steps]
        ys = dataframe['Y'].values[i: i + time_steps]
        zs = dataframe['Z'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        class_label = dataframe[label_class][i: i + time_steps].mode()[0]
        class_reg = dataframe[label_2][i: i + time_steps].mean()
        segments.append([xs, ys, zs])
        labels.append(class_label)
        regression_values.append(class_reg)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, n_features)
    labels = np.asarray(labels)
    regression_values = np.asarray(regression_values)

    return {'segments': reshaped_segments, 'activity_classes': labels, 'energy_e': regression_values}


if __name__ == "__main__":

    TIME_PERIODS_LIST = [3000, 6000]
    N_FEATURES = 3
    LABEL_CLASS = 'waist_intensity'
    LABEL_REG = 'waist_ee'

    GROUPS = ['test', 'train_test']
    WEEKS = {'test': 'Week 2', 'train_test': 'Week 2'}

    req_cols = ['X', 'Y', 'Z', 'waist_ee', 'waist_intensity']
    input_cols = ['X', 'Y', 'Z']
    target_cols = ['waist_ee', 'waist_intensity']

    INPUT_DATA_ROOT = "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1_Combined/"
    OUTPUT_FOLDER_ROOT = join(INPUT_DATA_ROOT, 'model_ready_dec21')
    TRAIN_TEST_SUBJECT_PICKLE = '../participant_split/train_test_split.v2.pickle'

    # Test Train Split
    with open(TRAIN_TEST_SUBJECT_PICKLE, 'rb') as handle:
        split_dict = pickle.load(handle)
    split_dict['train_test'] = split_dict['train'][:]

    for user_group in GROUPS:

        INPUT_DATA_FOLDER = join(INPUT_DATA_ROOT, WEEKS[user_group])
        all_files = [f for f in listdir(INPUT_DATA_FOLDER) if isfile(join(INPUT_DATA_FOLDER, f))]

        for f in tqdm(all_files, desc=user_group):

            if f.split()[0] not in split_dict[user_group]:
                continue

            try:
                df = pd.read_csv(join(INPUT_DATA_FOLDER, f), usecols=req_cols)

                for time_window in TIME_PERIODS_LIST:

                    # No overlap for test data
                    STEP_DISTANCE = time_window

                    reshaped_outcomes = create_segments_and_labels(df, time_window, STEP_DISTANCE,
                                                                   N_FEATURES, LABEL_CLASS, LABEL_REG)

                    OUTPUT_FOLDER = join(OUTPUT_FOLDER_ROOT, 'window-{}-overlap-{}'.format(time_window, int(STEP_DISTANCE/2)),
                                         user_group)
                    if not exists(OUTPUT_FOLDER):
                        makedirs(OUTPUT_FOLDER)
                    out_name = join(OUTPUT_FOLDER, f.replace('.csv', '_test.npy'))
                    np.save(out_name, reshaped_outcomes)

            except Exception as e:
                print('Error loading file {}.\nError: {}'.format(f, e))

    print('Completed.')
