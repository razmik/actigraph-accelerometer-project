import pandas as pd
from os import listdir
from os.path import join, isfile
from tqdm import tqdm
import pickle


SUBJECT_DETAIL_FILE = '../../analyze/user_details.csv'
DATA_ROOT_FOLDER = "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1_Combined/"
WEEKS = {'train': 'Week 1', 'test': 'Week 2', 'train_test': 'Week 2'}
TRAIN_TEST_SUBJECT_PICKLE = '../participant_split/train_test_split.v2.pickle'

# Test Train Split
with open(TRAIN_TEST_SUBJECT_PICKLE, 'rb') as handle:
    split_dict = pickle.load(handle)
split_dict['train_test'] = split_dict['train'][:]

if __name__ == '__main__':

    result_data = []
    result_col_names = ['Group', 'Week', 'Subject', 'Total Time', 'SB Time', 'LPA Time', 'MVPA Time']

    for data_key in WEEKS.keys():

        folder_name = join(DATA_ROOT_FOLDER, WEEKS[data_key])
        files = [f for f in listdir(folder_name) if isfile(join(folder_name, f)) and '.csv' in f]

        counter = 0
        for f in tqdm(files, desc='Processing {}'.format(data_key)):

            # LSM101 Wrist (2016-10-05)_row-499020_to_538185.csv
            subject = f.split(' ')[0]
            if subject not in split_dict[data_key]:
                continue

            from_rec = int(f.split('_')[-3].split("-")[1])
            to_rec = int(f.split('_')[-1].split(".")[0])

            df = pd.read_csv(join(folder_name, f))

            # Total time
            total_time = (to_rec - from_rec) / 60

            # Time in each activity intensity
            time_sb = df.loc[df['waist_intensity'] == 1].shape[0] / (100 * 60)
            time_lpa = df.loc[df['waist_intensity'] == 2].shape[0] / (100 * 60)
            time_mpa = df.loc[df['waist_intensity'] == 3].shape[0] / (100 * 60)
            time_vpa = df.loc[df['waist_intensity'] == 4].shape[0] / (100 * 60)
            time_mvpa = time_mpa + time_vpa

            # Construct data row
            row = [data_key, WEEKS[data_key], subject, total_time, time_sb, time_lpa, time_mvpa]

            result_data.append(row)

            # counter += 1
            # if counter > 10:
            #     break

    # Completed loop

    # Load dataframe
    df_out = pd.DataFrame(result_data, columns=result_col_names)

    df_agg = df_out.groupby(['Group', 'Week', 'Subject'], as_index=False).agg(
        {'Total Time': 'sum', 'SB Time': 'sum', 'LPA Time': 'sum', 'MVPA Time': 'sum'})

    # Merge data with demographics
    subject_df = pd.read_csv(SUBJECT_DETAIL_FILE)
    df_summary = pd.merge(df_agg, subject_df, left_on='Subject', right_on='subject', how='left')

    # Output the file
    df_summary.to_csv('subject_summary_v2.csv', index=None)

    print('Completed')


