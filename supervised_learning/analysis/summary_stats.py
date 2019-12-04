import pandas as pd
from os import listdir
from os.path import join, isfile
from tqdm import tqdm


SUBJECT_DETAIL_FILE = '../../analyze/user_details.csv'
DATA_ROOT_FOLDER = "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/"
DATA_KEYS = ['Week 1', 'Week 2']

if __name__ == '__main__':

    result_data = []
    result_col_names = ['Week', 'Subject', 'Total Time', 'SB Time', 'LPA Time', 'MVPA Time', 'Mean EE']

    for key in DATA_KEYS:

        folder_name = join(DATA_ROOT_FOLDER, key)
        files = [f for f in listdir(folder_name) if isfile(join(folder_name, f)) and '.csv' in f]

        for f in tqdm(files, desc='Processing {}'.format(key)):

            # LSM101 Wrist (2016-10-05)_row-499020_to_538185.csv
            subject = f.split(' ')[0]
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

            # Mean EE
            mean_ee = df['waist_ee'].mean()

            # Construct data row
            row = [key, subject, total_time, time_sb, time_lpa, (time_mpa + time_vpa), mean_ee]

            result_data.append(row)

    # Completed loop

    # Load dataframe
    df_out = pd.DataFrame(result_data, columns=result_col_names)

    df_agg = df_out.groupby(['Week', 'Subject'], as_index=False).agg(
        {'Total Time': 'sum', 'SB Time': 'sum', 'LPA Time': 'sum', 'MVPA Time': 'sum',
         'Mean EE': 'mean'})

    # Merge data with demographics
    subject_df = pd.read_csv(SUBJECT_DETAIL_FILE)
    df_summary = pd.merge(df_agg, subject_df, left_on='Subject', right_on='subject', how='left')

    # Output the file
    df_summary.to_csv('subject_summary.csv', index=None)

    print('Completed')


