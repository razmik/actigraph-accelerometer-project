from tqdm import tqdm
import pandas as pd
import time
import pickle
import supervised_learning.preprocess.generate_statistical_features as stat_feature_generator
import supervised_learning.preprocess.generate_raw_features as raw_feature_generator


RAW_DATA_ROOT_FOLDER = 'E:/Data/Accelerometer_Dataset_Rashmika/Staff_Activity_Challenege/'
EPOCH_DATA_FOLDER = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-ActilifeProcessedEpochs/'
STAT_FEATURE_OUT_FOLDER = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Statistical_features/'
RAW_FEATURE_OUT_FOLDER = 'E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/'
# TRAIN_TEST_SUBJECT_PICKLE = '../participant_split/train_test_split.pickle'

input_detail_filenames_list = [
    # "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/Processed/wear-time-validation/LSM1_Week1_ActiveTimeline_Details.csv",
    # "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/Processed/wear-time-validation/LSM2_Week1_ActiveTimeline_Details.csv",
    # "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/Processed/wear-time-validation/LSM1_Week2_ActiveTimeline_Details.csv",
    "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/Processed/wear-time-validation/LSM2_Week2_ActiveTimeline_Details.csv"
]

"""NOTE: ONLY ONE EPOCH AT A TIME NOW!"""
TIME_EPOCH_DICT = {
    # 'Epoch1': 100,
    # 'Epoch5': 500,
    # 'Epoch6': 6000,
    # 'Epoch7': 7000,
    # 'Epoch10': 1000,
    # 'Epoch15': 1500,
    # 'Epoch30': 3000,
    'Epoch60': 6000
}
USE_STAT_FEATURES = True

# Test Train Split
# with open(TRAIN_TEST_SUBJECT_PICKLE, 'rb') as handle:
#     split_dict = pickle.load(handle)
# test_subjects = split_dict['test']
# train_subjects = split_dict['train']


if __name__ == "__main__":

    start_time = time.time()

    # Apply the process for experiment LSM1 and LSM2
    for input_detail_filename in input_detail_filenames_list:

        print('Processing {}'.format(input_detail_filename))

        input_details = pd.read_csv(input_detail_filename, usecols=[0, 1, 2, 3, 4, 9, 10])
        input_details.columns = ['experiment', 'week', 'day', 'date', 'subject', 'row_start', 'row_end']

        input_details['clean_date'] = input_details['date'].str.replace('/', '-')
        if input_detail_filename == "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/Processed/wear-time-validation/LSM2_Week2_ActiveTimeline_Details.csv":
            input_details.loc[input_details['clean_date'] == "02-11-16", 'clean_date'] = "02-11-2016"

        try:
            input_details['clean_date'] = pd.to_datetime(input_details['clean_date'], format="%d-%m-%Y")
            print('Use full year - YYYY')
        except:
            input_details['clean_date'] = pd.to_datetime(input_details['clean_date'], format="%d-%m-%y")
            print('Use 2-digit year - yy')


        for index, row in tqdm(input_details.iterrows(), total=input_details.shape[0]):

            experiment = row['experiment']
            week = row['week']
            day = row['day']

            date = '({}-{}-{:02d})'.format(row['clean_date'].year, row['clean_date'].month, row['clean_date'].day)
            user = row['subject'].split(' ')[0]
            device = 'Wrist'

            starting_row = row['row_start']
            end_row = row['row_end']

            exclude_subject_1 = ('LSM1', 'Week 2', 'Thursday', 'LSM138')
            if experiment == exclude_subject_1[0] and week == exclude_subject_1[1] and day == exclude_subject_1[2] and user == exclude_subject_1[3]:
                continue

            if end_row > starting_row > -1 and end_row > -1:

                if USE_STAT_FEATURES:
                    stat_feature_generator.process(RAW_DATA_ROOT_FOLDER, EPOCH_DATA_FOLDER, STAT_FEATURE_OUT_FOLDER,
                                               starting_row, end_row, experiment, week, day, user, date,
                                               TIME_EPOCH_DICT, device=device)
                else:
                    raw_feature_generator.process(RAW_DATA_ROOT_FOLDER, EPOCH_DATA_FOLDER, RAW_FEATURE_OUT_FOLDER,
                                               starting_row, end_row, experiment, week, day, user, date,
                                               TIME_EPOCH_DICT, device=device)

    end_time = time.time()
    total_duration = round(end_time - start_time, 2)
    if total_duration <= 60:
        print('\n\nData processing successfully completed in', total_duration, '(s)')
    elif total_duration <= 3600:
        print('\n\nData processing successfully completed in', (total_duration / 60), '(min)')
    else:
        print('\n\nData processing successfully completed in', (total_duration / 3600), '(hours)')
