import importlib
import pandas as pd
import time
import sys


def get_converted_day(day_str):
    if int(day_str) < 10:
        return '0' + day_str
    else:
        return day_str


not_filtered_processor = importlib.import_module('private_combine_raw_and_epoch_files_no_filter')
input_detail_filenames_list = [
    "D:\Accelerometer Data\Processed/LSM1_ActiveTimeline_Details_v1.csv".replace('\\', '/'),
    "D:\Accelerometer Data\Processed/LSM2_ActiveTimeline_Details_v1.csv".replace('\\', '/')
]
time_epoch_dictionary = {
    'Epoch1': 100
    # 'Epoch5': 500,
    # 'Epoch15': 1500,
    # 'Epoch30': 3000,
    # 'Epoch60': 6000
}

start_time = time.time()

# Apply the process for experiment LSM1 and LSM2
for input_detail_filename in input_detail_filenames_list:

    input_details = pd.read_csv(input_detail_filename, usecols=[0, 1, 2, 3, 4, 7, 8, 9, 10])
    input_details.columns = ['experiment', 'week', 'day', 'date', 'subject', 'epoch_start', 'epoch_end', 'row_start', 'row_end']

    for index, row in input_details.iterrows():

        experiment = row['experiment']
        week = row['week']
        day = row['day']
        date_line = row['date'].split('/')
        date = '(' + date_line[2] + '-' + date_line[1] + '-' + get_converted_day(date_line[0]) + ')'  # 2/11/2016 -> (2016-11-02)
        user = row['subject'].split(' ')[0]
        device = 'Wrist'

        starting_row = row['row_start']
        end_row = row['row_end']

        print("\nProcessing", week, day, user, date, starting_row, 'to', end_row)
        if end_row > starting_row > -1 and end_row > -1:

            # Apply the process for each time epoch
            for key, value in time_epoch_dictionary.items():
                not_filtered_processor.process_without_filter(starting_row, end_row, experiment, week, day, user,
                                                              date, key, value, device=device)
                # print(experiment, week, day, user, date, key, value, starting_row, end_row)
        else:
            print("Inactive details for activity intensity.")
        print("Completed processing", week, day, user, date, starting_row, 'to', end_row)


end_time = time.time()
total_duration = round(end_time - start_time, 2)
if total_duration <= 60:
    print('\n\nData processing successfully completed in', total_duration, '(s)')
elif total_duration <= 3600:
    print('\n\nData processing successfully completed in', (total_duration / 60), '(min)')
else:
    print('\n\nData processing successfully completed in', (total_duration / 3600), '(hours)')
