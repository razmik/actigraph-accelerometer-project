import pandas as pd
import time
import os


def get_converted_day(day_str):
    if int(day_str) < 10:
        return '0' + day_str
    else:
        return day_str


selected_users = ['LSM124', 'LSM251', 'LSM255']
root_folder = 'E:/Data/Accelerometer_Processed_Raw_Epoch_Data/'.replace('\\', '/')
input_detail_filenames_list = [
    "D:\Accelerometer Data\Processed/LSM1_ActiveTimeline_Details_v1.csv".replace('\\', '/'),
    "D:\Accelerometer Data\Processed/LSM2_ActiveTimeline_Details_v1.csv".replace('\\', '/')
]
time_epoch_dictionary = {
    'Epoch5': 500,
    'Epoch15': 1500,
    'Epoch30': 3000,
    'Epoch60': 6000
}

file_pointer_dictionary = {
    'Epoch5': {
        'start': 'epoch_start_5s',
        'end': 'epoch_end_5s'
    },
    'Epoch15': {
        'start': 'epoch_start_15s',
        'end': 'epoch_end_15s'
    },
    'Epoch30': {
        'start': 'epoch_start_30s',
        'end': 'epoch_end_30s'
    },
    'Epoch60': {
        'start': 'epoch_start_60s',
        'end': 'epoch_end_60s'
    }
}

start_time = time.time()

# Apply the process for experiment LSM1 and LSM2
for input_detail_filename in input_detail_filenames_list:

    input_details = pd.read_csv(input_detail_filename, usecols=[0, 1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19])
    input_details.columns = ['experiment', 'week', 'day', 'date', 'subject', 'epoch_start_15s', 'epoch_end_15s',
                             'row_start_100hz', 'row_end_100hz', 'epoch_start_5s', 'epoch_end_5s',
                             'epoch_start_30s', 'epoch_end_30s', 'epoch_start_60s', 'epoch_end_60s',
                             'incorrect_row_start_100hz', 'incorrect_row_end_100hz']

    for index, row in input_details.iterrows():

        experiment = row['experiment']
        week = row['week']
        day = row['day']
        date_line = row['date'].split('/')
        date = '(' + date_line[2] + '-' + date_line[1] + '-' + get_converted_day(date_line[0]) + ')'  # 2/11/2016 -> (2016-11-02)
        user = row['subject'].split(' ')[0]
        device = 'Wrist'

        incorrect_starting_row = row['incorrect_row_start_100hz']
        incorrect_end_row = row['incorrect_row_end_100hz']
        starting_row = row['row_start_100hz']
        end_row = row['row_end_100hz']

        if user in selected_users:

            print("\nProcessing", week, day, user, date, starting_row, 'to', end_row)
            if end_row > incorrect_starting_row >= 0 and incorrect_end_row >= 0:

                # Apply the process for each time epoch
                for key, value in time_epoch_dictionary.items():

                    incorrect_filename = root_folder + '/' + experiment + '/' + week + '/' + day + '/' + key + '/' \
                               + user + '_' + date + '_' + 'row_' + str(int(incorrect_starting_row / value)) \
                               + '_to_' + str(int(incorrect_end_row / value)) + '.csv'

                    corrected_file_start = int(starting_row / value)
                    corrected_file_end = int(end_row / value)

                    correct_filename = root_folder + '/' + experiment + '/' + week + '/' + day + '/' + key + '/' \
                                       + user + '_' + date + '_' + 'row_' + str(corrected_file_start) \
                                       + '_to_' + str(corrected_file_end) + '.csv'

                    if os.path.isfile(correct_filename):

                        print('Already exist', correct_filename)

                    elif os.path.isfile(incorrect_filename):

                        if starting_row == 0:
                            continue

                        # read the file as a dataframe
                        df = pd.read_csv(incorrect_filename)
                        # update the dataframe with corrected values
                        last_row_id = int(row[file_pointer_dictionary[key]['end']]) - int(row[file_pointer_dictionary[key]['start']])
                        df = df[:last_row_id]
                        # write the new file
                        df.to_csv(correct_filename, index=None)
                        # delete the incorrect file
                        os.remove(incorrect_filename)

                        print('Update file for', correct_filename)

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
