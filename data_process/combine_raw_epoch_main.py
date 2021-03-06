import importlib
import numpy as np
import pandas as pd
import sys

# experiment = 'LSM2'
# week = 'Week 1'
# day = 'Wednesday'
# user = 'LSM204'
# date = '(2016-11-02)'
# device = 'Wrist'
#
# starting_row = 58957500
# end_row = 60570000


def get_converted_day(day_str):
    if int(day_str) < 10:
        return '0' + day_str
    else:
        return day_str

filtered_processor = importlib.import_module('combine_raw_and_epoch_files_butterworth_filter')
not_filtered_processor = importlib.import_module('combine_raw_and_epoch_files_no_filter')

input_detail_filename = "D:\Accelerometer Data\Processed/LSM2_ActiveTimeline_Details_v1.csv".replace('\\', '/')

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
    if end_row > starting_row > -1 and end_row > -1 and day == 'Thursday' and user != 'LSM246':
        print("processing not-filtered data")
        not_filtered_processor.process_without_filter(starting_row, end_row, experiment, week, day, user, date,
                                                      device='Wrist')
        print("processing filtered data")
        filtered_processor.process_with_filter(starting_row, end_row, experiment, week, day, user, date, device='Wrist')
    else:
        print("Inactive details for activity intensity.")
    print("Completed processing", week, day, user, date, starting_row, 'to', end_row)

