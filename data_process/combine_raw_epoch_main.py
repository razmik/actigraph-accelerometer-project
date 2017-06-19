import importlib

filtered_processor = importlib.import_module('combine_raw_and_epoch_files_butterworth_filter')
not_filtered_processor = importlib.import_module('combine_raw_and_epoch_files_no_filter')

experiment = 'LSM2'
week = 'Week 1'
day = 'Wednesday'
user = 'LSM204'
date = '(2016-11-02)'
device = 'Wrist'

starting_row = 58957500
end_row = 60570000

if end_row > starting_row:
    print("processing not-filtered data")
    not_filtered_processor.process_without_filter(starting_row, end_row, experiment, week, day, user, date, device='Wrist')
    print("processing filtered data")
    filtered_processor.process_with_filter(starting_row, end_row, experiment, week, day, user, date, device='Wrist')
    print("Completed.")
else:
    print("Incorrect row numbers. Starting row should be less than end row.")
