import pandas as pd


def get_converted_day(day_str):
    if int(day_str) < 10:
        return '0' + day_str
    else:
        return day_str


input_detail_filenames_list = [
    "D:\Accelerometer Data\Processed/LSM1_ActiveTimeline_Details_v1.csv".replace('\\', '/'),
    "D:\Accelerometer Data\Processed/LSM2_ActiveTimeline_Details_v1.csv".replace('\\', '/')
]
# Apply the process for experiment LSM1 and LSM2
for input_detail_filename in input_detail_filenames_list:

    input_details = pd.read_csv(input_detail_filename, usecols=[4, 11, 20])
    input_details.columns = ['subject', 'active_duration', 'active_duration_with_null']

    result_time = input_details.groupby('subject', as_index=False)['active_duration'].agg({'total_active_time': 'sum'})
    result_days = input_details.groupby('subject', as_index=False)['active_duration_with_null'].agg({'total_active_days': 'count'})

    result_time.to_csv('result_time'+input_detail_filename.split('/')[3], index=False)
    result_days.to_csv('result_days'+input_detail_filename.split('/')[3], index=False)
