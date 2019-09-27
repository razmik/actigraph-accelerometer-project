import numpy as np
import pandas as pd
import sys, math
import scipy.stats as stats
import matplotlib.pyplot as plt

experiment = 'LSM2'
week = 'Week 1'
day = 'Wednesday'
user = 'LSM203'
date = '(2016-11-02)'
device = 'Wrist'

# wrist_raw_data_filename = "D:/Accelerometer Data/"+experiment+"/"+week+"/"+day+"/"+user+" "+device+" "+date+"RAW.csv".replace('\\', '/')
# epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs/"+experiment+"/"+week+"/"+day+"/processed/"+user+"_"+experiment+"_"+week.replace(' ', '_')+"_"+day+"_"+date+".csv".replace('\\', '/')
epoch_filename = 'D:\Accelerometer Data\Processed\LSM2\Week 1\Wednesday/not_filtered\epoch_15/LSM219_(2016-11-02)_row_16522_to_20063.csv'.replace('\\', '/')
# epoch granularity
n = 1500
starting_row = 0
end_row = -1

# summarize parameters
# summarized_duration = 4 * 60  # 1 hour
# summarized_duration = 4 * 60 * (3 / 6)  # 30 min
# summarized_duration = 4 * 60 * (2 / 6)  # 20 min
# summarized_duration = 4 * 60 * (1 / 6)  # 10 min
# summarized_duration = 4 * 60 * (1 / 12)  # 5 min
summarized_duration = 4 * 60 * (1 / 60)  # 1 min
# summarized_duration = 2                   # 30 sec
# summarized_duration = 1                   # 15 sec

#  row_count = int((end_row - starting_row) / summarize_duration)
# epoch_start = int(starting_row/summarize_duration)

if end_row == -1:
    epoch_data = pd.read_csv(epoch_filename, skiprows=0,
                         usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# else:
    # epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=row_count,
    #                      usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

epoch_data.columns = ['actilife_wrist_Axis1', 'actilife_wrist_Axis2', 'actilife_wrist_Axis3', 'actilife_wrist_vm_15', 'actilife_wrist_vm_60',
                      'actilife_wrist_Axis1_waist_eq', 'actilife_wrist_Axis2_waist_eq', 'actilife_wrist_Axis3_waist_eq',
                      'actilife_wrist_vm_waist_eq', 'actilife_wrist_cpm', 'actilife_wrist_vm_cpm', 'actilife_waist_vm_15',
                      'actilife_waist_vm_60', 'actilife_waist_vm_cpm', 'actilife_waist_cpm', 'actilife_waist_intensity',
                      'actilife_waist_ee', 'actilife_waist_Axis1', 'actilife_waist_Axis2', 'actilife_waist_Axis3']

epoch_data = epoch_data.groupby(epoch_data.index // summarized_duration).mean()


def get_normalized(parameter):
    return (epoch_data[parameter] - np.amin(epoch_data[parameter])) / (np.amax(epoch_data[parameter]) - np.amin(epoch_data[parameter]))


x_range = np.arange(len(epoch_data))
normalized_enrgy = get_normalized('actilife_waist_ee')
normalize_intensity = get_normalized('actilife_waist_intensity')
normalized_wrist_vm = get_normalized('actilife_wrist_vm_15')
normalized_waist_vm = get_normalized('actilife_waist_vm_15')

# plt.figure(1)
# plt.title('Red - Activity Intensity,    Blue - Wrist,    Green - Waist')
# plt.plot(x_range, normalize_intensity, 'r', x_range, normalized_wrist_vm, 'b', x_range, normalized_waist_vm, 'g')

plt.figure(2)
plt.title('Energy Expenditure for 15 sec time periods')
plt.plot(x_range, normalized_enrgy, 'r', x_range, normalized_wrist_vm, 'b', x_range, normalized_waist_vm, 'g')
# plt.xlabel('Red - Actual Energy Expenditure,    Blue - Wrist Acc.,    Green - Waist Acc.')
plt.legend(['Actual Energy Expenditure', 'Wrist Acceleration', 'Hip Acceleration'], loc='upper right')

# plt.savefig('output2.jpg', dpi=1200)
plt.show()
