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
epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs/"+experiment+"/"+week+"/"+day+"/processed/"+user+"_"+experiment+"_"+week.replace(' ', '_')+"_"+day+"_"+date+".csv".replace('\\', '/')

# epoch granularity
n = 1500
starting_row = 0
end_row = 29160010

# summarize parameters
one_hour = 4 * 60 * n  # 1 hour = (15 seconds epoch * 4 * 60)
min_5 = n * 4 * 5
summarize_duration = one_hour * 1
processed_epoch_summarize_duration = summarize_duration / n

timeline = "5 minute epochs"

start = starting_row + 10
row_count = (end_row - starting_row) / summarize_duration
epoch_start = int(starting_row/summarize_duration)

if row_count == -1:
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start, nrows=row_count,
                         usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
else:
    epoch_data = pd.read_csv(epoch_filename, skiprows=epoch_start,
                         usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

epoch_data.columns = ['actilife_wrist_Axis1', 'actilife_wrist_Axis2', 'actilife_wrist_Axis3', 'actilife_wrist_vm_15', 'actilife_wrist_vm_60',
                      'actilife_wrist_Axis1_waist_eq', 'actilife_wrist_Axis2_waist_eq', 'actilife_wrist_Axis3_waist_eq',
                      'actilife_wrist_vm_waist_eq', 'actilife_wrist_cpm', 'actilife_wrist_vm_cpm', 'actilife_waist_vm_15',
                      'actilife_waist_vm_60', 'actilife_waist_vm_cpm', 'actilife_waist_cpm', 'actilife_waist_intensity',
                      'actilife_waist_ee', 'actilife_waist_Axis1', 'actilife_waist_Axis2', 'actilife_waist_Axis3']

epoch_data = epoch_data.groupby(np.arange(len(epoch_data))//processed_epoch_summarize_duration).mean()

x_range = np.arange(len(epoch_data))

normalize_intensity = (epoch_data['actilife_waist_intensity'] - np.amin(epoch_data['actilife_waist_intensity'])) / (np.amax(epoch_data['actilife_waist_intensity']) - np.amin(epoch_data['actilife_waist_intensity']))
normalize_energy_expenditure = (epoch_data['actilife_waist_ee'] - np.amin(epoch_data['actilife_waist_ee'])) / (np.amax(epoch_data['actilife_waist_ee']) - np.amin(epoch_data['actilife_waist_ee']))
normalize_waist_vm_60 = (epoch_data['actilife_waist_vm_60'] - np.amin(epoch_data['actilife_waist_vm_60'])) / (np.amax(epoch_data['actilife_waist_vm_60']) - np.amin(epoch_data['actilife_waist_vm_60']))


plt.figure(1)
plt.xlabel(("Timeline - " + timeline))
plt.title('Red - Activity Intensity, Blue - Energy Expenditure, Green - Waist VM Per Min')
plt.plot(x_range, normalize_intensity, 'r', x_range, normalize_energy_expenditure, 'b', x_range, normalize_waist_vm_60, 'g')

plt.show()
