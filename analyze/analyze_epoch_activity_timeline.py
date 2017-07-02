import numpy as np
import pandas as pd
import sys, math
import scipy.stats as stats
import matplotlib.pyplot as plt

experiment = 'LSM2'
week = 'Week 2'
day = 'Wednesday'
user = 'LSM246'
date = '(2016-11-16)'
device = 'Wrist'

epoch_filename = "D:\Accelerometer Data\ActilifeProcessedEpochs/"+experiment+"/"+week+"/"+day+"/processed/"+user+"_"+experiment+"_"+week.replace(' ', '_')+"_"+day+"_"+date+".csv".replace('\\', '/')

epoch_data = pd.read_csv(epoch_filename, skiprows=0, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

epoch_data.columns = ['actilife_wrist_Axis1', 'actilife_wrist_Axis2', 'actilife_wrist_Axis3', 'actilife_wrist_vm_15', 'actilife_wrist_vm_60',
                      'actilife_wrist_Axis1_waist_eq', 'actilife_wrist_Axis2_waist_eq', 'actilife_wrist_Axis3_waist_eq',
                      'actilife_wrist_vm_waist_eq', 'actilife_wrist_cpm', 'actilife_wrist_vm_cpm', 'actilife_waist_vm_15',
                      'actilife_waist_vm_60', 'actilife_waist_vm_cpm', 'actilife_waist_cpm', 'actilife_waist_intensity',
                      'actilife_waist_ee', 'actilife_waist_Axis1', 'actilife_waist_Axis2', 'actilife_waist_Axis3']

# epoch_data = epoch_data.groupby(epoch_data.index // 10).mean()

x_range = np.arange(len(epoch_data))
normalize_intensity = (epoch_data['actilife_waist_intensity'] - np.amin(epoch_data['actilife_waist_intensity'])) / (np.amax(epoch_data['actilife_waist_intensity']) - np.amin(epoch_data['actilife_waist_intensity']))
normalize_energy_expenditure = (epoch_data['actilife_waist_ee'] - np.amin(epoch_data['actilife_waist_ee'])) / (np.amax(epoch_data['actilife_waist_ee']) - np.amin(epoch_data['actilife_waist_ee']))
normalize_waist_vm_60 = (epoch_data['actilife_waist_vm_60'] - np.amin(epoch_data['actilife_waist_vm_60'])) / (np.amax(epoch_data['actilife_waist_vm_60']) - np.amin(epoch_data['actilife_waist_vm_60']))

print('Results for', epoch_filename.split('ActilifeProcessedEpochs/')[1])
# identify starting points
print('Starting points:')
for i in range(0, len(epoch_data)):
    if normalize_intensity[i] > 0.2 and i > 1000 and sum(normalize_intensity[i-1000:i]) == 0:
        print(i)

# identify ending points
print('\nEnding points:')
for i in range(0, len(epoch_data)):
    if normalize_intensity[i] > 0.2 and i < len(epoch_data) and sum(normalize_intensity[i+1:i+1000]) == 0:
        print(i)
    elif i == len(epoch_data)-1:
        print('end:', i)

plt.figure(1)
plt.title('Red - Activity Intensity, Blue - Energy Expenditure, Green - Waist VM Per Min')
plt.plot(x_range, normalize_intensity, 'r', x_range, normalize_energy_expenditure, 'b')
# plt.plot(x_range, normalize_intensity, 'r', x_range, normalize_energy_expenditure, 'b', x_range, normalize_waist_vm_60, 'g')

plt.show()
