import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

"""
This file is configured for 15 seconds epochs
"""

experiment = 'LSM1'
week = 'Week 2'
day = 'Thursday'
user = 'LSM101'  # Good - 101, 106, 110
date = '(2016-10-20)'
device = 'Waist'
epoch = 'Epoch15'

epoch_filename = ("E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-ActilifeProcessedEpochs/" + epoch + "/" + experiment + "/" + week + "/"
                  + day + "/processed_1min_ref/" + user + "_" + experiment + "_" + week.replace(' ', '_') +
                   "_" + day + "_" + date + "15s.csv").replace('\\', '/')

epoch_data = pd.read_csv(epoch_filename)

# epoch_data = epoch_data.groupby(epoch_data.index // 10).mean()

x_range = np.arange(len(epoch_data))
normalize_intensity = (epoch_data['waist_intensity'] - np.amin(epoch_data['waist_intensity'])) / (
np.amax(epoch_data['waist_intensity']) - np.amin(epoch_data['waist_intensity']))
normalize_energy_expenditure = (epoch_data['waist_ee'] - np.amin(epoch_data['waist_ee'])) / (
np.amax(epoch_data['waist_ee']) - np.amin(epoch_data['waist_ee']))
normalize_waist_vm_60 = (epoch_data['waist_vm_60'] - np.amin(epoch_data['waist_vm_60'])) / (
np.amax(epoch_data['waist_vm_60']) - np.amin(epoch_data['waist_vm_60']))

print('Results for', epoch_filename.split('ActilifeProcessedEpochs/')[1])
# identify starting points

control_distance = 500

print('Starting points:')

selected_starts = []
def not_near_starts(point):
    for start in selected_starts:
        if abs(point-start) < 800:
            return False
    return True


for i in range(0, len(epoch_data)):
    if normalize_energy_expenditure[i] > 0.2 and i > control_distance and sum(normalize_energy_expenditure[i - control_distance:i]) < 5 and not_near_starts(i):
        selected_starts.append(i)
        print(i)


# identify ending points
print('\nEnding points:')


selected_ends = []
def not_near_end(point):
    for end in selected_ends:
        if abs(point-end) < 800:
            return False
    return True

for i in range(0, len(epoch_data)):
    if normalize_energy_expenditure[i] > 0.2 and i < len(epoch_data) and sum(normalize_energy_expenditure[i + 1:i + control_distance]) < 5 and not_near_end(i):
        selected_ends.append(i)
        print(i)
    elif i == len(epoch_data) - 1:
        print('end:', i)

plt.figure(1)
plt.title('Red - Activity Intensity, Blue - Energy Expenditure, Green - Waist VM Per Min')
plt.plot(x_range, normalize_intensity, 'r', x_range, normalize_energy_expenditure, 'b')
# plt.plot(x_range, normalize_intensity, 'r', x_range, normalize_energy_expenditure, 'b', x_range, normalize_waist_vm_60, 'g')

# plt.imsave('output.jpeg')
plt.show()
