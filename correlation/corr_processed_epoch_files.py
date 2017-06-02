import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import time

input_file = "D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/LSM255_(2016-11-01)_row_1800000_to_3600000.csv"

epoch_data = pd.read_csv(input_file, skiprows=0)

"""
Index(['Unnamed: 0', 'wrist_vm_15', 'wrist_vm_60', 'waist_eq_wrist_vm_60',
       'waist_vm_60', 'waist_intensity', 'wrist_mvm', 'wrist_sdvm',
       'wrist_maxvm', 'wrist_minvm', 'wrist_10perc', 'wrist_25perc',
       'wrist_50perc', 'wrist_75perc', 'wrist_90perc'],
      dtype='object')
"""

"""
 Plot the results
"""
print("Pearson Wrist Processed VM vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_vm_60'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist Processed VM Equal vs Hip Counts:", round(stats.pearsonr(epoch_data['waist_eq_wrist_vm_60'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist Raw VM vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_mvm'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist Raw SDVM vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_sdvm'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist Min vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_minvm'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist Max vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_maxvm'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist 10perc vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_10perc'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist 25perc vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_25perc'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist 50perc vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_50perc'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist 75perc vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_75perc'], epoch_data['waist_vm_60'])[0], 2))
print("Pearson Wrist 90perc vs Hip Counts:", round(stats.pearsonr(epoch_data['wrist_90perc'], epoch_data['waist_vm_60'])[0], 2))


print("Pearson Wrist Processed VM vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_vm_60'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist Processed VM Equal vs Activity Intensity:", round(stats.pearsonr(epoch_data['waist_eq_wrist_vm_60'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist Raw VM vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_mvm'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist Raw SDVM vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_sdvm'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist Min vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_minvm'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist Max vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_maxvm'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist 10perc vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_10perc'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist 25perc vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_25perc'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist 50perc vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_50perc'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist 75perc vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_75perc'], epoch_data['waist_intensity'])[0], 2))
print("Pearson Wrist 90perc vs Activity Intensity:", round(stats.pearsonr(epoch_data['wrist_90perc'], epoch_data['waist_intensity'])[0], 2))

sys.exit(0)

seq = np.arange(len(aggregated_wrist['mvm']))

plt.figure(1)
plt.title("15s Epoch MVM")
plt.grid(True)
plt.xlabel("Blue: Wrist MVM | Red: Hip MVM")
plt.plot(seq, aggregated_wrist['mvm'], 'b-', seq, aggregated_waist['mvm'], 'r-')

plt.figure(2)
plt.title("15s Epoch SDVM")
plt.grid(True)
plt.xlabel("Blue: Wrist SDVM | Red: Hip SDVM")
plt.plot(seq, aggregated_wrist['sdvm'], 'b-', seq, aggregated_waist['sdvm'], 'r-')

plt.figure(3)
plt.title("15s Hip VM - Hip Counts")
plt.grid(True)
plt.xlabel("Blue: Wrist MVM | Red: Hip Counts")
plt.plot(seq, aggregated_waist['mvm'], 'b-', seq, epoch_data_waist['Counts'], 'r-')

plt.figure(4)
plt.title("15s Wrist VM - Hip Counts")
plt.grid(True)
plt.xlabel("Blue: Wrist MVM | Red: Hip Counts")
plt.plot(seq, aggregated_waist['mvm'], 'b-', seq, epoch_data_waist['Counts'], 'r-')

plt.figure(5)
plt.title("15s Hip SDVM - Hip Counts")
plt.grid(True)
plt.xlabel("Blue: Wrist SDVM | Red: Hip Counts")
plt.plot(seq, aggregated_waist['sdvm'], 'b-', seq, epoch_data_waist['Counts'], 'r-')

plt.figure(6)
plt.title("15s Wrist SDVM - Hip Counts")
plt.grid(True)
plt.xlabel("Blue: Wrist SDVM | Red: Hip Counts")
plt.plot(seq, aggregated_wrist['sdvm'], 'b-', seq, epoch_data_waist['Counts'], 'r-')

plt.figure(7)
plt.title("15s Hip VM vs Activity Intensity (Based on Hip)")
plt.grid(True)
plt.xlabel("Blue: Wrist MVM | Red: Activity Intensity")
plt.plot(seq, aggregated_waist['mvm'], 'b-', seq, epoch_data_waist['Intensity'], 'r-')

plt.figure(8)
plt.title("15s Hip SDVM vs Activity Intensity (Based on Hip)")
plt.grid(True)
plt.xlabel("Blue: Wrist SDVM | Red: Activity Intensity")
plt.plot(seq, aggregated_waist['sdvm'], 'b-', seq, epoch_data_waist['Intensity'], 'r-')

plt.figure(9)
plt.title("15s Wrist SDVM - Activity Intensity")
plt.grid(True)
plt.xlabel("Blue: Wrist SDVM | Red: Activity Intensity")
plt.plot(seq, aggregated_wrist['sdvm'], 'b-', seq, epoch_data_waist['Intensity'], 'r-')

seq = np.arange(len(raw_data_wrist['vm']))

plt.figure(10)
plt.title("100 Hz Raw - VM")
plt.grid(True)
plt.xlabel("Blue: Wrist Raw VM | Red: Hip Raw VM")
plt.plot(seq, raw_data_wrist['vm'], 'b-', seq, raw_data_waist['vm'], 'r-')

plt.show()