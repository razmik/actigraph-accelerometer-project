import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import time

# waist_raw_data_filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM 146 Waist (2016-10-19)RAW.csv"
# wrist_raw_data_filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM 146 Wrist (2016-10-19)RAW.csv"
waist_raw_data_filename = "D:/Accelerometer Data/Sample/LSM216/LSM216 Waist (2016-11-03)RAW.csv"
wrist_raw_data_filename = "D:/Accelerometer Data/Sample/LSM216/LSM216 Waist (2016-11-03)RAW.csv"
waist_epoch_filename = "D:/Accelerometer Data/Sample/LSM216/LSM216 Waist (2016-11-03)15sec.csv"
wrist_epoch_filename = "D:/Accelerometer Data/Sample/LSM216/LSM216 Waist (2016-11-03)15sec.csv"
print(wrist_raw_data_filename)

# epoch granularity
n = 1500
starting_row = 360000*12
end_row = (360000*12) + 360000*3
# end_row = 4000000

start = starting_row + 10
row_count = end_row - starting_row

print("Duration:", ((end_row-starting_row)/(100*3600)), "hours")

start_time = time.time()

epoch_start = int(starting_row/n) + 10
# epoch_raw_count = int(row_count/n) #This needs to be same as aggregated dataframe row count.

raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count)
raw_data_waist = pd.read_csv(waist_raw_data_filename, skiprows=start, nrows=row_count)

raw_data_wrist.columns = ['X', 'Y', 'Z']
raw_data_waist.columns = ['X', 'Y', 'Z']

raw_data_wrist['X'] = raw_data_wrist['X'].abs()
raw_data_wrist['Y'] = raw_data_wrist['Y'].abs()
raw_data_wrist['Z'] = raw_data_wrist['Z'].abs()
raw_data_waist['X'] = raw_data_waist['X'].abs()
raw_data_waist['Y'] = raw_data_waist['Y'].abs()
raw_data_waist['Z'] = raw_data_waist['Z'].abs()

"""
Calculate the statistical inputs (Features)
"""

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]
raw_data_waist['vm'] = np.sqrt([(raw_data_waist.X ** 2) + (raw_data_waist.Y ** 2) + (raw_data_waist.Z ** 2)])[0]

aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_waist = raw_data_waist.groupby(np.arange(len(raw_data_waist))//n).mean()

aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm']
aggregated_waist.columns = ['X', 'Y', 'Z', 'mvm']


def getWristSD(row):
    return (row.vm - aggregated_wrist['mvm'][int(row.name/n)]) ** 2


def getWaistSD(row):
    return (row.vm - aggregated_waist['mvm'][int(row.name/n)]) ** 2

raw_data_wrist['sd'] = raw_data_wrist.apply(getWristSD, axis=1)
raw_data_waist['sd'] = raw_data_waist.apply(getWaistSD, axis=1)

aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_waist = raw_data_waist.groupby(np.arange(len(raw_data_waist))//n).mean()

aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm', 'sdvm']
aggregated_waist.columns = ['X', 'Y', 'Z', 'mvm', 'sdvm']

wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)
waist_grouped_temp = raw_data_waist.groupby(raw_data_waist.index // n)

aggregated_wrist['maxvm'] = wrist_grouped_temp.max()['vm']
aggregated_waist['maxvm'] = waist_grouped_temp.max()['vm']

aggregated_wrist['minvm'] = wrist_grouped_temp.min()['vm']
aggregated_waist['minvm'] = waist_grouped_temp.min()['vm']

aggregated_wrist['10perc'] = wrist_grouped_temp.quantile(.1)['vm']
aggregated_waist['10perc'] = waist_grouped_temp.quantile(.1)['vm']

aggregated_wrist['25perc'] = wrist_grouped_temp.quantile(.25)['vm']
aggregated_waist['25perc'] = waist_grouped_temp.quantile(.25)['vm']

aggregated_wrist['50perc'] = wrist_grouped_temp.quantile(.5)['vm']
aggregated_waist['50perc'] = waist_grouped_temp.quantile(.5)['vm']

aggregated_wrist['75perc'] = wrist_grouped_temp.quantile(.75)['vm']
aggregated_waist['75perc'] = waist_grouped_temp.quantile(.75)['vm']

aggregated_wrist['90perc'] = wrist_grouped_temp.quantile(.9)['vm']
aggregated_waist['90perc'] = waist_grouped_temp.quantile(.9)['vm']

"""
Include the epoch counts for 15 seconds and CPM values in aggregated dataframe
"""
epoch_data_waist = pd.read_csv(waist_epoch_filename, skiprows=epoch_start, nrows=len(aggregated_waist), usecols=[0, 1, 2, 3, 4, 5])
epoch_data_waist.columns = ['X', 'Y', 'Z', 'Counts', 'CPM', 'Intensity']

# normalize epoch counts
epoch_data_waist['Counts'] = epoch_data_waist['Counts'].apply(lambda x: x/1000)

end_time = time.time()

print("Process time:", (end_time-start_time), "seconds")

# print(aggregated_wrist)
# print(epoch_data_waist)

"""
 Plot the results
"""
# print("Pearson MVM for Hip vs Wrist:", round(stats.pearsonr(aggregated_wrist['mvm'], aggregated_waist['mvm'])[0], 2))
# print("Pearson SDVM for Hip vs Wrist:", round(stats.pearsonr(aggregated_wrist['sdvm'], aggregated_waist['sdvm'])[0], 2))
# print("Pearson Hip VM vs Hip Counts:", round(stats.pearsonr(aggregated_waist['mvm'], epoch_data_waist['Counts'])[0], 2))
print("Pearson Wrist VM vs Hip Counts:", round(stats.pearsonr(aggregated_wrist['mvm'], epoch_data_waist['Counts'])[0], 2))
# print("Pearson Hip SDVM vs Hip Counts:", round(stats.pearsonr(aggregated_waist['sdvm'], epoch_data_waist['Counts'])[0], 2))
print("Pearson Wrist SDVM vs Hip Counts:", round(stats.pearsonr(aggregated_wrist['sdvm'], epoch_data_waist['Counts'])[0], 2))
# print("Pearson Hip Min vs Hip Counts:", round(stats.pearsonr(aggregated_waist['minvm'], epoch_data_waist['Counts'])[0], 2))
print("Pearson Wrist Min vs Hip Counts:", round(stats.pearsonr(aggregated_wrist['minvm'], epoch_data_waist['Counts'])[0], 2))
# print("Pearson Hip Max vs Hip Counts:", round(stats.pearsonr(aggregated_waist['maxvm'], epoch_data_waist['Counts'])[0], 2))
print("Pearson Wrist Max vs Hip Counts:", round(stats.pearsonr(aggregated_wrist['maxvm'], epoch_data_waist['Counts'])[0], 2))
# print("Pearson Hip 10perc vs Hip Counts:", round(stats.pearsonr(aggregated_waist['10perc'], epoch_data_waist['Counts'])[0], 2))
print("Pearson Wrist 10perc vs Hip Counts:", round(stats.pearsonr(aggregated_wrist['10perc'], epoch_data_waist['Counts'])[0], 2))
# print("Pearson Hip 25perc vs Hip Counts:", round(stats.pearsonr(aggregated_waist['25perc'], epoch_data_waist['Counts'])[0], 2))
print("Pearson Wrist 25perc vs Hip Counts:", round(stats.pearsonr(aggregated_wrist['25perc'], epoch_data_waist['Counts'])[0], 2))
# print("Pearson Hip 50perc vs Hip Counts:", round(stats.pearsonr(aggregated_waist['50perc'], epoch_data_waist['Counts'])[0], 2))
print("Pearson Wrist 50perc vs Hip Counts:", round(stats.pearsonr(aggregated_wrist['50perc'], epoch_data_waist['Counts'])[0], 2))
# print("Pearson Hip 75perc vs Hip Counts:", round(stats.pearsonr(aggregated_waist['75perc'], epoch_data_waist['Counts'])[0], 2))
print("Pearson Wrist 75perc vs Hip Counts:", round(stats.pearsonr(aggregated_wrist['75perc'], epoch_data_waist['Counts'])[0], 2))
# print("Pearson Hip 90perc vs Hip Counts:", round(stats.pearsonr(aggregated_waist['90perc'], epoch_data_waist['Counts'])[0], 2))
print("Pearson Wrist 90perc vs Hip Counts:", round(stats.pearsonr(aggregated_wrist['90perc'], epoch_data_waist['Counts'])[0], 2))
# print("Pearson Hip VM vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_waist['mvm'], epoch_data_waist['Intensity'])[0], 2))
print("Pearson Wrist VM vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_wrist['mvm'], epoch_data_waist['Intensity'])[0], 2))
# print("Pearson Hip SDVM vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_waist['sdvm'], epoch_data_waist['Intensity'])[0], 2))
print("Pearson Wrist SDVM vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_wrist['sdvm'], epoch_data_waist['Intensity'])[0], 2))
# print("Pearson Hip Min vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_waist['minvm'], epoch_data_waist['Intensity'])[0], 2))
print("Pearson Wrist Min vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_wrist['minvm'], epoch_data_waist['Intensity'])[0], 2))
# print("Pearson Hip Max vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_waist['maxvm'], epoch_data_waist['Intensity'])[0], 2))
print("Pearson Wrist Max vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_wrist['maxvm'], epoch_data_waist['Intensity'])[0], 2))
# print("Pearson Hip 10perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_waist['10perc'], epoch_data_waist['Intensity'])[0], 2))
print("Pearson Wrist 10perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_wrist['10perc'], epoch_data_waist['Intensity'])[0], 2))
# print("Pearson Hip 25perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_waist['25perc'], epoch_data_waist['Intensity'])[0], 2))
print("Pearson Wrist 25perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_wrist['25perc'], epoch_data_waist['Intensity'])[0], 2))
# print("Pearson Hip 50perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_waist['50perc'], epoch_data_waist['Intensity'])[0], 2))
print("Pearson Wrist 50perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_wrist['50perc'], epoch_data_waist['Intensity'])[0], 2))
# print("Pearson Hip 75perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_waist['75perc'], epoch_data_waist['Intensity'])[0], 2))
print("Pearson Wrist 75perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_wrist['75perc'], epoch_data_waist['Intensity'])[0], 2))
# print("Pearson Hip 90perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_waist['90perc'], epoch_data_waist['Intensity'])[0], 2))
print("Pearson Wrist 90perc vs Activity Intensity (Based on Hip):", round(stats.pearsonr(aggregated_wrist['90perc'], epoch_data_waist['Intensity'])[0], 2))

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