import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


# test_filename_wrist = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM_146_Wrist_RAW_test.csv"
# test_filename_waist = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM_146_Waist_RAW_test.csv"
# waist_raw_data_filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM 146 Waist (2016-10-19)RAW.csv"
# wrist_raw_data_filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM 146 Wrist (2016-10-19)RAW.csv"
waist_raw_data_filename = "D:/Accelerometer Data/LSM2/Week 2/Wednesday/LSM221 Waist (2016-11-16)RAW.csv"
wrist_raw_data_filename = "D:/Accelerometer Data/LSM2/Week 2/Wednesday/LSM221 Wrist (2016-11-16)RAW.csv"
print(wrist_raw_data_filename)
# Execution time: 30.083904027938843 seconds for true filename
# 10 hours = 3600000 records
raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=10, nrows=360000)
raw_data_waist = pd.read_csv(waist_raw_data_filename, skiprows=10, nrows=360000)
# raw_data_wrist = pd.read_csv(test_filename_wrist)
# raw_data_waist = pd.read_csv(test_filename_waist)

raw_data_wrist.columns = ['X', 'Y', 'Z']
raw_data_waist.columns = ['X', 'Y', 'Z']

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]
raw_data_waist['vm'] = np.sqrt([(raw_data_waist.X ** 2) + (raw_data_waist.Y ** 2) + (raw_data_waist.Z ** 2)])[0]

# # Calculate the angle of arcsin from X and VM, arcsin(axis used/vector magnitude)/(pi/2)
# raw_data_wrist['angle'] = np.arcsin(raw_data_wrist.X/raw_data_wrist['vm']) / (math.pi/2)
# raw_data_waist['angle'] = np.arcsin(raw_data_waist.X/raw_data_waist['vm']) / (math.pi/2)

n = 1500 # epoch granularity

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


# def getWristSDAngle(row):
#     return (row.angle - aggregated_wrist['mangle'][int(row.name/n)]) ** 2
#
#
# def getWaistSDAngle(row):
#     return (row.angle - aggregated_waist['mangle'][int(row.name/n)]) ** 2
#
# raw_data_wrist['sdangle'] = raw_data_wrist.apply(getWristSDAngle, axis=1)
# raw_data_waist['sdangle'] = raw_data_waist.apply(getWaistSDAngle, axis=1)


aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_waist = raw_data_waist.groupby(np.arange(len(raw_data_waist))//n).mean()

aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm', 'sdvm']
aggregated_waist.columns = ['X', 'Y', 'Z', 'mvm', 'sdvm']

"""
Correlation plot of Mean Vector Magnitude
"""

mx = 'mvm' # Wrist | X axis
my = 'mvm' # Hip | Y axis

pearson_coefficient = stats.pearsonr(aggregated_wrist[mx], aggregated_waist[my])
print("Pearson MVM: ", str(pearson_coefficient))

seq = np.arange(len(aggregated_wrist[mx]))

plt.figure(1)
plt.subplot(211)
plt.title('Pearson Correlation MVM : ' + str(pearson_coefficient))
plt.xlabel('Wrist MVM')
plt.ylabel('Waist MVM')
plt.grid(True)
plt.plot(aggregated_wrist[mx], aggregated_waist[my], 'bo')

plt.subplot(212)
plt.title('MVM Intensity')
plt.xlabel('Sequence (Red: Waist | Blue: Wrist)')
plt.ylabel('Acc. Vector Magnitude (g)')
plt.grid(True)
plt.plot(seq, aggregated_wrist[mx], 'b-', seq, aggregated_waist[my], 'r-')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

"""
Correlation plot of Mean SDVM
"""

mx = 'sdvm' # Wrist | X axis
my = 'sdvm' # Hip | Y axis

pearson_coefficient = stats.pearsonr(aggregated_wrist[mx], aggregated_waist[my])
print("Pearson SDVM: ", str(pearson_coefficient))

seq = np.arange(len(aggregated_wrist[mx]))

plt.figure(2)
plt.subplot(211)
plt.title('Pearson Correlation SDVM : ' + str(pearson_coefficient))
plt.xlabel('Wrist SDVM')
plt.ylabel('Waist SDVM')
plt.grid(True)
plt.plot(aggregated_wrist[mx], aggregated_waist[my], 'bo')

plt.subplot(212)
plt.title('SDVM Intensity')
plt.xlabel('Sequence (Red: Waist | Blue: Wrist)')
plt.ylabel('SD of VM (g)')
plt.grid(True)
plt.plot(seq, aggregated_wrist[mx], 'b-', seq, aggregated_waist[my], 'r-')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

# (Wrist, Waist)
req_axes = [['X', 'X'], ['Y', 'Y'], ['Z', 'Z'], ['X', 'Y'], ['X', 'Z'], ['Y', 'Z'], ['Y', 'X'], ['Z', 'X'], ['Z', 'Y']]

i = 3
for axes in req_axes:
    pearson_coefficient = stats.pearsonr(aggregated_wrist[axes[0]], aggregated_waist[axes[1]])
    print("Pearson "+str(axes)+": " + str(pearson_coefficient))

    seq = np.arange(len(aggregated_wrist[axes[1]]))

    plt.figure(i)
    plt.subplot(211)
    plt.title('Pearson Correlation '+str(axes)+' : ' + str(pearson_coefficient))
    plt.xlabel('Wrist')
    plt.ylabel('Waist')
    plt.grid(True)
    plt.plot(aggregated_wrist[axes[0]], aggregated_waist[axes[1]], 'bo')

    plt.subplot(212)
    plt.title('Acceleration Intensity')
    plt.xlabel('Sequence (Red: Waist | Blue: Wrist)')
    plt.ylabel('Acceleration (g)')
    plt.grid(True)
    plt.plot(seq, aggregated_wrist[axes[0]], 'b-', seq, aggregated_waist[axes[1]], 'r-')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)

    i += 1

# Show all figures
plt.show()


# The statistic Angle Does not make any sense when comparing with the waist because it was a stat related to
# the positioning of the wrist
#
# """
# Correlation plot of Mean Angle
# """
#
# mx = 'mangle' # Wrist | X axis
# my = 'mangle' # Hip | Y axis
#
# pearson_coefficient = stats.pearsonr(aggregated_wrist[mx], aggregated_waist[my])
# print("Pearson Mangle: ", str(pearson_coefficient))
#
# seq = np.arange(len(aggregated_wrist[mx]))
#
# plt.figure(3)
# plt.subplot(211)
# plt.title('Pearson Correlation Mean Angle : ' + str(pearson_coefficient))
# plt.xlabel('Wrist mangle')
# plt.ylabel('Waist mangle')
# plt.grid(True)
# plt.plot(aggregated_wrist[mx], aggregated_waist[my], 'bo')
#
# plt.subplot(212)
# plt.title('Mean Angle Intensity')
# plt.xlabel('Sequence (Red: Waist | Blue: Wrist)')
# plt.ylabel('Mean Angle (g)')
# plt.grid(True)
# plt.plot(seq, aggregated_wrist[mx], 'b-', seq, aggregated_waist[my], 'r-')
#
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
#                     wspace=0.35)
# """
# Correlation plot of Mean SD Angle
# """
#
#
# mx = 'sdangle' # Wrist | X axis
# my = 'sdangle' # Hip | Y axis
#
# pearson_coefficient = stats.pearsonr(aggregated_wrist[mx], aggregated_waist[my])
# print("Pearson sdangle: ", str(pearson_coefficient))
#
# seq = np.arange(len(aggregated_wrist[mx]))
#
# plt.figure(4)
# plt.subplot(211)
# plt.title('Pearson Correlation SD of Mean Angle : ' + str(pearson_coefficient))
# plt.xlabel('Wrist mangle')
# plt.ylabel('Waist mangle')
# plt.grid(True)
# plt.plot(aggregated_wrist[mx], aggregated_waist[my], 'bo')
#
# plt.subplot(212)
# plt.title('SD of Mean Angle Intensity')
# plt.xlabel('Sequence (Red: Waist | Blue: Wrist)')
# plt.ylabel('SD of Mean Angle (g)')
# plt.grid(True)
# plt.plot(seq, aggregated_wrist[mx], 'b-', seq, aggregated_waist[my], 'r-')
#
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
#                     wspace=0.35)
#

