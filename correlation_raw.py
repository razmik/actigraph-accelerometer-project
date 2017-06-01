import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

# test_filename_wrist = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM_146_Wrist_RAW_test.csv"
# test_filename_waist = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM_146_Waist_RAW_test.csv"
waist_raw_data_filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM 146 Waist (2016-10-19)RAW.csv"
wrist_raw_data_filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM 146 Wrist (2016-10-19)RAW.csv"

# Execution time: 30.083904027938843 seconds for true filename
raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=10, nrows=11000000)
raw_data_waist = pd.read_csv(waist_raw_data_filename, skiprows=10, nrows=11000000)
# raw_data_wrist = pd.read_csv(test_filename_wrist)
# raw_data_waist = pd.read_csv(test_filename_waist)

raw_data_wrist.columns = ['X', 'Y', 'Z']
raw_data_waist.columns = ['X', 'Y', 'Z']

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]
raw_data_waist['vm'] = np.sqrt([(raw_data_waist.X ** 2) + (raw_data_waist.Y ** 2) + (raw_data_waist.Z ** 2)])[0]

# Calculate the angle of arcsin from X and VM, arcsin(axis used/vector magnitude)/(pi/2)
# raw_data_wrist['angle'] = np.arcsin(raw_data_wrist.X/raw_data_wrist['vm']) / (math.pi/2)
# raw_data_waist['angle'] = np.arcsin(raw_data_waist.X/raw_data_waist['vm']) / (math.pi/2)

mx = 'vm' # Wrist | X axis
my = 'vm' # Hip | Y axis

pearson_coefficient = stats.pearsonr(raw_data_wrist[mx], raw_data_waist[my])
print(pearson_coefficient)

seq = np.arange(len(raw_data_wrist[mx]))

plt.figure(1)
plt.subplot(211)
plt.title('Pearson Correlation : ' + str(pearson_coefficient))
plt.xlabel('Wrist Acceleration')
plt.ylabel('Waist Acceleration')
plt.grid(True)
plt.plot(raw_data_wrist[mx], raw_data_waist[my], 'bo')

plt.subplot(212)
plt.title('Acceleration Intensity')
plt.xlabel('Sequence (Red: Waist | Blue: Wrist)')
plt.ylabel('Acceleration (g)')
plt.grid(True)
plt.plot(seq, raw_data_wrist[mx], 'b-', seq, raw_data_waist[my], 'r-')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()
