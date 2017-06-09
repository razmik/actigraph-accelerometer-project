import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys

test_filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM_146_Wrist_RAW_test.csv"
filename = "E:/Accelerometry project/Dataset/Example Accelerometry Files/LSM 146 Waist (2016-10-19)RAW.csv"

start_read = time.time()

# Execution time: 30.083904027938843 seconds for true filename
# raw_data = pd.read_csv(filename, skiprows=10)
raw_data = pd.read_csv(test_filename)

end_read = time.time()
read_time = end_read - start_read
print("Read time: "+str(read_time) + " seconds ")

start_process = time.time()

raw_data.columns = ['X', 'Y', 'Z']

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data['vm'] = np.sqrt([(raw_data.X ** 2) + (raw_data.Y ** 2) + (raw_data.Z ** 2)])[0]

# Calculate the angle of arcsin from X and VM, arcsin(axis used/vector magnitude)/(pi/2)
raw_data['angle'] = np.arcsin(raw_data.X/raw_data['vm']) / (math.pi/2)

# Estimate a spectral density of vm using a fast Fourier transform. Compute the modulus corresponding to each frequency.
# This statistic is the sum of those moduli divided by the sum of the moduli at each frequency.
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html
# power_spectrum = np.abs(np.fft.fft(raw_data.vm))**2
# freqs = np.fft.fftfreq(raw_data.vm.size, 0.01)
# idx = np.argsort(freqs)
# print("freqs size: "+str(freqs.size)+"\n"+str(freqs))
# plt.plot(freqs[idx], power_spectrum[idx])
# plt.show()
# print(freqs, power_spectrum)
# sys.exit(0)


n = 1500

aggregated = raw_data.groupby(np.arange(len(raw_data))//n).mean()
print("Aggregate MVM and Mangle")

del aggregated['X']
del aggregated['Y']
del aggregated['Z']
aggregated.columns = ['mvm', 'mangle']

print("Processing SDVM and SDANGLE")

def getSD(row):
    return (row.vm - aggregated['mvm'][int(row.name/n)]) ** 2
raw_data['sd'] = raw_data.apply(getSD, axis=1)

print("getSD done")
def getSDAngle(row):
    return (row.angle - aggregated['mangle'][int(row.name/n)]) ** 2
raw_data['sdangle'] = raw_data.apply(getSDAngle, axis=1)


aggregated2 = raw_data.groupby(np.arange(len(raw_data))//n).mean()
del aggregated2['X']
del aggregated2['Y']
del aggregated2['Z']
aggregated2.columns = ['mvm', 'mangle', 'sdvm', 'sdangle']

aggregated2['sdvm'] = aggregated2['sdvm'].apply(lambda x: np.sqrt(x))
aggregated2['sdangle'] = aggregated2['sdangle'].apply(lambda x: np.sqrt(x))


# for i in range(0, len(raw_data.vm)):
#     category = int(i/n)
#
#     temp_msd = temp_msd + ((raw_data.vm[i] - aggregated.mvm[category]) ** 2)
#     temp_mangle = temp_mangle + ((raw_data.angle[i] - aggregated.mangle[category]) ** 2)
#
#     if i % n == 0:
#         aggregated.ix[category, 'sdvm'] = np.sqrt([(temp_msd/n)])
#         aggregated.ix[category, 'sdangle'] = np.sqrt([(temp_mangle/n)])
#
#         temp_msd = 0
#         temp_mangle = 0


end_process = time.time()

# print(raw_data.head(5))
# print(aggregated.head(5))

raw_data.to_csv("raw_data.csv", sep=',')
aggregated.to_csv("aggregated.csv", sep=',')
aggregated2.to_csv("aggregated2.csv", sep=',')

process_time = end_process - start_process
print("Read time: "+str(read_time) + " seconds ")
print("Process time: "+str(process_time) + " seconds ")
