"""
1. Read wrist raw data file
2. Aggregate into time interval - epoch
3. Calculate time domain statistical parameters - eg. vm, max, min, percentiles, etc.
4. Calculate frequency domain statistical parameters
5. Combine statistical parameters with ActiGraph processed wrist-waist epoch counts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math

wrist_raw_data_filename = "D:/Accelerometer Data/Sample/LSM216/LSM216 Waist (2016-11-03)RAW.csv"
epoch_filename = "D:/Accelerometer Data/Sample/LSM216/LSM216 Waist (2016-11-03)15sec.csv"

# epoch granularity
n = 1500
starting_row = 0
end_row = 360000*10
epoch_start = int(starting_row/n) + 10

start = starting_row + 10
row_count = end_row - starting_row

print("Duration:", ((end_row-starting_row)/(100*3600)), "hours")
print("Reading raw data file.")

raw_data_wrist = pd.read_csv(wrist_raw_data_filename, skiprows=start, nrows=row_count)

raw_data_wrist.columns = ['X', 'Y', 'Z']

raw_data_wrist['X'] = raw_data_wrist['X'].abs()
raw_data_wrist['Y'] = raw_data_wrist['Y'].abs()
raw_data_wrist['Z'] = raw_data_wrist['Z'].abs()

"""
Calculate the statistical inputs (Features)
"""
print("Calculating statistical parameters.")

# Calculate the vector magnitude from X, Y, Z raw readings
raw_data_wrist['vm'] = np.sqrt([(raw_data_wrist.X ** 2) + (raw_data_wrist.Y ** 2) + (raw_data_wrist.Z ** 2)])[0]
aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm']


aggregated_wrist = raw_data_wrist.groupby(np.arange(len(raw_data_wrist))//n).mean()
aggregated_wrist.columns = ['X', 'Y', 'Z', 'mvm']
wrist_grouped_temp = raw_data_wrist.groupby(raw_data_wrist.index // n)

# print(raw_data_wrist)

raw_data_wrist_groups = np.array_split(raw_data_wrist, len(aggregated_wrist))

# print(raw_data_wrist_groups[0])
# frequency domain features

for i in range(0, len(raw_data_wrist_groups)):

    """
    dt = 0.01  # sample rate
    N = 1500  # number of samples
    T = dt * N  # time period (duration of signal)

    # fundamental frequency
    df = 1 / T  # Hz
    dw = (2 * math.pi) / T  # rad/s

    ny = dw * N / 2  # top frequency

    f = np.fft.fftfreq(N) * N * df
    """

    spectrum = np.fft.fft(np.sin(raw_data_wrist_groups[i]['vm']))
    freqs = np.fft.fftfreq(raw_data_wrist_groups[i]['vm'].shape[-1], 0.01)


    idx = np.argmax(np.abs(spectrum))
    freq = freqs[idx]

    print("range", i, freq, np.amax(np.abs(spectrum)))

    plt.plot(freqs, abs(spectrum))
    plt.show()

    sys.exit(0)

"""
Helpers:

https://docs.scipy.org/doc/numpy/reference/routines.fft.html

https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python#comment3894827_3694976



"""


# plt.plot(freq, np.abs(spectrum))
# plt.show()

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
