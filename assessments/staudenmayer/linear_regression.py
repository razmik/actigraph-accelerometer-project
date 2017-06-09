import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import scipy.stats as stats
import matplotlib.pyplot as plt

epoch_filename = "D:/Accelerometer Data/Processed/LSM2/Week 1/Wednesday/LSM255_(2016-11-01)_row_0_to_1920.csv".replace('\\', '/')

data = pd.read_csv(epoch_filename)
del data['Unnamed: 0']

print("bandpass wrist vm vs. wrist_epoch_15:", round(stats.pearsonr(data['band_vm'], data['wrist_vm_15'])[0], 2))
print("wrist processed vm vs. waist process:", round(stats.pearsonr(data['waist_vm_60'], data['wrist_vm_60'])[0], 2))
print("bandpass wrist vm vs. waist processed:", round(stats.pearsonr(data['band_vm'], data['waist_vm_60'])[0], 2))


data.loc[data['waist_intensity'] == 1, 'target_met_category'] = 1
data.loc[data['waist_intensity'] == 2, 'target_met_category'] = 2
data.loc[data['waist_intensity'] == 3, 'target_met_category'] = 3
data.loc[data['waist_intensity'] == 4, 'target_met_category'] = 3

data['estimated_met'] = 1.89378 + (5.50821 * data['wrist_sdvm']) - (0.02705 * data['wrist_mangle'])
data.loc[data['estimated_met'] < 3, 'estimated_met_category'] = 1
data.loc[(3 <= data['estimated_met']) & (data['estimated_met'] < 6), 'estimated_met_category'] = 2
data.loc[6 <= data['estimated_met'], 'estimated_met_category'] = 3

target = data['target_met_category']
estimated = data['estimated_met_category']

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((estimated - target) ** 2))

# The R squared score
# r2_score(y_true, y_pred, sample_weight=None, multioutput=None)
print("R squared score: %.2f"
      % r2_score(target, estimated))

plt.plot(np.arange(len(data)), target, 'r', np.arange(len(data)), estimated, 'b')
plt.show()
