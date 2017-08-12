import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


dataframe = pd.read_csv('E:\Accelerometry project\MSSE Examples\BA_Adj_2007_Sample_Data.csv'.replace('\\', '/'))
# dataframe = pd.read_csv('E:\Accelerometry project\MSSE Examples\\anova_hand_test.csv'.replace('\\', '/'))
# https://www.youtube.com/watch?v=WUjsSB7E-ko
# https://mcfromnz.wordpress.com/2011/03/02/anova-type-iiiiii-ss-explained/
dataframe['subject'] = dataframe['subject'].astype(str)

dataframe['waist_ee'] = np.log10(dataframe['waist_ee'])
dataframe['predicted_ee'] = np.log10(dataframe['predicted_ee'])

dataframe['mean'] = np.mean([dataframe.as_matrix(columns=['waist_ee']), dataframe.as_matrix(columns=['predicted_ee'])], axis=0)
dataframe['diff'] = dataframe['waist_ee'] - dataframe['predicted_ee']

mod = ols('diff ~ subject',
          data=dataframe).fit()

# http://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html
# https://mcfromnz.wordpress.com/2011/03/02/anova-type-iiiiii-ss-explained/
aov_table = sm.stats.anova_lm(mod, typ=2)

subject_ss = aov_table['sum_sq']['subject']
residual_ss = aov_table['sum_sq']['Residual']
subject_df = aov_table['df']['subject']
residual_df = aov_table['df']['Residual']
f = aov_table['F']['subject']
print(aov_table)

# sys.exit(0)
print('\n')
# d = {'subject': ['1','1','1','2','2','2','3','3','3'], 'diff': [1,2,5,2,4,2,2,3,4]}
# dataframe = pd.DataFrame(data=d)


k = len(pd.unique(dataframe.subject))  # number of conditions
N = len(dataframe.values)  # conditions times participants

DFbetween = k - 1
DFwithin = N - k
DFtotal = N - 1

# print('DF_between', DFbetween)
# print('DF_within', DFwithin)
# print('DF_total', DFtotal)

anova_data = pd.DataFrame()
dataframe_summary = dataframe.groupby(['subject'])

# anova_data['subject'] = dataframe_summary['subject']
anova_data['count'] = dataframe_summary['diff'].count()  # number of values in each group ng
anova_data['sum'] = dataframe_summary['diff'].sum()  # sum of values in each group
anova_data['mean'] = dataframe_summary['diff'].mean()  # mean of values in each group Xg
anova_data['variance'] = dataframe_summary['diff'].var()
anova_data['sd'] = np.sqrt(anova_data['variance'])
anova_data['count_sqr'] = anova_data['count'] ** 2

grand_mean = anova_data['sum'].sum() / anova_data['count'].sum()  # XG

# print('Grand mean', grand_mean)

# Calculate the MSS within

squared_within = 0
for name, group in dataframe_summary:
    group_mean = group['diff'].sum() / group['diff'].count()

    squared = 0
    for index, row in group.iterrows():
        squared += (row['diff'] - group_mean) ** 2

    squared_within += squared


SSwithin = squared_within

# Calculate the MSS between
ss_between_partial = 0
for index, row in anova_data.iterrows():
    ss_between_partial += row['count'] * ((row['mean'] - grand_mean) ** 2)

SSbetween = ss_between_partial


#  Calculate SS total

squared_total = 0

for index, row in dataframe.iterrows():
    squared_total += (row['diff'] - grand_mean) ** 2

SStotal = squared_total

print('SS_between', SSbetween)
print('SS_within', SSwithin)
print('SS_total', SStotal)

MSbetween = SSbetween / DFbetween
MSwithin = SSwithin / DFwithin
F = MSbetween / MSwithin
p = stats.f.sf(F, DFbetween, DFwithin)

print('MS_between', MSbetween)
print('MS_within', MSwithin)

print('F', F)

n = DFbetween + 1
m = DFtotal + 1
sigma_m2 = sum(anova_data['count_sqr'])

variance_b_method = MSwithin

diff_bet_within = MSbetween - MSwithin
divisor = (m ** 2 - sigma_m2) / ((n-1) * m)
variance = diff_bet_within / divisor

total_variance = variance + variance_b_method
sd = np.sqrt(total_variance)

mean_bias = sum(anova_data['sum']) / m
upper_loa = mean_bias + (1.96 * sd)
lower_loa = mean_bias - (1.96 * sd)

print('Upper LOA', upper_loa)
print('Mean Bias', mean_bias)
print('Lower LOA', lower_loa)

plt.title('Montoye 2017 ANN left_wrist')

plt.scatter(dataframe['mean'], dataframe['diff'])  # x - mean | y - diff
plt.axhline(mean_bias, color='gray', linestyle='--')
plt.axhline(upper_loa, color='gray', linestyle='--')
plt.axhline(lower_loa, color='gray', linestyle='--')

plt.show()