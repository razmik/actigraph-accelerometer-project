import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + " Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print(title + ' confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# https://stackoverflow.com/questions/16399279/bland-altman-plot-in-python
def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)  # x - mean | y - diff
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')


def bland_altman_paired_plot_tested(dataframex):

    # Get the minimum count by subject
    min_count = min(dataframex.groupby(['subject'])['waist_ee'].count())
    dataframe = dataframex.groupby('subject').head(min_count)
    #
    # print(dataframe)

    log_transformed = True

    # dataframe = pd.read_csv('E:\Accelerometry project\MSSE Examples\BA_Adj_2007_Sample_Data.csv'.replace('\\', '/'))
    # dataframe = pd.read_csv('E:\Accelerometry project\MSSE Examples\\anova_hand_test.csv'.replace('\\', '/'))
    # dataframe['subject'] = dataframe['subject'].astype(str)

    dataframe['waist_ee_transformed'] = dataframe['waist_ee']
    dataframe.loc[(dataframe['waist_ee'] <= 0.0), 'waist_ee_transformed'] = 0.000000001
    dataframe['predicted_ee_transformed'] = dataframe['predicted_ee']
    dataframe.loc[(dataframe['predicted_ee'] <= 0.0), 'predicted_ee_transformed'] = 0.000000001

    if log_transformed:
        dataframe['waist_ee_transformed'] = np.log10(dataframe['waist_ee_transformed'])
        dataframe['predicted_ee_transformed'] = np.log10(dataframe['predicted_ee_transformed'])

    dataframe['mean'] = np.mean([dataframe.as_matrix(columns=['waist_ee_transformed']),
                                 dataframe.as_matrix(columns=['predicted_ee_transformed'])], axis=0)
    dataframe['diff'] = dataframe['waist_ee_transformed'] - dataframe['predicted_ee_transformed']

    # for index, row in dataframe.iterrows():
    #     if row['waist_ee_transformed'] < 0 or row['predicted_ee_transformed'] < 0:
    #         print(row['waist_ee_transformed'], row['predicted_ee_transformed'])
    # sys.exit(0)

    k = len(pd.unique(dataframe.subject))  # number of conditions
    N = len(dataframe.values)  # conditions times participants

    DFbetween = k - 1
    DFwithin = N - k
    DFtotal = N - 1

    print('DF_between', DFbetween)
    print('DF_within', DFwithin)
    print('DF_total', DFtotal)

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

    print('Grand mean', grand_mean)

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
    divisor = (m ** 2 - sigma_m2) / ((n - 1) * m)
    variance = diff_bet_within / divisor

    total_variance = variance + variance_b_method
    sd = np.sqrt(total_variance)

    mean_bias = sum(anova_data['sum']) / m
    upper_loa = mean_bias + (1.96 * sd)
    lower_loa = mean_bias - (1.96 * sd)

    print(mean_bias, upper_loa, lower_loa)

    plt.title('Montoye 2017 ANN left_wrist')

    plt.scatter(dataframe['mean'], dataframe['diff'])  # x - mean | y - diff
    plt.axhline(mean_bias, color='gray', linestyle='--')
    plt.axhline(upper_loa, color='gray', linestyle='--')
    plt.axhline(lower_loa, color='gray', linestyle='--')

    plt.show()


# # http://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
# # https://www.youtube.com/watch?v=q48uKU_KWas
# def bland_altman_paired_plot(dataframe):
#
#     # d = {'subject': ['1','1','1','2','2','2','3','3','3'], 'diff': [1,2,5,2,4,2,2,3,4]}
#     # dataframe = pd.DataFrame(data=d)
#
#     dataframe = pd.read_csv('E:\Accelerometry project\MSSE Examples\BA_Adj_2007_Sample_Data.csv'.replace('\\', '/'))
#     dataframe['subject'] = dataframe['subject'].astype(str)
#
#     dataframe['mean'] = np.mean([dataframe.as_matrix(columns=['waist_ee']), dataframe.as_matrix(columns=['predicted_ee'])], axis=0)
#     # dataframe['diff'] = dataframe['waist_ee'] - dataframe['predicted_ee']
#     dataframe['diff'] = np.log10(dataframe['waist_ee']) - np.log10(dataframe['predicted_ee'])
#
#     anova_data = pd.DataFrame()
#     dataframe_summary = dataframe.groupby(['subject'])
#
#     anova_data['count'] = dataframe_summary['diff'].count()
#     anova_data['sum'] = dataframe_summary['diff'].sum()
#     anova_data['mean'] = dataframe_summary['diff'].mean()
#     anova_data['variance'] = dataframe_summary['diff'].var()
#     anova_data['sd'] = np.sqrt(anova_data['variance'])
#     anova_data['count_sqr'] = anova_data['count'] ** 2
#
#     # plt.figure()
#     # dataframe.boxplot('diff', by='subject', figsize=(12, 8))
#     # plt.show()
#
#     # grps = pd.unique(dataframe.subject.values)
#     # d_data = {grp: dataframe['diff'][dataframe.subject == grp] for grp in grps}
#
#     k = len(pd.unique(dataframe.subject))  # number of conditions
#     N = len(dataframe.values)  # conditions times participants
#     n = dataframe.groupby('subject').size()[0]  # Participants in each condition
#
#     DFbetween = k - 1
#     DFwithin = N - k
#     DFtotal = N - 1
#
#     print('DF_between', DFbetween)
#     print('DF_within', DFwithin)
#     print('DF_total', DFtotal)
#
#     SSbetween = (sum(dataframe.groupby('subject').sum()['diff'] ** 2) / n) - (dataframe['diff'].sum() ** 2) / N
#     sum_y_squared = sum([value ** 2 for value in dataframe['diff'].values])
#     SSwithin = sum_y_squared - sum(dataframe.groupby('subject').sum()['diff'] ** 2) / n
#     SStotal = sum_y_squared - (dataframe['diff'].sum() ** 2) / N
#
#     print('SS_between', SSbetween)
#     print('SS_within', SSwithin)
#     print('SS_total', SStotal)
#
#     MSbetween = SSbetween / DFbetween
#     MSwithin = SSwithin / DFwithin
#     F = MSbetween / MSwithin
#     p = stats.f.sf(F, DFbetween, DFwithin)
#
#     print('MS_between', MSbetween)
#     print('MS_within', MSwithin)
#
#     print('F', F)
#
#     m = DFtotal + 1
#     sigma_m2 = sum(anova_data['count_sqr'])
#
#     variance_b_method = MSwithin
#
#     diff_bet_within = MSbetween - MSwithin
#     divisor = (m ** 2 - sigma_m2) / ((n-1) * m)
#     variance = diff_bet_within / divisor
#
#     total_variance = variance + variance_b_method
#     sd = np.sqrt(total_variance)
#
#     mean_bias = sum(anova_data['sum']) / m
#     upper_loa = mean_bias + (1.96 * sd)
#     lower_loa = mean_bias - (1.96 * sd)
#
#     print(mean_bias, upper_loa, lower_loa)
#
#     plt.title('Montoye 2017 ANN left_wrist')
#
#     plt.scatter(dataframe['mean'], dataframe['diff'])  # x - mean | y - diff
#     plt.axhline(mean_bias, color='gray', linestyle='--')
#     plt.axhline(upper_loa, color='gray', linestyle='--')
#     plt.axhline(lower_loa, color='gray', linestyle='--')
#
#     plt.show()


wrist = 'right_wrist'
model = 'v1v2'
# model = 'v2'
print('Montoye 2017 ANN', wrist)

result_folder = 'D:\Accelerometer Data\Montoye\Features\LSM2\Week 1\Wednesday\Assessment2\Montoye_2017_predictions/' + wrist + '/60sec/' + model + '/combined/'.replace(
    '\\', '/')
result_data_files = [f for f in listdir(result_folder) if isfile(join(result_folder, f))]

"""
Evaluate the users as a whole
"""

count = 0
for file in result_data_files:

    dataframe = pd.read_csv(result_folder + file)
    dataframe['subject'] = file.split('_(2016')[0]

    if count == 0:
        results = dataframe
    else:
        results = results.append(dataframe, ignore_index=True)

    count += 1


bland_altman_paired_plot_tested(results)

sys.exit(0)

# count = 0
# for file in result_data_files:
#
#     key = file.split('_(2016')[0]
#
#     if count == 0:
#         results = pd.read_csv(result_folder + file)
#     else:
#         results = results.append(pd.read_csv(result_folder + file), ignore_index=True)
#     count += 1

target_intensity = results['actual_category']
predicted_intensity = results['predicted_category']

target_ee = results['waist_ee']
predicted_ee = results['predicted_ee']


bland_altman_plot(target_ee, predicted_ee)
plt.title('Bland-Altman Plot')
plt.show()
sys.exit(0)

class_names = ['SED', 'LPA', 'MVPA']

# The mean squared error
print("Montoye 2017 ANN (MET) Mean squared error: %.2f" % np.mean((predicted_ee - target_ee) ** 2))
print("Montoye 2017 ANN (Instensity) Mean squared error: %.2f" % np.mean((predicted_intensity - target_intensity) ** 2))

# The R squared score
print("Montoye 2017 ANN (MET) R squared score: %.2f" % r2_score(target_ee, predicted_ee))
print("Montoye 2017 ANN (Instensity) R squared score: %.2f" % r2_score(target_intensity, predicted_intensity))

# Precision and Recall
precision, recall, fscore, support = precision_recall_fscore_support(target_intensity, predicted_intensity,
                                                                     average='macro')
print('overall precision: {}'.format(precision))
print('overall recall: {}'.format(recall))
print('overall fscore: {}'.format(fscore))
print('overall support: {}'.format(support))

precision, recall, fscore, support = precision_recall_fscore_support(target_intensity, predicted_intensity)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

# Compute confusion matrix for intensity
cnf_matrix = confusion_matrix(target_intensity, predicted_intensity)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(1)
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Montoye 2017 ' + wrist + ' ANN')
plt.show()
