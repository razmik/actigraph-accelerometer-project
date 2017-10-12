import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy.stats.stats import pearsonr


class BlandAltman:

    """
    Example method call:
    waist_ee = results.loc[(results['waist_ee'] >= 1) & (results['waist_ee'] < 3)]['waist_ee'].as_matrix()
    statistical_extensions.BlandAltman.plot_histogram(waist_ee, 300, 'Histogram of Energy Expenditure Value Distribution',
                                                  'Energy Expenditure (MET) from Waist', 9)
    """
    @staticmethod
    def plot_histogram(values_array, bins, title, xaxis_label, figure_id):
        plt.figure(figure_id)
        plt.title(title)
        plt.hist(values_array, bins=bins)
        plt.xlabel(xaxis_label)
        plt.show()

    @staticmethod
    def log_annotate_data(dataframe):

        new_df = dataframe.loc[((dataframe['subject'] == 'LSM263') | (dataframe['subject'] == 'LSM265'))
                               & (dataframe['mean'] < 0.08) & (dataframe['diff'] < 0.5)]
        for index, row in new_df.iterrows():
            print('A', row['waist_ee'], row['predicted_ee'])

        new_df = dataframe.loc[((dataframe['subject'] == 'LSM263') | (dataframe['subject'] == 'LSM265'))
                               & (0.08 < dataframe['mean']) & (dataframe['mean'] < 0.27) & (dataframe['diff'] < 0.4)]
        for index, row in new_df.iterrows():
            print('B', row['waist_ee'], row['predicted_ee'])

        new_df = dataframe.loc[((dataframe['subject'] == 'LSM263') | (dataframe['subject'] == 'LSM265'))
                               & (0.27 < dataframe['mean']) & (dataframe['mean'] < 0.48) & (dataframe['diff'] < 0.1)]
        for index, row in new_df.iterrows():
            print('C', row['waist_ee'], row['predicted_ee'])

        new_df = dataframe.loc[((dataframe['subject'] == 'LSM263') | (dataframe['subject'] == 'LSM265'))
                               & (0.27 < dataframe['mean']) & (dataframe['mean'] < 0.4) & (0.05 < dataframe['diff'])]
        for index, row in new_df.iterrows():
            print('D', row['waist_ee'], row['predicted_ee'])

        new_df = dataframe.loc[((dataframe['subject'] == 'LSM263') | (dataframe['subject'] == 'LSM265'))
                               & (0.4 < dataframe['mean']) & (dataframe['mean'] < 0.53) & (0.05 < dataframe['diff'])]
        for index, row in new_df.iterrows():
            print('E', row['waist_ee'], row['predicted_ee'])

        new_df = dataframe.loc[((dataframe['subject'] == 'LSM263') | (dataframe['subject'] == 'LSM265'))
                               & (0.53 < dataframe['mean']) & (dataframe['mean'] < 0.7) & (dataframe['diff'] < 0.4)]
        for index, row in new_df.iterrows():
            print('F', row['waist_ee'], row['predicted_ee'])

        new_df = dataframe.loc[(0.7 < dataframe['mean']) & (dataframe['diff'] < 0.4)]
        for index, row in new_df.iterrows():
            print('G', row['waist_ee'], row['predicted_ee'])

    @staticmethod
    def get_min_regularised_data_per_subject(data):

        min_count = min(data.groupby(['subject'])['waist_ee'].count())
        return data.groupby('subject').head(min_count)

    @staticmethod
    def annotate_bland_altman_paired_plot(dataframe):

        for label, x, y in zip(dataframe['subject'], dataframe['mean'], dataframe['diff']):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    @staticmethod
    def bland_altman_paired_plot_tested(dataframe, plot_title, plot_number, log_transformed=False, min_count_regularise=False, output_filename=''):

        """Plot 2 different BA plots for 2 MET calculatin equations"""
        dataframe_freedson = dataframe.loc[dataframe['waist_vm_cpm'] > 2453]
        dataframe_williams = dataframe.loc[dataframe['waist_vm_cpm'] <= 2453]

        dataframe_freedson, mean_bias_freedson, upper_loa_freedson, lower_loa_freedson = BlandAltman._bland_altman_analyse(dataframe_freedson, log_transformed=log_transformed, min_count_regularise=min_count_regularise)

        plt.figure(plot_number)
        plt.title(plot_title + ' - Freedson VM3 Combination (11)')
        # plt.scatter(np.log10(dataframe['waist_ee_cleaned']), dataframe['diff'])
        plt.scatter((dataframe_freedson['mean']), dataframe_freedson['diff'])
        plt.axhline(mean_bias_freedson, color='gray', linestyle='--')
        plt.axhline(upper_loa_freedson, color='gray', linestyle='--')
        plt.axhline(lower_loa_freedson, color='gray', linestyle='--')
        plt.xlabel('Mean Energy Expenditure (MET)')
        plt.ylabel('Difference between Energy Expenditure (MET)')

        plt.savefig(output_filename + '_freedson_bland_altman.png')

        dataframe_williams, mean_bias_williams, upper_loa_williams, lower_loa_williams = BlandAltman._bland_altman_analyse(dataframe_williams, log_transformed=log_transformed, min_count_regularise=min_count_regularise)

        plt.figure(plot_number+1)
        plt.title(plot_title + ' - Williams Work-Energy (98)')
        plt.scatter((dataframe_williams['mean']), dataframe_williams['diff'])
        plt.axhline(mean_bias_williams, color='gray', linestyle='--')
        plt.axhline(upper_loa_williams, color='gray', linestyle='--')
        plt.axhline(lower_loa_williams, color='gray', linestyle='--')
        plt.xlabel('Mean Energy Expenditure (MET)')
        plt.ylabel('Difference between Energy Expenditure (MET)')

        plt.savefig(output_filename + '_williams_bland_altman.png')

    @staticmethod
    def _bland_altman_analyse(dataframe, log_transformed=False, min_count_regularise=False):

        if min_count_regularise:
            dataframe = BlandAltman.get_min_regularised_data_per_subject(dataframe)

        if log_transformed:
            dataframe = dataframe.assign(waist_ee_log_transformed=np.log10(dataframe['waist_ee_cleaned']))
            dataframe = dataframe.assign(predicted_ee_log_transformed=np.log10(dataframe['predicted_ee_cleaned']))

        dataframe = dataframe.assign(mean=np.mean([dataframe.as_matrix(columns=['waist_ee_cleaned']),
                                     dataframe.as_matrix(columns=['predicted_ee_cleaned'])], axis=0))
        dataframe = dataframe.assign(diff=dataframe['waist_ee_log_transformed'] - dataframe['predicted_ee_log_transformed'])

        k = len(pd.unique(dataframe.subject))  # number of conditions
        N = len(dataframe.values)  # conditions times participants

        DFbetween = k - 1
        DFwithin = N - k
        DFtotal = N - 1

        anova_data = pd.DataFrame()
        dataframe_summary = dataframe.groupby(['subject'])
        anova_data['count'] = dataframe_summary['diff'].count()  # number of values in each group ng
        anova_data['sum'] = dataframe_summary['diff'].sum()  # sum of values in each group
        anova_data['mean'] = dataframe_summary['diff'].mean()  # mean of values in each group Xg
        anova_data['variance'] = dataframe_summary['diff'].var()
        anova_data['sd'] = np.sqrt(anova_data['variance'])
        anova_data['count_sqr'] = anova_data['count'] ** 2

        grand_mean = anova_data['sum'].sum() / anova_data['count'].sum()  # XG

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

        MSbetween = SSbetween / DFbetween
        MSwithin = SSwithin / DFwithin
        # F = MSbetween / MSwithin
        # p = stats.f.sf(F, DFbetween, DFwithin)

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

        # print('Upper LOA:', upper_loa)
        # print('Mean LOA:', mean_bias)
        # print('Lower LOA:', lower_loa)

        return dataframe, mean_bias, upper_loa, lower_loa


    @staticmethod
    def clean_data_points(data):
        # Remove row if reference MET value is less than 1
        data = data[data.waist_ee >= 1]

        data = data.assign(waist_ee_cleaned=data['waist_ee'])
        data = data.assign(predicted_ee_cleaned=data['predicted_ee'])

        # data.loc[(data['waist_ee'] < 1), 'waist_ee_cleaned'] = 1
        data.loc[(data['predicted_ee'] < 1), 'predicted_ee_cleaned'] = 1

        return data

    @staticmethod
    def clean_data_points_reference_only(data):
        # Remove row if reference MET value is less than 1
        data = data[data.waist_ee >= 1]

        data = data.assign(waist_ee_transformed=data['waist_ee'])

        return data

    @staticmethod
    def clean_data_points_prediction_only(data, prediction_column):

        data.loc[(data[prediction_column] < 0), prediction_column] = 0

        return data


class GeneralStats:

    @staticmethod
    def evaluation_statistics(confusion_matrix):

        # Name the values
        tpa = confusion_matrix[0, 0]
        tpb = confusion_matrix[1, 1]
        tpc = confusion_matrix[2, 2]
        eab = confusion_matrix[0, 1]
        eac = confusion_matrix[0, 2]
        eba = confusion_matrix[1, 0]
        ebc = confusion_matrix[1, 2]
        eca = confusion_matrix[2, 0]
        ecb = confusion_matrix[2, 1]

        # Calculate accuracy for label 1
        total_classifications = sum(sum(confusion_matrix))
        accuracy = (tpa + tpb + tpc) / total_classifications

        # Calculate Precision for label 1
        precisionA = tpa / (tpa + eba + eca)

        # Calculate Sensitivity for label 1
        sensitivityA = tpa / (tpa + eab + eac)

        # Calculate Specificity for label 1
        tna = tpb + ebc + ecb + tpc
        specificityA = tna / (tna + eba + eca)

        # Calculate Precision for label 2
        precisionB = tpb / (tpb + eab + ecb)

        # Calculate Sensitivity for label 2
        sensitivityB = tpb / (tpb + eba + ebc)

        # Calculate Specificity for label 2
        tnb = tpa + eac + eca + tpc
        specificityB = tnb / (tnb + eab + ecb)

        # Calculate Precision for label 2
        precisionC = tpc / (tpc + eac + ebc)

        # Calculate Sensitivity for label 2
        sensitivityC = tpc / (tpc + eca + ecb)

        # Calculate Specificity for label 2
        tnc = tpa + eab + eba + tpb
        specificityC = tnc / (tnc + eac + ebc)

        return {
            'accuracy': accuracy,
            'precision': [precisionA, precisionB, precisionC],
            'recall': [sensitivityA, sensitivityB, sensitivityC],
            'sensitivity': [sensitivityA, sensitivityB, sensitivityC],
            'specificity': [specificityA, specificityB, specificityC]
        }

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.Blues, output_filename=''):
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

        plt.savefig(output_filename)

    """
     Parameters
     ----------
     x : 1D array
     y : 1D array the same length as x
    
     Returns
     -------
     (Pearson's correlation coefficient, 2-tailed p-value)
    """
    @staticmethod
    def pearson_correlation(x, y):
        return pearsonr(x, y)


class Utils:

    @staticmethod
    def print_assessment_results(output_filename, result_string):
        with open(output_filename, "w") as text_file:
            text_file.write(result_string)
