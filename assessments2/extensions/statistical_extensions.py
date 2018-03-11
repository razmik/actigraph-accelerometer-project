import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
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
    def get_antilog(log_val):
        return round(math.pow(10, log_val), 2)

    @staticmethod
    def plot_graph(plot_number, plot_title, x_values, y_values, upper_loa, mean_bias, lower_loa, output_filename):

        x_label = 'Mean Energy Expenditure (METs)'
        y_label = 'Difference (Prediction - Reference) (log METs)'
        x_lim = (1, 24)
        y_lim = (-1.2, 1.4)
        # x_lim = (1, 12)
        # y_lim = (-1.2, 1.0)
        x_annotate_begin = 10.4
        y_gap = 0.05
        ratio_suffix = ''

        plt.figure(plot_number, dpi=1200)
        # plt.title(plot_title)
        plt.scatter(x_values, y_values)

        # Black: #000000
        plt.axhline(upper_loa, color='gray', linestyle='dotted')
        plt.axhline(mean_bias, color='gray', linestyle='--')
        plt.axhline(lower_loa, color='gray', linestyle='dotted')

        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # http://www.anjuke.tech/questions/843083/matplotlib-savefig-in-jpeg-format

        plt.savefig(output_filename + '.jpg', dpi=1200)

        plt.annotate(str(BlandAltman.get_antilog(upper_loa)) + ratio_suffix, xy=(x_annotate_begin, (upper_loa + y_gap)))
        plt.annotate(str(BlandAltman.get_antilog(mean_bias)) + ratio_suffix, xy=(x_annotate_begin, (mean_bias + y_gap)))
        plt.annotate(str(BlandAltman.get_antilog(lower_loa)) + ratio_suffix, xy=(x_annotate_begin, (lower_loa + y_gap)))

        plt.savefig(output_filename + '_annotated.jpg', dpi=1200)
        # plt.savefig(output_filename+'.png', dpi=1200)
        # plt.savefig(output_filename+'.eps')
        # plt.savefig(output_filename+'.pdf')
        # plt.savefig(output_filename+'.svg', format='svg')

    @staticmethod
    def bland_altman_paired_plot_tested(dataframe, plot_title, plot_number, log_transformed=False,
                                        min_count_regularise=False, output_filename=''):

        """Define multiple dataframes based on the activity intensity"""
        dataframe_sb = dataframe.loc[dataframe['waist_ee'] <= 1.5]
        dataframe_lpa = dataframe.loc[(1.5 < dataframe['waist_ee']) & (dataframe['waist_ee'] < 3)]
        dataframe_mvpa = dataframe.loc[3 <= dataframe['waist_ee']]

        """
        Process BA plot for SB
        """
        # dataframe_sb_freedson = dataframe_sb.loc[dataframe_sb['waist_vm_cpm'] > 2453]
        dataframe_sb_williams = dataframe_sb.loc[dataframe_sb['waist_vm_cpm'] <= 2453]

        # if len(dataframe_sb_freedson) > 0:
        #     dataframe_sb_freedson, mean_bias_sb_freedson, upper_loa_sb_freedson, lower_loa_sb_freedson = BlandAltman._bland_altman_analyse(dataframe_sb_freedson, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        #     BlandAltman.plot_graph(plot_number, plot_title + ' - SB - Freedson VM3 Combination (11)',
        #                            dataframe_sb_freedson['mean'], dataframe_sb_freedson['diff'],
        #                            upper_loa_sb_freedson, mean_bias_sb_freedson, lower_loa_sb_freedson,
        #                            output_filename + '_sb_freedson_bland_altman.png')

        if len(dataframe_sb_williams) > 0:
            dataframe_sb_williams, mean_bias_sb_williams, upper_loa_sb_williams, lower_loa_sb_williams = BlandAltman._bland_altman_analyse(
                dataframe_sb_williams, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
            BlandAltman.plot_graph(plot_number + 1, plot_title + ' - SB - Williams Work-Energy (98)',
                                   dataframe_sb_williams['mean'], dataframe_sb_williams['diff'],
                                   upper_loa_sb_williams, mean_bias_sb_williams, lower_loa_sb_williams,
                                   output_filename + '_sb')
            # print(plot_title, 'SB', upper_loa_sb_williams, mean_bias_sb_williams, lower_loa_sb_williams)

        """
        Process BA plot for LPA
        """
        # dataframe_lpa_freedson = dataframe_lpa.loc[dataframe_lpa['waist_vm_cpm'] > 2453]
        dataframe_lpa_williams = dataframe_lpa.loc[dataframe_lpa['waist_vm_cpm'] <= 2453]

        # if len(dataframe_lpa_freedson) > 0:
        #     dataframe_lpa_freedson, mean_bias_lpa_freedson, upper_loa_lpa_freedson, lower_loa_lpa_freedson = BlandAltman._bland_altman_analyse(dataframe_lpa_freedson, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        #     BlandAltman.plot_graph(plot_number+2, plot_title + ' - LPA - Freedson VM3 Combination (11)',
        #                            dataframe_lpa_freedson['mean'], dataframe_lpa_freedson['diff'],
        #                            upper_loa_lpa_freedson, mean_bias_lpa_freedson, lower_loa_lpa_freedson,
        #                            output_filename + '_lpa_freedson_bland_altman.png')

        if len(dataframe_lpa_williams) > 0:
            dataframe_lpa_williams, mean_bias_lpa_williams, upper_loa_lpa_williams, lower_loa_lpa_williams = BlandAltman._bland_altman_analyse(
                dataframe_lpa_williams, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
            BlandAltman.plot_graph(plot_number + 3, plot_title + ' - LPA - Williams Work-Energy (98)',
                                   dataframe_lpa_williams['mean'], dataframe_lpa_williams['diff'],
                                   upper_loa_lpa_williams, mean_bias_lpa_williams, lower_loa_lpa_williams,
                                   output_filename + '_lpa')
            # print(plot_title, 'LPA', upper_loa_lpa_williams, mean_bias_lpa_williams, lower_loa_lpa_williams)

        """
        Process BA plot for MVPA
        """
        dataframe_mvpa_freedson = dataframe_mvpa.loc[dataframe_mvpa['waist_vm_cpm'] > 2453]
        # dataframe_mvpa_williams = dataframe_mvpa.loc[dataframe_mvpa['waist_vm_cpm'] <= 2453]

        if len(dataframe_mvpa_freedson) > 0:
            dataframe_mvpa_freedson, mean_bias_mvpa_freedson, upper_loa_mvpa_freedson, lower_loa_mvpa_freedson = BlandAltman._bland_altman_analyse(
                dataframe_mvpa_freedson, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
            BlandAltman.plot_graph(plot_number + 4, plot_title + ' - MVPA - Freedson VM3 Combination (11)',
                                   dataframe_mvpa_freedson['mean'], dataframe_mvpa_freedson['diff'],
                                   upper_loa_mvpa_freedson, mean_bias_mvpa_freedson, lower_loa_mvpa_freedson,
                                   output_filename + '_mvpa')
            # print(plot_title, 'MVPA', upper_loa_mvpa_freedson, mean_bias_mvpa_freedson, lower_loa_mvpa_freedson)

        # if len(dataframe_mvpa_williams) > 0:
        #     dataframe_mvpa_williams, mean_bias_mvpa_williams, upper_loa_mvpa_williams, lower_loa_mvpa_williams = BlandAltman._bland_altman_analyse(dataframe_mvpa_williams, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
        #     BlandAltman.plot_graph(plot_number+5, plot_title + ' - MVPA - Williams Work-Energy (98)',
        #                            dataframe_mvpa_williams['mean'], dataframe_mvpa_williams['diff'],
        #                            upper_loa_mvpa_williams, mean_bias_mvpa_williams, lower_loa_mvpa_williams,
        #                            output_filename + '_mvpa_williams_bland_altman.png')

    @staticmethod
    def bland_altman_paired_plot_for_two_catagories(dataframe, plot_title, plot_number, log_transformed=False,
                                                    min_count_regularise=False, output_filename=''):

        """Define multiple dataframes based on the activity intensity"""
        dataframe_sb_lpa = dataframe.loc[dataframe['waist_ee'] < 3]
        dataframe_mvpa = dataframe.loc[3 <= dataframe['waist_ee']]

        """
        Process BA plot for SB + LPA
        """
        dataframe_sb_lpa_freedson = dataframe_sb_lpa.loc[dataframe_sb_lpa['waist_vm_cpm'] > 2453]
        dataframe_sb_lpa_williams = dataframe_sb_lpa.loc[dataframe_sb_lpa['waist_vm_cpm'] <= 2453]

        if len(dataframe_sb_lpa_freedson) > 0:
            dataframe_sb_lpa_freedson, mean_bias_sb_lpa_freedson, upper_loa_sb_lpa_freedson, lower_loa_sb_lpa_freedson = BlandAltman._bland_altman_analyse(
                dataframe_sb_lpa_freedson, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
            BlandAltman.plot_graph(plot_number, plot_title + ' - SB-LPA - Freedson VM3 Combination (11)',
                                   dataframe_sb_lpa_freedson['mean'], dataframe_sb_lpa_freedson['diff'],
                                   upper_loa_sb_lpa_freedson, mean_bias_sb_lpa_freedson, lower_loa_sb_lpa_freedson,
                                   output_filename + '_sb_lpa_freedson_bland_altman.png')

        if len(dataframe_sb_lpa_williams) > 0:
            dataframe_sb_lpa_williams, mean_bias_sb_lpa_williams, upper_loa_sb_lpa_williams, lower_loa_sb_lpa_williams = BlandAltman._bland_altman_analyse(
                dataframe_sb_lpa_williams, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
            BlandAltman.plot_graph(plot_number + 1, plot_title + ' - SB-LPA - Williams Work-Energy (98)',
                                   dataframe_sb_lpa_williams['mean'], dataframe_sb_lpa_williams['diff'],
                                   upper_loa_sb_lpa_williams, mean_bias_sb_lpa_williams, lower_loa_sb_lpa_williams,
                                   output_filename + '_sb')

        """
       Process BA plot for MVPA
       """
        dataframe_mvpa_freedson = dataframe_mvpa.loc[dataframe_mvpa['waist_vm_cpm'] > 2453]

        if len(dataframe_mvpa_freedson) > 0:
            dataframe_mvpa_freedson, mean_bias_mvpa_freedson, upper_loa_mvpa_freedson, lower_loa_mvpa_freedson = BlandAltman._bland_altman_analyse(
                dataframe_mvpa_freedson, log_transformed=log_transformed, min_count_regularise=min_count_regularise)
            BlandAltman.plot_graph(plot_number + 4, plot_title + ' - MVPA - Freedson VM3 Combination (11)',
                                   dataframe_mvpa_freedson['mean'], dataframe_mvpa_freedson['diff'],
                                   upper_loa_mvpa_freedson, mean_bias_mvpa_freedson, lower_loa_mvpa_freedson,
                                   output_filename + '_mvpa')

    @staticmethod
    def _bland_altman_analyse(dataframe, log_transformed=False, min_count_regularise=False):

        if min_count_regularise:
            dataframe = BlandAltman.get_min_regularised_data_per_subject(dataframe)

        if log_transformed:
            dataframe = dataframe.assign(waist_ee_log_transformed=np.log10(dataframe['waist_ee_cleaned']))
            dataframe = dataframe.assign(predicted_ee_log_transformed=np.log10(dataframe['predicted_ee_cleaned']))

        dataframe = dataframe.assign(mean=np.mean([dataframe.as_matrix(columns=['waist_ee_cleaned']),
                                                   dataframe.as_matrix(columns=['predicted_ee_cleaned'])], axis=0))
        dataframe = dataframe.assign(
            diff=dataframe['predicted_ee_log_transformed'] - dataframe['waist_ee_log_transformed'])
        # dataframe = dataframe.assign(diff=dataframe['waist_ee_cleaned']/dataframe['predicted_ee_cleaned'])

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

        return dataframe, mean_bias, upper_loa, lower_loa

    @staticmethod
    def clean_data_points(data):
        # Remove row if reference MET value is less than 1
        data = data[data.waist_ee >= 1]

        data = data.assign(waist_ee_cleaned=data['waist_ee'])
        data = data.assign(predicted_ee_cleaned=data['predicted_ee'])

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
    """
    Confidence Interval:
        https://github.com/cmrivers/epipy/blob/master/epipy/analyses.py
        https://www.medcalc.org/calc/diagnostic_test.php
        https://www.wikihow.com/Calculate-95%25-Confidence-Interval-for-a-Test%27s-Sensitivity
    """

    @staticmethod
    def evaluation_statistics(confusion_matrix):

        if confusion_matrix.shape == (3, 3):

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
            accuracy_se = np.sqrt((accuracy * (1 - accuracy)) / total_classifications)
            accuracy_confidence_interval = (accuracy - (1.96 * accuracy_se), accuracy + (1.96 * accuracy_se))

            # Calculate Precision for label 1
            precisionA = tpa / (tpa + eba + eca)

            # Calculate Sensitivity for label 1
            sensitivityA = tpa / (tpa + eab + eac)
            senA_se = np.sqrt((sensitivityA * (1 - sensitivityA)) / (tpa + eab + eac))
            sensitivityA_confidence_interval = (sensitivityA - (1.96 * senA_se), sensitivityA + (1.96 * senA_se))

            # Calculate Specificity for label 1
            tna = tpb + ebc + ecb + tpc
            specificityA = tna / (tna + eba + eca)
            specA_se = np.sqrt((specificityA * (1 - specificityA)) / (tna + eba + eca))
            specificityA_confidence_interval = (specificityA - (1.96 * specA_se), specificityA + (1.96 * specA_se))

            # Calculate Precision for label 2
            precisionB = tpb / (tpb + eab + ecb)

            # Calculate Sensitivity for label 2
            sensitivityB = tpb / (tpb + eba + ebc)
            senB_se = np.sqrt((sensitivityB * (1 - sensitivityB)) / (tpb + eba + ebc))
            sensitivityB_confidence_interval = (sensitivityB - (1.96 * senB_se), sensitivityB + (1.96 * senB_se))

            # Calculate Specificity for label 2
            tnb = tpa + eac + eca + tpc
            specificityB = tnb / (tnb + eab + ecb)
            specB_se = np.sqrt((specificityB * (1 - specificityB)) / (tnb + eab + ecb))
            specificityB_confidence_interval = (specificityB - (1.96 * specB_se), specificityB + (1.96 * specB_se))

            # Calculate Precision for label 2
            precisionC = tpc / (tpc + eac + ebc)

            # Calculate Sensitivity for label 2
            sensitivityC = tpc / (tpc + eca + ecb)
            senC_se = np.sqrt((sensitivityC * (1 - sensitivityC)) / (tpc + eca + ecb))
            sensitivityC_confidence_interval = (sensitivityC - (1.96 * senC_se), sensitivityC + (1.96 * senC_se))

            # Calculate Specificity for label 2
            tnc = tpa + eab + eba + tpb
            specificityC = tnc / (tnc + eac + ebc)
            specC_se = np.sqrt((specificityC * (1 - specificityC)) / (tnc + eac + ebc))
            specificityC_confidence_interval = (specificityC - (1.96 * specC_se), specificityC + (1.96 * specC_se))

            round_digits = 5

            sensitivityA_confidence_interval = (round(sensitivityA_confidence_interval[0], round_digits),
                                                round(sensitivityA_confidence_interval[1], round_digits))
            sensitivityB_confidence_interval = (round(sensitivityB_confidence_interval[0], round_digits),
                                                round(sensitivityB_confidence_interval[1], round_digits))
            sensitivityC_confidence_interval = (round(sensitivityC_confidence_interval[0], round_digits),
                                                round(sensitivityC_confidence_interval[1], round_digits))
            specificityA_confidence_interval = (round(specificityA_confidence_interval[0], round_digits),
                                                round(specificityA_confidence_interval[1], round_digits))
            specificityB_confidence_interval = (round(specificityB_confidence_interval[0], round_digits),
                                                round(specificityB_confidence_interval[1], round_digits))
            specificityC_confidence_interval = (round(specificityC_confidence_interval[0], round_digits),
                                                round(specificityC_confidence_interval[1], round_digits))

            return {
                'accuracy': round(accuracy, round_digits),
                'accuracy_ci': (round(accuracy_confidence_interval[0], round_digits),
                                round(accuracy_confidence_interval[1], round_digits)),
                'precision': [round(precisionA, round_digits), round(precisionB, round_digits),
                              round(precisionC, round_digits)],
                'recall': [round(sensitivityA, round_digits), round(sensitivityB, round_digits),
                           round(sensitivityC, round_digits)],
                'sensitivity': [round(sensitivityA, round_digits), round(sensitivityB, round_digits),
                                round(sensitivityC, round_digits)],
                'specificity': [round(specificityA, round_digits), round(specificityB, round_digits),
                                round(specificityC, round_digits)],
                'sensitivity_ci': [sensitivityA_confidence_interval, sensitivityB_confidence_interval,
                                   sensitivityC_confidence_interval],
                'specificity_ci': [specificityA_confidence_interval, specificityB_confidence_interval,
                                   specificityC_confidence_interval]
            }

        elif confusion_matrix.shape == (2, 2):

            tpa = confusion_matrix[0, 0]
            tpb = confusion_matrix[1, 1]
            eab = confusion_matrix[0, 1]
            eba = confusion_matrix[1, 0]

            # Calculate accuracy for label 1
            total_classifications = sum(sum(confusion_matrix))
            accuracy = (tpa + tpb) / total_classifications
            accuracy_se = np.sqrt((accuracy * (1 - accuracy)) / total_classifications)
            accuracy_confidence_interval = (accuracy - (1.96 * accuracy_se), accuracy + (1.96 * accuracy_se))

            # Calculate Precision for label 1
            precisionA = tpa / (tpa + eba)

            # Calculate Sensitivity for label 1
            sensitivityA = tpa / (tpa + eab)
            senA_se = np.sqrt((sensitivityA * (1 - sensitivityA)) / (tpa + eab))
            sensitivityA_confidence_interval = (sensitivityA - (1.96 * senA_se), sensitivityA + (1.96 * senA_se))

            # Calculate Specificity for label 1
            tna = tpb
            specificityA = tna / (tna + eba)
            specA_se = np.sqrt((specificityA * (1 - specificityA)) / (tna + eba))
            specificityA_confidence_interval = (specificityA - (1.96 * specA_se), specificityA + (1.96 * specA_se))

            # Calculate Precision for label 2
            precisionB = tpb / (tpb + eab)

            # Calculate Sensitivity for label 2
            sensitivityB = tpb / (tpb + eba)
            senB_se = np.sqrt((sensitivityB * (1 - sensitivityB)) / (tpb + eba))
            sensitivityB_confidence_interval = (sensitivityB - (1.96 * senB_se), sensitivityB + (1.96 * senB_se))

            # Calculate Specificity for label 2
            tnb = tpa
            specificityB = tnb / (tnb + eab)
            specB_se = np.sqrt((specificityB * (1 - specificityB)) / (tnb + eab))
            specificityB_confidence_interval = (specificityB - (1.96 * specB_se), specificityB + (1.96 * specB_se))

            round_digits = 5

            sensitivityA_confidence_interval = (round(sensitivityA_confidence_interval[0], round_digits),
                                                round(sensitivityA_confidence_interval[1], round_digits))
            sensitivityB_confidence_interval = (round(sensitivityB_confidence_interval[0], round_digits),
                                                round(sensitivityB_confidence_interval[1], round_digits))
            specificityA_confidence_interval = (round(specificityA_confidence_interval[0], round_digits),
                                                round(specificityA_confidence_interval[1], round_digits))
            specificityB_confidence_interval = (round(specificityB_confidence_interval[0], round_digits),
                                                round(specificityB_confidence_interval[1], round_digits))

            return {
                'accuracy': round(accuracy, round_digits),
                'accuracy_ci': (round(accuracy_confidence_interval[0], round_digits),
                                round(accuracy_confidence_interval[1], round_digits)),
                'precision': [round(precisionA, round_digits), round(precisionB, round_digits)],
                'recall': [round(sensitivityA, round_digits), round(sensitivityB, round_digits)],
                'sensitivity': [round(sensitivityA, round_digits), round(sensitivityB, round_digits)],
                'specificity': [round(specificityA, round_digits), round(specificityB, round_digits)],
                'sensitivity_ci': [sensitivityA_confidence_interval, sensitivityB_confidence_interval],
                'specificity_ci': [specificityA_confidence_interval, specificityB_confidence_interval]
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

        plt.savefig(output_filename, dpi=1200)

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


class Average_Stats:

    @staticmethod
    def evaluate_average_measures(data, epoch):

        def get_averaged_df(dataset, count_field, new_col, multiplyer):
            dataset_count = dataset.groupby(['subject'])[count_field].count().reset_index(name=new_col)
            dataset_count[new_col] *= (multiplyer / (60 * 60))
            return dataset_count

        def get_average_counted_df(data_actual, data_predict, mul):
            return pd.merge(get_averaged_df(data_actual, 'waist_ee', 'actual_time', mul),
                            get_averaged_df(data_predict, 'predicted_ee', 'predicted_time', mul),
                            on='subject', how='outer')

        round_digits = 2
        mul = int(epoch.split('Epoch')[1])

        # Evaluate SB
        df_sb = get_average_counted_df(data.loc[data['waist_ee'] <= 1.5], data.loc[data['predicted_ee'] <= 1.5], mul)
        # df_sb.to_csv(output_folder_path + output_title + '_sb_averaged.csv', index=False)
        sb_actual_avg = str(round(df_sb['actual_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb['actual_time'].std(), round_digits))
        sb_predicted_avg = str(round(df_sb['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb['predicted_time'].std(), round_digits))

        # Evaluate LPA
        df_lpa = get_average_counted_df(data.loc[(data['waist_ee'] > 1.5) & (data['waist_ee'] < 3)],
                                        data.loc[(data['predicted_ee'] > 1.5) & (data['predicted_ee'] < 3)], mul)
        # df_lpa.to_csv(output_folder_path + output_title + '_lpa_averaged.csv', index=False)
        lpa_actual_avg = str(round(df_lpa['actual_time'].mean(), round_digits)) + "+-" + str(
            round(df_lpa['actual_time'].std(), round_digits))
        lpa_predicted_avg = str(round(df_lpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_lpa['predicted_time'].std(), round_digits))

        # Evaluate SB+LPA
        df_sb_lpa = get_average_counted_df(data.loc[(data['waist_ee'] < 3)], data.loc[(data['predicted_ee'] < 3)], mul)
        # df_sb_lpa.to_csv(output_folder_path + output_title + '_sb_lpa_averaged.csv', index=False)
        sb_lpa_actual_avg = str(df_sb_lpa['actual_time'].mean()) + "+-" + str(df_sb_lpa['actual_time'].std())
        sb_lpa_predicted_avg = str(round(df_sb_lpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb_lpa['predicted_time'].std(), round_digits))

        # Evaluate MVPA
        df_mvpa = get_average_counted_df(data.loc[data['waist_ee'] >= 3], data.loc[data['predicted_ee'] >= 3], mul)
        # df_mvpa.to_csv(output_folder_path + output_title + '_mvpa_averaged.csv', index=False)
        mvpa_actual_avg = str(df_mvpa['actual_time'].mean()) + "+-" + str(df_mvpa['actual_time'].std())
        mvpa_predicted_avg = str(round(df_mvpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_mvpa['predicted_time'].std(), round_digits))

        return [sb_actual_avg, sb_predicted_avg], [lpa_actual_avg, lpa_predicted_avg], [sb_lpa_actual_avg,
                                                                                        sb_lpa_predicted_avg], [
                   mvpa_actual_avg, mvpa_predicted_avg]

    @staticmethod
    def evaluate_average_measures_for_categorical(data, epoch):
        def get_averaged_df(dataset, count_field, new_col, multiplyer):
            dataset_count = dataset.groupby(['subject'])[count_field].count().reset_index(name=new_col)
            dataset_count[new_col] *= (multiplyer / (60 * 60))
            return dataset_count

        def get_average_counted_df(data_actual, data_predict, mul):
            return pd.merge(get_averaged_df(data_actual, 'waist_ee', 'actual_time', mul),
                            get_averaged_df(data_predict, 'predicted_category', 'predicted_time', mul),
                            on='subject', how='outer')

        round_digits = 2
        mul = int(epoch.split('Epoch')[1])

        # Evaluate SB
        df_sb = get_average_counted_df(data.loc[data['waist_ee'] <= 1.5], data.loc[data['predicted_category'] == 1],
                                       mul)
        # df_sb.to_csv(output_folder_path + output_title + '_sb_averaged.csv', index=False)
        sb_actual_avg = str(df_sb['actual_time'].mean()) + "+-" + str(df_sb['actual_time'].std())
        sb_predicted_avg = str(round(df_sb['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb['predicted_time'].std(), round_digits))

        # Evaluate LPA
        df_lpa = get_average_counted_df(data.loc[(data['waist_ee'] > 1.5) & (data['waist_ee'] < 3)],
                                        data.loc[data['predicted_category'] == 2], mul)
        # df_lpa.to_csv(output_folder_path + output_title + '_lpa_averaged.csv', index=False)
        lpa_actual_avg = str(df_lpa['actual_time'].mean()) + "+-" + str(df_lpa['actual_time'].std())
        lpa_predicted_avg = str(round(df_lpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_lpa['predicted_time'].std(), round_digits))

        # Evaluate SB+LPA
        df_sb_lpa = get_average_counted_df(data.loc[(data['waist_ee'] < 3)], data.loc[(data['predicted_category'] != 3)], mul)
        # df_sb_lpa.to_csv(output_folder_path + output_title + '_sb_lpa_averaged.csv', index=False)
        sb_lpa_actual_avg = str(df_sb_lpa['actual_time'].mean()) + "+-" + str(df_sb_lpa['actual_time'].std())
        sb_lpa_predicted_avg = str(round(df_sb_lpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_sb_lpa['predicted_time'].std(), round_digits))

        # Evaluate MVPA
        df_mvpa = get_average_counted_df(data.loc[data['waist_ee'] >= 3], data.loc[data['predicted_category'] == 3],
                                         mul)
        # df_mvpa.to_csv(output_folder_path + output_title + '_mvpa_averaged.csv', index=False)
        mvpa_actual_avg = str(df_mvpa['actual_time'].mean()) + "+-" + str(df_mvpa['actual_time'].std())
        mvpa_predicted_avg = str(round(df_mvpa['predicted_time'].mean(), round_digits)) + "+-" + str(
            round(df_mvpa['predicted_time'].std(), round_digits))

        return [sb_actual_avg, sb_predicted_avg], [lpa_actual_avg, lpa_predicted_avg], [sb_lpa_actual_avg,
                                                                                        sb_lpa_predicted_avg], [
                   mvpa_actual_avg, mvpa_predicted_avg]

    @staticmethod
    def evaluate_average_measures_controller(data, epoch, output_title, output_folder_path, is_categorical=False):

        def evaluate(category, filtered_data, epoch, assessment_result):

            if not is_categorical:
                sb, lpa, sb_lpa, mvpa = Average_Stats.evaluate_average_measures(filtered_data, epoch)
            else:
                sb, lpa, sb_lpa, mvpa = Average_Stats.evaluate_average_measures_for_categorical(filtered_data, epoch)

            assessment_result += '\n\n' + category + ' Assessment of Average time\n\n'
            assessment_result += 'SB actual:\t' + sb[0] + '\n'
            assessment_result += 'SB predicted:\t' + sb[1] + '\n'
            assessment_result += 'LPA actual:\t' + lpa[0] + '\n'
            assessment_result += 'LPA predicted:\t' + lpa[1] + '\n'
            assessment_result += 'SB+LPA actual:\t' + sb_lpa[0] + '\n'
            assessment_result += 'SB+LPA predicted:\t' + sb_lpa[1] + '\n'
            assessment_result += 'MVPA actual:\t' + mvpa[0] + '\n'
            assessment_result += 'MVPA predicted:\t' + mvpa[1] + '\n'

            return assessment_result

        results_output_str = ''
        general_user_details = pd.read_csv('E:/Projects/accelerometer-project/analyze/user_details.csv')
        data = pd.merge(data, general_user_details, on='subject', how='outer')

        # Overall evaluation
        results_output_str = evaluate('Overall', data, epoch, results_output_str)

        # Do the evaluation for Sex
        male_data = data.loc[data['gender'] == 'Male']
        results_output_str = evaluate('Male', male_data, epoch, results_output_str)

        female_data = data.loc[data['gender'] == 'Female']
        results_output_str = evaluate('Female', female_data, epoch, results_output_str)

        female_data = data.loc[data['gender'] == 'Other']
        results_output_str = evaluate('Other', female_data, epoch, results_output_str)

        # Do the evaluation based on BMI catagory
        underweight_data = data.loc[data['bmi'] < 18.5]
        results_output_str = evaluate('BMI - underweight', underweight_data, epoch, results_output_str)

        normal_data = data.loc[(data['bmi'] >= 18.5) & (data['bmi'] < 25)]
        results_output_str = evaluate('BMI - normal weight', normal_data, epoch, results_output_str)

        overweight_data = data.loc[(data['bmi'] >= 25) & (data['bmi'] < 30)]
        results_output_str = evaluate('BMI - overweight', overweight_data, epoch, results_output_str)

        obesity_data = data.loc[data['bmi'] >= 30]
        results_output_str = evaluate('BMI - obesity', obesity_data, epoch, results_output_str)

        results_output_filename = output_folder_path + output_title + '_average_time_assessment.txt'
        Utils.print_assessment_results(results_output_filename, results_output_str)


class Utils:

    @staticmethod
    def print_assessment_results(output_filename, result_string):
        with open(output_filename, "w") as text_file:
            text_file.write(result_string)
