import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BlandAltman:

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
    def bland_altman_paired_plot_tested(dataframe, plot_title, plot_number, log_transformed=False, min_count_regularise=False):

        if min_count_regularise:
            dataframe = BlandAltman.get_min_regularised_data_per_subject(dataframe)

        if log_transformed:
            dataframe = dataframe.assign(waist_ee_transformed=np.log10(dataframe['waist_ee_transformed']))
            dataframe = dataframe.assign(predicted_ee_transformed=np.log10(dataframe['predicted_ee_transformed']))

        dataframe = dataframe.assign(mean=np.mean([dataframe.as_matrix(columns=['waist_ee_transformed']),
                                     dataframe.as_matrix(columns=['predicted_ee_transformed'])], axis=0))
        dataframe = dataframe.assign(diff=dataframe['waist_ee_transformed'] - dataframe['predicted_ee_transformed'])

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

        # print('SS_between', SSbetween)
        # print('SS_within', SSwithin)
        # print('SS_total', SStotal)

        MSbetween = SSbetween / DFbetween
        MSwithin = SSwithin / DFwithin
        # F = MSbetween / MSwithin
        # p = stats.f.sf(F, DFbetween, DFwithin)

        # print('MS_between', MSbetween)
        # print('MS_within', MSwithin)

        # print('F', F)

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

        print('Upper LOA:', upper_loa)
        print('Mean LOA:', mean_bias)
        print('Lower LOA:', lower_loa)

        plt.figure(plot_number)
        plt.title(plot_title)
        plt.scatter(dataframe['mean'], dataframe['diff'])  # x - mean | y - diff
        plt.axhline(mean_bias, color='gray', linestyle='--')
        plt.axhline(upper_loa, color='gray', linestyle='--')
        plt.axhline(lower_loa, color='gray', linestyle='--')
        plt.xlabel('Mean between assessments')
        plt.ylabel('Difference between assessments')

    @staticmethod
    def clean_data_points(data):
        # Remove row if reference MET value is less than 1
        data = data[data.waist_ee >= 1]

        data = data.assign(waist_ee_transformed=data['waist_ee'])
        data = data.assign(predicted_ee_transformed=data['predicted_ee'])

        data.loc[(data['predicted_ee'] <= 0.0), 'predicted_ee_transformed'] = 1

        return data
