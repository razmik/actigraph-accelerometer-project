import pandas as pd
from os import listdir
from os.path import join


def load_df(folder_name):
    results_file = [f for f in listdir(folder_name) if '.csv' in f][0]
    return pd.read_csv(join(folder_name, results_file))


def get_PA_in_hours(df_all):

    # SB
    sb_reference = "{}#{}".format(round(df_all['actual_SB'].mean() / 60, 1), round(df_all['actual_SB'].std() / 60, 1))
    sb_cnn60 = "{}#{}".format(round(df_all['predicted_SB'].mean() / 60, 1), round(df_all['predicted_SB'].std() / 60, 1))

    # LPA
    lpa_reference = "{}#{}".format(round(df_all['actual_LPA'].mean() / 60, 1),
                                   round(df_all['actual_LPA'].std() / 60, 1))
    lpa_cnn60 = "{}#{}".format(round(df_all['predicted_LPA'].mean() / 60, 1),
                               round(df_all['predicted_LPA'].std() / 60, 1))

    # SB+LPA
    sblpa_reference = "{}#{}".format(round((df_all['actual_SB'] + df_all['actual_LPA']).mean() / 60, 1),
                                     round((df_all['actual_SB'] + df_all['actual_LPA']).std() / 60, 1))
    sblpa_cnn60 = "{}#{}".format(round((df_all['predicted_SB'] + df_all['predicted_LPA']).mean() / 60, 1),
                                 round((df_all['predicted_SB'] + df_all['predicted_LPA']).std() / 60, 1))

    # MVPA
    mvpa_reference = "{}#{}".format(round(df_all['actual_MVPA'].mean() / 60, 1),
                                    round(df_all['actual_MVPA'].std() / 60, 1))
    mvpa_cnn60 = "{}#{}".format(round(df_all['predicted_MVPA'].mean() / 60, 1),
                                round(df_all['predicted_MVPA'].std() / 60, 1))

    # Total
    total_time = sblpa_reference + mvpa_reference

    return sb_reference, sb_cnn60, lpa_reference, lpa_cnn60, sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60, total_time


def get_PA_in_hours_HBLR(df_all):

    # SBLPA
    sblpa_reference = "{}#{}".format(round(df_all['actual_SBLPA'].mean() / 60, 1), round(df_all['actual_SBLPA'].std() / 60, 1))
    sblpa_cnn60 = "{}#{}".format(round(df_all['predicted_SBLPA'].mean() / 60, 1), round(df_all['predicted_SBLPA'].std() / 60, 1))

    # MVPA
    mvpa_reference = "{}#{}".format(round(df_all['actual_MVPA'].mean() / 60, 1),
                                    round(df_all['actual_MVPA'].std() / 60, 1))
    mvpa_cnn60 = "{}#{}".format(round(df_all['predicted_MVPA'].mean() / 60, 1),
                                round(df_all['predicted_MVPA'].std() / 60, 1))

    # Total
    total_time = sblpa_reference + mvpa_reference

    return sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60, total_time


def append_data_results(result_data, df_3000_filt, df_6000_filt, df_hblr_filt, title):

    assert df_3000_filt.shape[0] == df_6000_filt.shape[0] == df_hblr_filt.shape[0]

    set_val = '{} (n={})'.format(title, df_3000_filtered.shape[0])

    sb_reference30, sb_cnn30, lpa_reference30, lpa_cnn30, sblpa_reference30, sblpa_cnn30, mvpa_reference30, mvpa_cnn30, total_30 = get_PA_in_hours(
        df_3000_filt)
    sb_reference60, sb_cnn60, lpa_reference60, lpa_cnn60, sblpa_reference60, sblpa_cnn60, mvpa_reference60, mvpa_cnn60, total_60 = get_PA_in_hours(
        df_6000_filt)
    sblpa_referencehb, sblpa_hb, mvpa_referencebh, mvpa_hb, total_hb = get_PA_in_hours_HBLR(df_hblr_filt)

    result_data.append([set_val, 'SB', sb_reference30, sb_cnn30, sb_reference60, sb_cnn60, '', ''])
    result_data.append([set_val, 'LPA', lpa_reference30, lpa_cnn30, lpa_reference60, lpa_cnn60, '', ''])
    result_data.append([set_val, 'SB+LPA', sblpa_reference30, sblpa_cnn30, sblpa_reference60, sblpa_cnn60, sblpa_referencehb, sblpa_hb])
    result_data.append([set_val, 'MVPA', mvpa_reference30, mvpa_cnn30, mvpa_reference60, mvpa_cnn60, mvpa_referencebh, mvpa_hb])

    return result_data


if __name__ == '__main__':

    training_version = '1-12_Dec'
    eval_category = ['train_test', 'test']

    SUBJECT_DETAIL_FILE = 'user_details.xlsx'
    HBLR_ROOT = "E:/Projects/Accelerometer-project_Rashmika/supervised_learning/hb-lr-eval/outputs"
    CNN_ROOT = "E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}".format(training_version)

    participant_df = pd.read_excel(SUBJECT_DETAIL_FILE).groupby('Participant ID').first().reset_index()

    result_col_names = ['Set', 'PA category', 'Reference_cnn30', 'cnn_30', 'Reference_cnn60', 'cnn_60', 'Reference_hblr', 'hblr']

    for data_cat in eval_category:

        class_3000_df = load_df(join(CNN_ROOT, 'classification', 'window-3000-overlap-1500', 'individual_results', data_cat))
        df_3000_summary = pd.merge(class_3000_df, participant_df, left_on='participant', right_on='Participant ID', how='left')

        class_6000_df = load_df(join(CNN_ROOT, 'classification', 'window-6000-overlap-3000', 'individual_results', data_cat))
        df_6000_summary = pd.merge(class_6000_df, participant_df, left_on='participant', right_on='Participant ID', how='left')

        class_hblr_df = load_df(join(HBLR_ROOT, data_cat, 'individual'))
        df_hblr_summary = pd.merge(class_hblr_df, participant_df, left_on='participant', right_on='Participant ID', how='left')

        result_data = []

        # All Participants
        df_3000_filtered = df_3000_summary[:]
        df_6000_filtered = df_6000_summary[:]
        df_hblr_filtered = df_hblr_summary[:]
        result_data = append_data_results(result_data, df_3000_filtered, df_6000_filtered, df_hblr_filtered, 'All')

        # Male
        df_3000_filtered = df_3000_summary.loc[df_3000_summary['gender'] == 'Male']
        df_6000_filtered = df_6000_summary.loc[df_6000_summary['gender'] == 'Male']
        df_hblr_filtered = df_hblr_summary.loc[df_hblr_summary['gender'] == 'Male']
        result_data = append_data_results(result_data, df_3000_filtered, df_6000_filtered, df_hblr_filtered, 'Male')

        # Female
        df_3000_filtered = df_3000_summary.loc[df_3000_summary['gender'] == 'Female']
        df_6000_filtered = df_6000_summary.loc[df_6000_summary['gender'] == 'Female']
        df_hblr_filtered = df_hblr_summary.loc[df_hblr_summary['gender'] == 'Female']
        result_data = append_data_results(result_data, df_3000_filtered, df_6000_filtered, df_hblr_filtered, 'Female')

        # Other
        df_3000_filtered = df_3000_summary.loc[df_3000_summary['gender'] == 'Other']
        df_6000_filtered = df_6000_summary.loc[df_6000_summary['gender'] == 'Other']
        df_hblr_filtered = df_hblr_summary.loc[df_hblr_summary['gender'] == 'Other']
        result_data = append_data_results(result_data, df_3000_filtered, df_6000_filtered, df_hblr_filtered, 'Other')

        # Normal
        df_3000_filtered = df_3000_summary.loc[df_3000_summary['bmi'] < 25]
        df_6000_filtered = df_6000_summary.loc[df_6000_summary['bmi'] < 25]
        df_hblr_filtered = df_hblr_summary.loc[df_hblr_summary['bmi'] < 25]
        result_data = append_data_results(result_data, df_3000_filtered, df_6000_filtered, df_hblr_filtered, 'Normal')

        # Overweight
        df_3000_filtered = df_3000_summary.loc[(25 <= df_3000_summary['bmi']) & (df_3000_summary['bmi'] < 30)]
        df_6000_filtered = df_6000_summary.loc[(25 <= df_6000_summary['bmi']) & (df_6000_summary['bmi'] < 30)]
        df_hblr_filtered = df_hblr_summary.loc[(25 <= df_hblr_summary['bmi']) & (df_hblr_summary['bmi'] < 30)]
        result_data = append_data_results(result_data, df_3000_filtered, df_6000_filtered, df_hblr_filtered, 'Overweight')

        # Obese
        df_3000_filtered = df_3000_summary.loc[30 <= df_3000_summary['bmi']]
        df_6000_filtered = df_6000_summary.loc[30 <= df_6000_summary['bmi']]
        df_hblr_filtered = df_hblr_summary.loc[30 <= df_hblr_summary['bmi']]
        result_data = append_data_results(result_data, df_3000_filtered, df_6000_filtered, df_hblr_filtered, 'Obese')

        # Save table 1 output
        pd.DataFrame(result_data, columns=result_col_names).to_csv('Table_1_Weekly_Analysis_{}_{}.csv'.format(training_version, data_cat), index=None)

    print('Completed')
