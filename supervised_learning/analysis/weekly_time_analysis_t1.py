import pandas as pd


SUBJECT_DETAIL_FILE = 'user_details.xlsx'
DATA_FILE = "EE_Prediction_subject_level_evaluation_v2.xlsx"


def get_values_60s(df_all):

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

    return sb_reference, sb_cnn60, lpa_reference, lpa_cnn60, sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60


def get_values_30s(df_all):

    # SB
    sb_reference = "{}#{}".format(round(df_all['actual_SB'].mean() / (4*60), 1), round(df_all['actual_SB'].std() / (4*60), 1))
    sb_cnn30 = "{}#{}".format(round(df_all['predicted_SB'].mean() / (4*60), 1), round(df_all['predicted_SB'].std() / (4*60), 1))

    # LPA
    lpa_reference = "{}#{}".format(round(df_all['actual_LPA'].mean() / (4*60), 1),
                                   round(df_all['actual_LPA'].std() / (4*60), 1))
    lpa_cnn30 = "{}#{}".format(round(df_all['predicted_LPA'].mean() / (4*60), 1),
                               round(df_all['predicted_LPA'].std() / (4*60), 1))

    # SB+LPA
    sblpa_reference = "{}#{}".format(round((df_all['actual_SB'] + df_all['actual_LPA']).mean() / (4*60), 1),
                                     round((df_all['actual_SB'] + df_all['actual_LPA']).std() / (4*60), 1))
    sblpa_cnn30 = "{}#{}".format(round((df_all['predicted_SB'] + df_all['predicted_LPA']).mean() / (4*60), 1),
                                 round((df_all['predicted_SB'] + df_all['predicted_LPA']).std() / (4*60), 1))

    # MVPA
    mvpa_reference = "{}#{}".format(round(df_all['actual_MVPA'].mean() / (4*60), 1),
                                    round(df_all['actual_MVPA'].std() / (4*60), 1))
    mvpa_cnn30 = "{}#{}".format(round(df_all['predicted_MVPA'].mean() / (4*60), 1),
                                round(df_all['predicted_MVPA'].std() / (4*60), 1))

    return sb_reference, sb_cnn30, lpa_reference, lpa_cnn30, sblpa_reference, sblpa_cnn30, mvpa_reference, mvpa_cnn30


if __name__ == '__main__':

    subject_df = pd.read_excel(SUBJECT_DETAIL_FILE).groupby('Participant ID').first().reset_index()

    result_data = []
    result_col_names = ['Set', 'PA category', 'Reference', 'cnn_60']

    data_df = pd.read_excel(DATA_FILE, sheet_name='unique_participants_wk2_60s')
    df_summary = pd.merge(data_df, subject_df, left_on='participant', right_on='Participant ID', how='left')

    # All
    df_filt = df_summary[:]
    set_val = 'All (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn60, lpa_reference, lpa_cnn60, sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60 = get_values_60s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn60])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn60])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn60])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn60])

    # Male
    df_filt = df_summary.loc[df_summary['gender'] == 'Male']
    set_val = 'Male (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn60, lpa_reference, lpa_cnn60, sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60 = get_values_60s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn60])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn60])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn60])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn60])

    # Female
    df_filt = df_summary.loc[df_summary['gender'] == 'Female']
    set_val = 'Female (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn60, lpa_reference, lpa_cnn60, sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60 = get_values_60s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn60])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn60])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn60])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn60])

    # Other
    df_filt = df_summary.loc[df_summary['gender'] == 'Other']
    set_val = 'Other (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn60, lpa_reference, lpa_cnn60, sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60 = get_values_60s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn60])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn60])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn60])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn60])

    # Normal
    df_filt = df_summary.loc[df_summary['bmi'] < 25]
    set_val = 'Normal (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn60, lpa_reference, lpa_cnn60, sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60 = get_values_60s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn60])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn60])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn60])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn60])

    # Overweight
    df_filt = df_summary.loc[(25 <= df_summary['bmi']) & (df_summary['bmi'] < 30)]
    set_val = 'Overweight (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn60, lpa_reference, lpa_cnn60, sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60 = get_values_60s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn60])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn60])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn60])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn60])

    # Obese
    df_filt = df_summary.loc[30 <= df_summary['bmi']]
    set_val = 'Obese (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn60, lpa_reference, lpa_cnn60, sblpa_reference, sblpa_cnn60, mvpa_reference, mvpa_cnn60 = get_values_60s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn60])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn60])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn60])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn60])

    #Save
    pd.DataFrame(result_data, columns=result_col_names).to_csv('weekly_analysis_60s.csv', index=None)




    result_data = []
    result_col_names = ['Set', 'PA category', 'Reference', 'cnn_30']

    data_df = pd.read_excel(DATA_FILE, sheet_name='unique_participants_wk2_30s')
    df_summary = pd.merge(data_df, subject_df, left_on='participant', right_on='Participant ID', how='left')

    # All
    df_filt = df_summary[:]
    set_val = 'All (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn30, lpa_reference, lpa_cnn30, sblpa_reference, sblpa_cnn30, mvpa_reference, mvpa_cnn30 = get_values_30s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn30])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn30])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn30])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn30])

    # Male
    df_filt = df_summary.loc[df_summary['gender'] == 'Male']
    set_val = 'Male (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn30, lpa_reference, lpa_cnn30, sblpa_reference, sblpa_cnn30, mvpa_reference, mvpa_cnn30 = get_values_30s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn30])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn30])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn30])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn30])

    # Female
    df_filt = df_summary.loc[df_summary['gender'] == 'Female']
    set_val = 'Female (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn30, lpa_reference, lpa_cnn30, sblpa_reference, sblpa_cnn30, mvpa_reference, mvpa_cnn30 = get_values_30s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn30])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn30])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn30])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn30])

    # Other
    df_filt = df_summary.loc[df_summary['gender'] == 'Other']
    set_val = 'Other (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn30, lpa_reference, lpa_cnn30, sblpa_reference, sblpa_cnn30, mvpa_reference, mvpa_cnn30 = get_values_30s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn30])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn30])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn30])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn30])

    # Normal
    df_filt = df_summary.loc[df_summary['bmi'] < 25]
    set_val = 'Normal (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn30, lpa_reference, lpa_cnn30, sblpa_reference, sblpa_cnn30, mvpa_reference, mvpa_cnn30 = get_values_30s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn30])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn30])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn30])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn30])

    # Overweight
    df_filt = df_summary.loc[(25 <= df_summary['bmi']) & (df_summary['bmi'] < 30)]
    set_val = 'Overweight (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn30, lpa_reference, lpa_cnn30, sblpa_reference, sblpa_cnn30, mvpa_reference, mvpa_cnn30 = get_values_30s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn30])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn30])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn30])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn30])

    # Obese
    df_filt = df_summary.loc[30 <= df_summary['bmi']]
    set_val = 'Obese (n={})'.format(df_filt.shape[0])
    sb_reference, sb_cnn30, lpa_reference, lpa_cnn30, sblpa_reference, sblpa_cnn30, mvpa_reference, mvpa_cnn30 = get_values_30s(df_filt)
    result_data.append([set_val, 'SB', sb_reference, sb_cnn30])
    result_data.append([set_val, 'LPA', lpa_reference, lpa_cnn30])
    result_data.append([set_val, 'SB+LPA', sblpa_reference, sblpa_cnn30])
    result_data.append([set_val, 'MVPA', mvpa_reference, mvpa_cnn30])

    #Save
    pd.DataFrame(result_data, columns=result_col_names).to_csv('weekly_analysis_30s.csv', index=None)

    print('Completed')
