import numpy as np
import pandas as pd


def numeric_sequence_characteristics(field):
    print('\n*', field, 'characteristics')
    print('Male', master_user_list.loc[master_user_list['gender'] == 'Male'][field].mean(),
          '+-', master_user_list.loc[master_user_list['gender'] == 'Male'][field].std())
    print('Female', master_user_list.loc[master_user_list['gender'] == 'Female'][field].mean(),
          '+-', master_user_list.loc[master_user_list['gender'] == 'Female'][field].std())
    print('Other', master_user_list.loc[master_user_list['gender'] == 'Other'][field].mean(),
          '+-', master_user_list.loc[master_user_list['gender'] == 'Other'][field].std())


if __name__ == '__main__':

    root_folder = 'C:/Users\pc\OneDrive - LA TROBE UNIVERSITY\Accelerometer Project\Paper 1 - Assessment\Additional Information\csv/'.replace('\\', '/')
    lsm1_gender_fname = 'LSM1.csv'
    lsm2_gender_fname = 'LSM2.csv'
    additional_data_fname = 'LSM_additional_data.csv'
    lsm1_activitytimeline_fname = 'LSM1_ActiveTimeline_Details_v1.csv'
    lsm2_activitytimeline_fname = 'LSM2_ActiveTimeline_Details_v1.csv'

    # Compose the user list
    df_lsm1_at = pd.read_csv(root_folder+lsm1_activitytimeline_fname)
    df_lsm2_at = pd.read_csv(root_folder+lsm2_activitytimeline_fname)
    lsm1_user_list = [user.split(' ')[0] for user in df_lsm1_at['Subject'].unique()]
    lsm2_user_list = [user.split(' ')[0] for user in df_lsm2_at['Subject'].unique()]

    # Master list of users
    columns = ['gender', 'age', 'height', 'weight', 'bmi']
    df_lsm1 = pd.DataFrame(index=lsm1_user_list, columns=columns)
    df_lsm2 = pd.DataFrame(index=lsm2_user_list, columns=columns)

    # Add gender and age
    df_lsm1_gender_list = pd.read_csv(root_folder+lsm1_gender_fname)
    for index, row in df_lsm1_gender_list.iterrows():
        if row['Q1'] in df_lsm1.index:
            df_lsm1.loc[row['Q1'], 'gender'] = row['Q13']
            df_lsm1.loc[row['Q1'], 'age'] = row['Q14']

    df_lsm2_gender_list = pd.read_csv(root_folder+lsm2_gender_fname)
    for index, row in df_lsm2_gender_list.iterrows():
        if row['Q1'] in df_lsm2.index:
            df_lsm2.loc[row['Q1'], 'gender'] = row['Q13']
            df_lsm2.loc[row['Q1'], 'age'] = row['Q14']

    # Add weight and height
    df_add_data = pd.read_csv(root_folder+additional_data_fname)
    df_add_data = df_add_data.set_index('lsm_id')

    for index, row in df_lsm1.iterrows():
        if index in df_add_data.index and row['height'] != '' and row['weight'] != '':
            df_lsm1.loc[index, 'height'] = df_add_data.loc[index, 'height']
            df_lsm1.loc[index, 'weight'] = df_add_data.loc[index, 'weight']
            df_lsm1.loc[index, 'bmi'] = df_add_data.loc[index, 'bmi']
        else:
            print('HW error:', index)

    for index, row in df_lsm2.iterrows():
        if index in df_add_data.index and row['height'] != '' and row['weight'] != '':
            df_lsm2.loc[index, 'height'] = df_add_data.loc[index, 'height']
            df_lsm2.loc[index, 'weight'] = df_add_data.loc[index, 'weight']
            df_lsm2.loc[index, 'bmi'] = df_add_data.loc[index, 'bmi']
        else:
            print('HW error:', index)

    # Fill in missing values with mean
    df_lsm1["height"].fillna(df_lsm1["height"].mean(), inplace=True)
    df_lsm2["height"].fillna(df_lsm2["height"].mean(), inplace=True)
    df_lsm1["weight"].fillna(df_lsm1["weight"].mean(), inplace=True)
    df_lsm2["weight"].fillna(df_lsm2["weight"].mean(), inplace=True)
    df_lsm1["age"].fillna(df_lsm1["age"].mean(), inplace=True)
    df_lsm2["age"].fillna(df_lsm2["age"].mean(), inplace=True)

    # combine master datalist
    master_user_list = df_lsm1.append(df_lsm2)
    master_user_list.age = master_user_list.age.astype(np.float32)
    # master_user_list.to_csv('temp.csv')

    # Results
    grouped_users = master_user_list.groupby('gender')
    print('Total number of participants')
    print('Male', len(master_user_list.loc[master_user_list['gender'] == 'Male']))
    print('Female', len(master_user_list.loc[master_user_list['gender'] == 'Female']))
    print('Other', len(master_user_list.loc[master_user_list['gender'] == 'Other']))

    numeric_sequence_characteristics('age')
    numeric_sequence_characteristics('height')
    numeric_sequence_characteristics('weight')
    numeric_sequence_characteristics('bmi')

    print('\n Completed.')
