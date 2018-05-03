import pandas as pd
import numpy as np
import os


file_dict = {
    'hlr_mvpa': 'input/hlr_mvpa.csv',
    'silr_sb': 'input/silr_sb.csv'
}

output_root = 'preprocess/output'
out_header = ["subject", "freedson", "predicting", "diff"]


def save_df(dataframe, prefix, category):

    dataframe.to_csv(os.path.join(output_root, (prefix + '_' + category + '.csv')), columns=out_header, index=None)


for key, filename in file_dict.items():

    df = pd.read_csv(filename)
    df['diff'] = df['predicting'] - df['freedson']

    # overall
    save_df(df, key, 'overall')

    # male
    df_updated = df.loc[df['gender'] == 'Male']
    save_df(df_updated, key, 'male')

    # female
    del df_updated
    df_updated = df.loc[df['gender'] == 'Female']
    save_df(df_updated, key, 'female')

    # other
    del df_updated
    df_updated = df.loc[df['gender'] == 'Other']
    save_df(df_updated, key, 'other')

    # normal
    del df_updated
    df_updated = df.loc[df['bmi_cat'] == 'normal']
    save_df(df_updated, key, 'normal')

    # overweight
    del df_updated
    df_updated = df.loc[df['bmi_cat'] == 'overweight']
    save_df(df_updated, key, 'overweight')

    # obese
    del df_updated
    df_updated = df.loc[df['bmi_cat'] == 'obese']
    save_df(df_updated, key, 'obese')

