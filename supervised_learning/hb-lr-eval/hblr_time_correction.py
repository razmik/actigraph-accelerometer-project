import pandas as pd
from os import listdir, remove
from os.path import join


def load_df(folder_name):
    results_file = [f for f in listdir(folder_name) if '.csv' in f][0]
    return pd.read_csv(join(folder_name, results_file)), join(folder_name, results_file)


if __name__ == '__main__':

    training_version = '1-12_Dec'
    eval_category = ['train_test', 'test']

    CNN_ROOT = "E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}/classification/window-6000-overlap-3000/individual_results/".format(training_version)
    HBLR_ROOT = "E:/Projects/Accelerometer-project_Rashmika/supervised_learning/hb-lr-eval/outputs"

    for data_cat in eval_category:

        class_6000_df, _ = load_df(join(CNN_ROOT, data_cat))
        class_hblr_df, class_hblr_df_filename = load_df(join(HBLR_ROOT, data_cat, 'individual'))

        for index, row in class_hblr_df.iterrows():

            participant = row['participant']

            # Correct SB+LPA
            min_diff = class_6000_df.loc[class_6000_df['participant'] == participant]['actual_SBLPA'].iloc[0] - row['actual_SBLPA']
            class_hblr_df.loc[class_hblr_df['participant'] == participant, 'actual_SBLPA'] = row['actual_SBLPA'] + min_diff
            class_hblr_df.loc[class_hblr_df['participant'] == participant, 'predicted_SBLPA'] = row['predicted_SBLPA'] + min_diff

            # Correct MVPA
            min_diff = class_6000_df.loc[class_6000_df['participant'] == participant]['actual_MVPA'].iloc[0] - row['actual_MVPA']
            class_hblr_df.loc[class_hblr_df['participant'] == participant, 'actual_MVPA'] = row['actual_MVPA'] + min_diff
            class_hblr_df.loc[class_hblr_df['participant'] == participant, 'predicted_MVPA'] = row['predicted_MVPA'] + min_diff

        # Remove old file
        remove(class_hblr_df_filename)

        # Crate new file with the same name
        class_hblr_df.to_csv(class_hblr_df_filename.replace('.csv', '_corrected.csv'), index=None)

    print('Completed')
