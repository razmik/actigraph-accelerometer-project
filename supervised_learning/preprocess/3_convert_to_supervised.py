import pandas as pd
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Reference: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def reshape_to_windows(dataframe, window_length, input_cols, target_cols):

    # Setup output column names
    col_names_out = target_cols
    for i in range(window_length-1, -1, -1):
        col_names_out.extend(['{}(t-{})'.format(cn, i) for cn in input_cols])

    # Reshape for Windowing
    result_data = []
    for i in range(0, (dataframe.shape[0]-window_length), window_length):

        df_inputs = dataframe.loc[i:(i+window_length-1), input_cols]
        target_0_value = dataframe.at[i+window_length-1, target_cols[0]]
        target_1_value = dataframe.at[i+window_length-1, target_cols[1]]

        data_row = list(list(np.reshape(df_inputs.values, (window_length * len(input_cols), 1)).T)[0])

        result_data.append([target_0_value, target_1_value] + data_row)

    # Set the output dataframe
    return pd.DataFrame(result_data, columns=col_names_out)


if __name__ == "__main__":

    # config
    window_length = 500  # 5 second
    req_cols = ['X', 'Y', 'Z', 'vm', 'angle', 'enmo', 'waist_ee', 'waist_intensity']
    input_cols = ['X', 'Y', 'Z']
    target_cols = ['waist_ee', 'waist_intensity']

    INPUT_DATA_FOLDER = "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Raw_features/Epoch1/"
    OUTPUT_FOLDER = join(INPUT_DATA_FOLDER, 'supervised_data', 'window-{}'.format(window_length))
    if not exists(OUTPUT_FOLDER):
        makedirs(OUTPUT_FOLDER)

    all_files = [f for f in listdir(INPUT_DATA_FOLDER) if isfile(join(INPUT_DATA_FOLDER, f))]

    for f in tqdm(all_files):
        #  84%|████████▎ | 705/842 [5:39:10<1:15:31, 33.08s/it]

        df = pd.read_csv(join(INPUT_DATA_FOLDER, f), usecols=req_cols)

        # dataframe, window_length, input_cols, target_col
        supervised_data = reshape_to_windows(df, window_length, input_cols, target_cols)

        out_name = join(OUTPUT_FOLDER, f.replace('.csv', '_supervised.csv'))
        supervised_data.to_csv(out_name, index=None)

    print('Completed.')
