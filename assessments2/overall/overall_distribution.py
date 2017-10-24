import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import itertools, sys
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
from os import listdir
from os.path import isfile, join
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')
import statistical_extensions as SE


if __name__ == '__main__':

    count = 0

    input_file_path = ('E:\Data\Accelerometer_LR\staudenmayer\Epoch60/').replace('\\', '/')
    input_filenames = [f for f in listdir(input_file_path) if isfile(join(input_file_path, f))]

    for file in input_filenames:
        dataframe = pd.read_csv(input_file_path + file)
        # dataframe['subject'] = file.split('_(2016')[0]

        if count != 0:
            results = results.append(dataframe, ignore_index=True)
        else:
            results = dataframe

        count += 1

    print('SB', len(results.loc[results['waist_ee'] <= 1.5]))
    print('LPA', len(results.loc[(results['waist_ee'] > 1.5) & (results['waist_ee'] < 3)]))
    print('MVPA', len(results.loc[results['waist_ee'] > 3]))
    print('\n')

    print('Completed.')
