import pandas as pd
from os import listdir
from os.path import join
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


def load_df(folder_name):
    results_file = [f for f in listdir(folder_name) if '.csv' in f][0]
    return pd.read_csv(join(folder_name, results_file))


if __name__ == '__main__':

    training_version = '1-12_Dec'
    eval_category = ['test']

    CNN_ROOT = "E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}".format(training_version)

    for data_cat in eval_category:

        class_6000_df = load_df(join(CNN_ROOT, 'classification', 'window-6000-overlap-3000', 'individual_results', data_cat))

        # class_6000_df.boxplot(['actual_MVPA', 'predicted_MVPA'])
        # plt.show()

        anova = f_oneway(class_6000_df['actual_SB'], class_6000_df['predicted_SB'])

        print(anova)


