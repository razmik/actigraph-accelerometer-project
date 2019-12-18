import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from os import listdir, makedirs
from os.path import isfile, join, exists
from scipy.stats.stats import pearsonr
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score, confusion_matrix
import supervised_learning.modeling.statistical_extensions as SE


def predict(data):
    # ENMO is acceleration in milig's
    data['predicted_ee'] = ((0.0320 * data['enmo'] * 1000) + 7.28) / 3.5
    return data


def get_time_in(labels, time_epoch=6000):

    unique, counts = np.unique(labels, return_counts=True)
    outcomes = {}
    for i, j in zip(unique, counts):
        outcomes[i] = j

    SB = outcomes[0] / (6000 / time_epoch) if 0 in outcomes else 0
    LPA = outcomes[1] / (6000 / time_epoch) if 1 in outcomes else 0
    MVPA = outcomes[2] / (6000 / time_epoch) if 2 in outcomes else 0

    return SB, LPA, MVPA


def evaluate_regression_modal_overall(results_df, REGRESSION_RESULTS_FOLDER):

    # Evaluate against test data
    print('Evaluating - Overall model ...')

    results_descriptions = []

    # print('Model Prediction')
    y_pred_test = results_df['predicted_ee']
    y_test = results_df['waist_ee']
    ID_test = results_df['subject']

    # print('Model Evaluation - plot regression plot')
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test)
    plt.xlabel('Actual EE')
    plt.ylabel('Predicted EE')
    plt.savefig(join(REGRESSION_RESULTS_FOLDER, 'actual_vs_predicted_met.png'))
    plt.clf()
    plt.close()

    # print('Model Evaluation - Regression stats')
    corr = pearsonr(y_test, y_pred_test)

    results_descriptions.append('\n\n -------RESULTS-------\n\n')
    results_descriptions.append('Pearsons Correlation = {}'.format(corr))
    results_descriptions.append('MSE - {}'.format(mean_squared_error(y_test, y_pred_test)))
    results_descriptions.append('RMSE - {}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
    results_descriptions.append('R2 Error - {}'.format(r2_score(y_test, y_pred_test)))
    results_descriptions.append('Explained Variance Score - {}'.format(explained_variance_score(y_test, y_pred_test)))

    # print('Model Evaluation - Regression to Intensity')
    class_names = ['SED+LPA', 'MVPA']
    y_test_ai = SE.EnergyTransform.met_to_intensity_sblpa_mvpa(y_test)
    y_pred_test_ai = SE.EnergyTransform.met_to_intensity_sblpa_mvpa(y_pred_test)

    cnf_matrix = confusion_matrix(y_test_ai, y_pred_test_ai)

    stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)

    assessment_result = 'Classes' + '\t' + str(class_names) + '\t' + '\n'
    assessment_result += 'Accuracy' + '\t' + str(stats['accuracy']) + '\t' + str(stats['accuracy_ci']) + '\n'
    assessment_result += 'Sensitivity' + '\t' + str(stats['sensitivity']) + '\n'
    assessment_result += 'Sensitivity CI' + '\t' + str(stats['sensitivity_ci']) + '\n'
    assessment_result += 'Specificity' + '\t' + str(stats['specificity']) + '\n'
    assessment_result += 'Specificity CI' + '\t' + str(stats['specificity_ci']) + '\n'

    results_descriptions.append(assessment_result)

    SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title='CM',
                                          output_filename=join(REGRESSION_RESULTS_FOLDER, 'confusion_matrix.png'))

    results_df = pd.DataFrame(
        {'subject': ID_test,
         'waist_ee': y_test,
         'predicted_ee': y_pred_test
         })

    def clean_data_points(data):
        data = data.assign(waist_ee_cleaned=data['waist_ee'])
        data = data.assign(predicted_ee_cleaned=data['predicted_ee'])
        data.loc[(data['predicted_ee'] < 1), 'predicted_ee_cleaned'] = 1
        return data

    results_df = clean_data_points(results_df)

    # print('Model Evaluation - BlandAltman')
    SE.BlandAltman.bland_altman_paired_plot_tested(results_df, '{}'.format(REGRESSION_RESULTS_FOLDER), 1, log_transformed=True,
                                                   min_count_regularise=False,
                                                   output_filename=join(REGRESSION_RESULTS_FOLDER, 'bland_altman'))

    result_string = '\n'.join(results_descriptions)
    with open(join(REGRESSION_RESULTS_FOLDER, 'result_report.txt'), "w") as text_file:
        text_file.write(result_string)


def evaluate_regression_modal_individual(overall_df, REGRESSION_RESULTS_FOLDER):

    out_col_names = ['participant', 'R2_Overall', 'R2_SBLPA', 'R2_MVPA', 'Pearson_Overall', 'Pearson_Overall_pval',
                     'Pearson_SBLPA', 'Pearson_SBLPA_pval', 'Pearson_MVPA', 'Pearson_MVPA_pval',
                     'accuracy', 'accuracy_ci', 'sensitivity_SBLPA', 'sensitivity_MVPA', 'sensitivity_ci_SBLPA', 'sensitivity_ci_MVPA',
                     'specificity_SBLPA', 'specificity_MVPA', 'specificity_ci_SBLPA', 'specificity_ci_MVPA',
                     'actual_SBLPA', 'predicted_SBLPA', 'actual_MVPA', 'predicted_MVPA']
    output_results = []
    unique_participants = overall_df['subject'].unique()
    for key in tqdm(unique_participants, desc='User level: '):

        results_df = overall_df.loc[overall_df['subject'] == key]

        # regression
        person_corr, pearson_pval = pearsonr(results_df['waist_ee'], results_df['predicted_ee'])
        r2_err = r2_score(results_df['waist_ee'], results_df['predicted_ee'])

        df_sblpa = results_df.loc[results_df['waist_ee'] < 3.0]
        person_corr_sblpa, pearson_pval_sblpa = pearsonr(df_sblpa['waist_ee'], df_sblpa['predicted_ee'])
        r2_sblpa = r2_score(df_sblpa['waist_ee'], df_sblpa['predicted_ee'])

        df_mvpa = results_df.loc[results_df['waist_ee'] >= 3.0]
        person_corr_mvpa, pearson_pval_mvpa = pearsonr(df_mvpa['waist_ee'], df_mvpa['predicted_ee'])
        r2_mvpa = r2_score(df_mvpa['waist_ee'], df_mvpa['predicted_ee'])

        # Classification
        y_test_ai = SE.EnergyTransform.met_to_intensity_sblpa_mvpa(results_df['waist_ee'])
        y_pred_ai = SE.EnergyTransform.met_to_intensity_sblpa_mvpa(results_df['predicted_ee'])

        stats = SE.GeneralStats.evaluation_statistics(confusion_matrix(y_test_ai, y_pred_ai))
        accuracy = stats['accuracy']
        accuracy_ci = stats['accuracy_ci']

        sensitivity_sblpa, sensitivity_mvpa = stats['sensitivity']
        sensitivity_ci_sblpa, sensitivity_ci_mvpa = stats['sensitivity_ci']

        specificity_sblpa, specificity_mvpa = stats['specificity']
        specificity_ci_sblpa, specificity_ci_mvpa = stats['specificity_ci']

        # Calculate PA Time
        _, actual_SBLPA, actual_MVPA = get_time_in(y_test_ai)
        _, predicted_SBLPA, predicted_MVPA = get_time_in(y_pred_ai)

        result_row = [key, r2_err, r2_sblpa, r2_mvpa, person_corr, pearson_pval,
                      person_corr_sblpa, pearson_pval_sblpa, person_corr_mvpa, pearson_pval_mvpa,
                      accuracy, accuracy_ci,
                      sensitivity_sblpa, sensitivity_mvpa, sensitivity_ci_sblpa, sensitivity_ci_mvpa,
                      specificity_sblpa, specificity_mvpa, specificity_ci_sblpa, specificity_ci_mvpa,
                      actual_SBLPA, predicted_SBLPA, actual_MVPA, predicted_MVPA]

        output_results.append(result_row)

    # test completed
    pd.DataFrame(output_results, columns=out_col_names).to_csv(
        join(REGRESSION_RESULTS_FOLDER, 'individual_results.csv'), index=None)


if __name__ == '__main__':

    INPUT_DATA_ROOT = "E:/Data/Accelerometer_Dataset_Rashmika/pre-processed/P2-Processed_Statistical_features/Epoch60_Combined/"
    OUTPUT_FOLDER_ROOT = 'outputs'
    TRAIN_TEST_SUBJECT_PICKLE = '../participant_split/train_test_split.v2.pickle'

    LABEL_REG = 'waist_ee'
    req_cols = ['enmo', LABEL_REG]
    GROUPS = ['test', 'train_test']
    WEEKS = {'test': 'Week 2', 'train_test': 'Week 2'}

    # Test Train Split
    with open(TRAIN_TEST_SUBJECT_PICKLE, 'rb') as handle:
        split_dict = pickle.load(handle)
    split_dict['train_test'] = split_dict['train'][:]

    for user_group in GROUPS:

        print('Evaluating - {}'.format(user_group))

        INPUT_DATA_FOLDER = join(INPUT_DATA_ROOT, WEEKS[user_group])
        OUTPUT_FOLDER_PATH = join(OUTPUT_FOLDER_ROOT, user_group)
        if not exists(OUTPUT_FOLDER_PATH):
            makedirs(join(OUTPUT_FOLDER_PATH, 'overall'))
            makedirs(join(OUTPUT_FOLDER_PATH, 'individual'))

        all_files = [f for f in listdir(INPUT_DATA_FOLDER) if isfile(join(INPUT_DATA_FOLDER, f))]

        counter = 0
        for f in tqdm(all_files, desc=user_group):

            if f.split()[0] not in split_dict[user_group]:
                continue

            # if file.split('_(2016')[0] != 'LSM219':
            #     continue

            dataframe = pd.read_csv(join(INPUT_DATA_FOLDER, f), usecols=req_cols)
            dataframe['subject'] = f.split(' ')[0]

            if counter != 0:
                results = results.append(dataframe, ignore_index=True)
            else:
                results = dataframe

            counter += 1

            # if count > 20:
            #     break

        # Hb-LR Prediction
        results = predict(results)

        # Evaluate overall model
        evaluate_regression_modal_overall(results, join(OUTPUT_FOLDER_PATH, 'overall'))

        # Evaluate individual participant
        evaluate_regression_modal_individual(results, join(OUTPUT_FOLDER_PATH, 'individual'))

    print('Assessment completed.')
