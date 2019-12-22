import pandas as pd
from os import listdir
from os.path import join
import numpy as np
import scipy.stats


def load_df(folder_name):
    results_file = [f for f in listdir(folder_name) if '.csv' in f][0]
    return pd.read_csv(join(folder_name, results_file))


def get_rloa(folder_name, key):

    filename = join(folder_name, 'bland_altman_{}_ba_stats.txt'.format(key.lower()))
    mean_bias, upper_loa, lower_loa = 0, 0, 0
    with open(filename) as f:
        for line in f.readlines():
            mean_bias = line.split(':')[1] if 'mean' in line else mean_bias
            upper_loa = line.split(':')[1] if 'upper' in line else upper_loa
            lower_loa = line.split(':')[1] if 'lower' in line else lower_loa

    return "{:.2f} ({:.2f}-{:.2f})".format(float(mean_bias), float(lower_loa), float(upper_loa))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m, 2), round(m-h, 2), round(m+h, 2)


def get_assessment_scores_cnn_overall(reg_df, class_df):

    # Pearson Corr
    pearson_mean = mean_confidence_interval(reg_df['Pearson_Overall'], confidence=0.95)
    pearson_corr = "{:.2f} ({:.2f}-{:.2f})".format(pearson_mean[0], pearson_mean[1], pearson_mean[2])

    # R2 Error
    r2_mean = mean_confidence_interval(reg_df['R2_Overall'], confidence=0.95)
    r2_error = "{:.2f} ({:.2f}-{:.2f})".format(r2_mean[0], r2_mean[1], r2_mean[2])

    # Spearman Corr (2 Class)
    spearman2_mean = mean_confidence_interval(class_df['spearman_corr_overall_2cls'], confidence=0.95)
    spearman2_corr = "{:.2f} ({:.2f}-{:.2f})".format(spearman2_mean[0], spearman2_mean[1], spearman2_mean[2])

    # Spearman Corr (3 Class)
    spearman3_mean = mean_confidence_interval(class_df['spearman_corr_overall'], confidence=0.95)
    spearman3_corr = "{:.2f} ({:.2f}-{:.2f})".format(spearman3_mean[0], spearman3_mean[1], spearman3_mean[2])

    # Agreement (2 Class)
    aggr2_mean = mean_confidence_interval(class_df['accuracy_2class'] * 100, confidence=0.95)
    aggr2_corr = "{:.2f} ({:.2f}-{:.2f})".format(aggr2_mean[0], aggr2_mean[1], aggr2_mean[2])

    # Agreement (3 Class)
    aggr3_mean = mean_confidence_interval(class_df['accuracy'] * 100, confidence=0.95)
    aggr3_corr = "{:.2f} ({:.2f}-{:.2f})".format(aggr3_mean[0], aggr3_mean[1], aggr3_mean[2])

    return pearson_corr, r2_error, spearman2_corr, spearman3_corr, aggr2_corr, aggr3_corr


def get_assessment_scores_hblr_overall(reg_df):

    # Pearson Corr
    pearson_mean = mean_confidence_interval(reg_df['Pearson_Overall'], confidence=0.95)
    pearson_corr = "{:.2f} ({:.2f}-{:.2f})".format(pearson_mean[0], pearson_mean[1], pearson_mean[2])

    # R2 Error
    r2_mean = mean_confidence_interval(reg_df['R2_Overall'], confidence=0.95)
    r2_error = "{:.2f} ({:.2f}-{:.2f})".format(r2_mean[0], r2_mean[1], r2_mean[2])

    # Spearman Corr (2 Class)
    spearman2_corr = ""

    # Spearman Corr (3 Class)
    spearman3_corr = ""

    # Agreement (2 Class)
    aggr2_mean = mean_confidence_interval(reg_df['accuracy'] * 100, confidence=0.95)
    aggr2_corr = "{:.2f} ({:.2f}-{:.2f})".format(aggr2_mean[0], aggr2_mean[1], aggr2_mean[2])

    # Agreement (3 Class)
    aggr3_corr = ""

    return pearson_corr, r2_error, spearman2_corr, spearman3_corr, aggr2_corr, aggr3_corr


def append_data_results_overall(result_data, class_3000_df, class_6000_df, hblr_df, reg_3000_df, reg_6000_df):

    assert class_3000_df.shape[0] == class_6000_df.shape[0] == hblr_df.shape[0] == reg_3000_df.shape[0] == reg_6000_df.shape[0]

    set_val = '{}'.format('Overall')

    pearson_corr_30, r2_error_30, spearman2_corr_30, spearman3_corr_30, aggr2_corr_30, aggr3_corr_30 = get_assessment_scores_cnn_overall(reg_3000_df, class_3000_df)
    pearson_corr_60, r2_error_60, spearman2_corr_60, spearman3_corr_60, aggr2_corr_60, aggr3_corr_60 = get_assessment_scores_cnn_overall(reg_6000_df, class_6000_df)
    pearson_corr_hblr, r2_error_hblr, spearman2_corr_hblr, spearman3_corr_hblr, aggr2_corr_hblr, aggr3_corr_hblr = get_assessment_scores_hblr_overall(hblr_df)

    result_data.append([set_val, 'Pearson', pearson_corr_30, pearson_corr_60, pearson_corr_hblr])
    result_data.append([set_val, 'R2', r2_error_30, r2_error_60, r2_error_hblr])
    result_data.append([set_val, 'Spearman (SBLPA and MVPA)', spearman2_corr_30, spearman2_corr_60, spearman2_corr_hblr])
    result_data.append([set_val, 'Spearman (SB, LPA and MVPA)', spearman3_corr_30, spearman3_corr_60, spearman3_corr_hblr])
    result_data.append([set_val, 'Agreement (SBLPA and MVPA)', aggr2_corr_30, aggr2_corr_60, aggr2_corr_hblr])
    result_data.append([set_val, 'Agreement (SB, LPA and MVPA)', aggr3_corr_30, aggr3_corr_60, aggr3_corr_hblr])

    return result_data


def append_data_results_pa_cat(result_data, class_3000_df, class_6000_df, hblr_df, reg_3000_df, reg_6000_df, reg_3000_ba_location, reg_6000_ba_location, hblr_ba_location, title):

    assert class_3000_df.shape[0] == class_6000_df.shape[0] == hblr_df.shape[0] == reg_3000_df.shape[0] == reg_6000_df.shape[0]

    set_val = '{}'.format(title)

    r2_error_30, pearson_error_30, rloa_30, sens_30, spec_30 = get_assessment_scores_cnn(reg_3000_df, class_3000_df, reg_3000_ba_location, title)
    r2_error_60, pearson_error_60, rloa_60, sens_60, spec_60 = get_assessment_scores_cnn(reg_6000_df, class_6000_df, reg_6000_ba_location, title)
    r2_error_hb, pearson_error_hb, rloa_hb, sens_hb, spec_hb = get_assessment_scores_hb_lr(hblr_df, hblr_ba_location, title)

    result_data.append([set_val, 'Pearson', pearson_error_30, pearson_error_60, pearson_error_hb])
    result_data.append([set_val, 'R2', r2_error_30, r2_error_60, r2_error_hb])
    result_data.append([set_val, 'RLOA', rloa_30, rloa_60, rloa_hb])
    result_data.append([set_val, 'Sensitivity', sens_30, sens_60, sens_hb])
    result_data.append([set_val, 'Specificity', spec_30, spec_60, spec_hb])

    return result_data


def get_assessment_scores_cnn(reg_df, class_df, reg_ba_loc, pa_cat):

    r2_mean = mean_confidence_interval(reg_df['R2_{}'.format(pa_cat)], confidence=0.95) if pa_cat != 'SBLPA' else ''
    r2_error = "{:.2f} ({:.2f}-{:.2f})".format(r2_mean[0], r2_mean[1], r2_mean[2]) if pa_cat != 'SBLPA' else ''

    # Pearson Corr
    pearson_mean = mean_confidence_interval(reg_df['Pearson_{}'.format(pa_cat)], confidence=0.95) if pa_cat != 'SBLPA' else ''
    pearson_error = "{:.2f} ({:.2f}-{:.2f})".format(pearson_mean[0], pearson_mean[1], pearson_mean[2]) if pa_cat != 'SBLPA' else ''

    rloa = get_rloa(reg_ba_loc, pa_cat) if pa_cat != 'SBLPA' else ''

    sens_mean = mean_confidence_interval(class_df['sensitivity {}'.format(pa_cat)] * 100, confidence=0.95)
    sens = "{:.2f} ({:.2f}-{:.2f})".format(sens_mean[0], sens_mean[1], sens_mean[2])

    spec_mean = mean_confidence_interval(class_df['specificity_{}'.format(pa_cat)] * 100, confidence=0.95)
    spec = "{:.2f} ({:.2f}-{:.2f})".format(spec_mean[0], spec_mean[1], spec_mean[2])

    return r2_error, pearson_error, rloa, sens, spec


def get_assessment_scores_hb_lr(hild_df, reg_ba_loc, pa_cat):

    r2_mean = mean_confidence_interval(hild_df['R2_{}'.format(pa_cat)], confidence=0.95) if pa_cat != 'SB' and pa_cat != 'LPA' else ''
    r2_error = "{:.2f} ({:.2f}-{:.2f})".format(r2_mean[0], r2_mean[1], r2_mean[2]) if pa_cat != 'SB' and pa_cat != 'LPA' else ''

    # Pearson Corr
    pearson_mean = mean_confidence_interval(hild_df['Pearson_{}'.format(pa_cat)], confidence=0.95) if pa_cat != 'SB' and pa_cat != 'LPA' else ''
    pearson_error = "{:.2f} ({:.2f}-{:.2f})".format(pearson_mean[0], pearson_mean[1], pearson_mean[2]) if pa_cat != 'SB' and pa_cat != 'LPA' else ''

    rloa = get_rloa(reg_ba_loc, pa_cat) if pa_cat != 'SBLPA' else ''

    sens_mean = mean_confidence_interval(hild_df['sensitivity_{}'.format(pa_cat)] * 100, confidence=0.95) if pa_cat != 'SB' and pa_cat != 'LPA' else ''
    sens = "{:.2f} ({:.2f}-{:.2f})".format(sens_mean[0], sens_mean[1], sens_mean[2]) if pa_cat != 'SB' and pa_cat != 'LPA' else ''

    spec_mean = mean_confidence_interval(hild_df['specificity_{}'.format(pa_cat)] * 100, confidence=0.95) if pa_cat != 'SB' and pa_cat != 'LPA' else ''
    spec = "{:.2f} ({:.2f}-{:.2f})".format(spec_mean[0], spec_mean[1], spec_mean[2]) if pa_cat != 'SB' and pa_cat != 'LPA' else ''

    return r2_error, pearson_error, rloa, sens, spec


if __name__ == '__main__':

    training_version = '1-12_Dec'
    eval_category = ['train_test', 'test']

    HBLR_ROOT = "E:/Projects/Accelerometer-project_Rashmika/supervised_learning/hb-lr-eval/outputs"
    CNN_ROOT = "E:/Projects/Accelerometer-project_Rashmika/supervised_learning/output/v{}".format(training_version)

    result_col_names = ['Set', 'Item', 'CNN30', 'CNN60', 'HBLR']

    for data_cat in eval_category:

        class_3000_df_main = load_df(join(CNN_ROOT, 'classification', 'window-3000-overlap-1500', 'individual_results', data_cat))
        class_6000_df_main = load_df(join(CNN_ROOT, 'classification', 'window-6000-overlap-3000', 'individual_results', data_cat))
        hblr_df_main = load_df(join(HBLR_ROOT, data_cat, 'individual'))

        reg_3000_df_main = load_df(join(CNN_ROOT, 'regression', 'window-3000-overlap-1500', 'individual_results', data_cat))
        reg_6000_df_main = load_df(join(CNN_ROOT, 'regression', 'window-6000-overlap-3000', 'individual_results', data_cat))

        reg_3000_ba_location_main = join(CNN_ROOT, 'regression', 'window-3000-overlap-1500', 'results', data_cat)
        reg_6000_ba_location_main = join(CNN_ROOT, 'regression', 'window-6000-overlap-3000', 'results', data_cat)
        hild_ba_location_main = join(HBLR_ROOT, data_cat, 'overall')

        res_det = []

        # Set data for main classes
        res_det = append_data_results_pa_cat(res_det, class_3000_df_main, class_6000_df_main, hblr_df_main,
                                           reg_3000_df_main, reg_6000_df_main, reg_3000_ba_location_main,
                                           reg_6000_ba_location_main, hild_ba_location_main, 'SB')
        res_det = append_data_results_pa_cat(res_det, class_3000_df_main, class_6000_df_main, hblr_df_main,
                                           reg_3000_df_main, reg_6000_df_main, reg_3000_ba_location_main,
                                           reg_6000_ba_location_main, hild_ba_location_main, 'LPA')
        res_det = append_data_results_pa_cat(res_det, class_3000_df_main, class_6000_df_main, hblr_df_main,
                                           reg_3000_df_main, reg_6000_df_main, reg_3000_ba_location_main,
                                           reg_6000_ba_location_main, hild_ba_location_main, 'SBLPA')
        res_det = append_data_results_pa_cat(res_det, class_3000_df_main, class_6000_df_main, hblr_df_main,
                                           reg_3000_df_main, reg_6000_df_main, reg_3000_ba_location_main,
                                           reg_6000_ba_location_main, hild_ba_location_main, 'MVPA')

        # Set data for overall
        res_det = append_data_results_overall(res_det, class_3000_df_main, class_6000_df_main, hblr_df_main,
                                              reg_3000_df_main, reg_6000_df_main)

        # Save table 2 output
        pd.DataFrame(res_det, columns=result_col_names).to_csv('Table_2_Agreement_{}_{}.csv'.format(training_version, data_cat), index=None)

    print('Completed')
