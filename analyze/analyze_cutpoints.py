import math

def get_freedson_vm3_combination_11_energy_expenditure(vm_cpm, vm_60, cpm):

    # https://actigraph.desk.com/customer/en/portal/articles/2515835-what-is-the-difference-among-the-energy-expenditure-algorithms-
    if vm_cpm > 2453:
        # Validation and comparison of ActiGraph activity monitors by Jeffer E. Sasaki
        met_value = (0.000863 * vm_60) + 0.668876
    else:
        # http://www.theactigraph.com/research-database/kcal-estimates-from-activity-counts-using-the-potential-energy-method/
        # Step 1: calculate the energy expenditure in Kcals per min
        Kcals_min = cpm * 0.0000191 * 80 * 9.81

        # Step 2: convert Energy Expenditure from Kcal/min to kJ/min
        KJ_min = Kcals_min * 4.184

        # Step 3: assuming that you use 80 kg as body mass, divide value by ((3.5/1000)*80*20.9)
        met_value = KJ_min / ((3.5 / 1000) * 80 * 20.9)

    return met_value


def calc_crouter_ee(cv, waist_count_p10):

    if waist_count_p10 <= 8:
        met_value = 1
    elif cv <= 10:
        met_value = 2.294275 * (math.exp(0.00084679 * waist_count_p10))
    else:
        met_value = 0.749395 + (0.716431 * (math.log(waist_count_p10))) \
                    - (0.179874 * ((math.log(waist_count_p10)) ** 2)) \
                    + (0.033173 * ((math.log(waist_count_p10)) ** 3))

    if met_value > 10:
        print('HIGH MET', met_value)

    return met_value


if __name__ == '__main__':

    # met = get_freedson_vm3_combination_11_energy_expenditure(743, 927, 400)
    # print(met)

    # import pandas as pd
    #
    # df = pd.read_csv('C:/Users/pc/Desktop/Dinithi/LSM219 Wrist (2016-11-02)RAW.csv', skiprows=10)
    # print(df.head(5))

    met = calc_crouter_ee(8, 2000)
    print(met)

    met = calc_crouter_ee(23, 2000)
    print(met)
