# -*- coding: utf-8 -*

import pandas as pd
import statsmodels.api as sm


def dumm_df(df, col_name):
    dumm = pd.get_dummies(df[col_name])
    df = pd.concat([df, dumm], axis=1)
    del df[col_name]
    return df


def split_df(df):
    size = (df.shape)[0]
    start = int(size * 0.9)
    sim_data = df[0:start]
    test_data = df[start:]
    return [sim_data, test_data]


def dumm_df_all(df):
    df = dumm_df(df, 'term')
    df = dumm_df(df, 'grade')
    df = dumm_df(df, 'emp_length')
    df = dumm_df(df, 'home_ownership')
    df = dumm_df(df, 'verification_status')
    df = dumm_df(df, 'pymnt_plan')
    df = dumm_df(df, 'purpose')
    df = dumm_df(df, 'application_type')
    return df


def sim_cols():
    param_cols = ['intercept', 'loan_amnt', 'installment', 'annual_inc',
                  ' 36 months', ' 60 months', 'A', 'B', 'C', 'D', 'E', 'F', 'G', '1 year',
                  '10+ years', '2 years', '3 years', '4 years', '5 years', '6 years',
                  '7 years', '8 years', '9 years', '< 1 year', 'unknown', 'ANY',
                  'MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT', 'Not Verified',
                  'Source Verified', 'Verified', 'n', 'y', 'car', 'credit_card',
                  'debt_consolidation', 'educational', 'home_improvement', 'house',
                  'major_purchase', 'medical', 'moving', 'other', 'renewable_energy',
                  'small_business', 'vacation', 'wedding', 'INDIVIDUAL', 'JOINT']
    return param_cols


def calc_correct(data):
    predicted = data['predicted']
    actual = data['actual']
    if abs(predicted - actual) < 0.5:
        return 1
    return 0


def print_sim_result():
    ds = DataSim()
    ds.sim_data()
    results = ds.get_fit_result()
    print(results.summary())
    ds.show_predict_result()


class DataSim(object):
    _sim_data = None
    _test_data = None
    _fit_result = None

    def get_fit_result(self):
        return self._fit_result

    def sim_data(self):
        df = pd.read_csv("./loan_trans.csv")
        df = dumm_df_all(df)
        df["intercept"] = 1
        split_data = split_df(df)
        self._sim_data = split_data[0]
        self._test_data = split_data[1]
        self._fit_result = sm.Logit(self._sim_data['risk'], self._sim_data[sim_cols()]).fit()

    def show_predict_result(self):
        predicted = self._fit_result.predict(self._test_data[sim_cols()])
        compare = pd.DataFrame({'predicted': predicted, 'actual': self._test_data['risk']})
        print(compare)
        ret_data = compare.apply(calc_correct, axis=1)
        ratio = sum(ret_data) / ret_data.size
        print("正确率：", ratio)


if __name__ == "__main__":
    print_sim_result()
