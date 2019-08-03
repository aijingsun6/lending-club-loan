# -*- coding: utf-8 -*
import pandas as pd
import statsmodels.api as sm

path = "./loan.csv"
col_set = {'loan_amnt', 'term', 'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc',
           'verification_status', 'loan_status', 'pymnt_plan', 'purpose', 'application_type'}

# only filter
# risk
status_set = ['Fully Paid',
              'Charged Off',
              'Late (16-30 days)',
              'Late (31-120 days)',
              'Does not meet the credit policy. Status:Fully Paid',
              'Does not meet the credit policy. Status:Charged Off'
              ]


def calc_risk(data):
    """
    Fully Paid ===> 0
    Does not meet the credit policy. Status:Fully Paid ===> 0
    other ===> 1
    :param data:
    :return:
    """
    status = data['loan_status']
    # print(status)
    if status == 'Fully Paid':
        return 0
    if status == 'Does not meet the credit policy. Status:Fully Paid':
        return 0
    return 1


def calc_emp_length(data):
    emp_length = data['emp_length']
    if isinstance(emp_length, str) and emp_length > '':
        return emp_length
    return "unknown"


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


df = pd.read_csv(path, low_memory=False, usecols=col_set)
df = df[df['loan_status'].isin(status_set)]
df = df[df['annual_inc'] > 0]
df['risk'] = df.apply(calc_risk, axis=1)
df['emp_length'] = df.apply(calc_emp_length, axis=1)
df = dumm_df_all(df)
df["intercept"] = 1
split_data = split_df(df)
sim_data = split_data[0]
test_data = split_data[1]
fit_result = sm.Logit(sim_data['risk'], sim_data[sim_cols()]).fit()
print(fit_result.summary())
predicted = fit_result.predict(test_data[sim_cols()])
compare = pd.DataFrame({'predicted': predicted, 'actual': test_data['risk']})
print(compare)
ret_data = compare.apply(calc_correct, axis=1)
ratio = sum(ret_data) / ret_data.size
print("正确率：", ratio)
