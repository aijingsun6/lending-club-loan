# -*- coding: utf-8 -*

import pandas as pd

path = "./loan.csv"
target_path = "./loan_trans.csv"
col_set = set({'loan_amnt',
               'term',
               'installment',
               'grade',
               'emp_length',
               'home_ownership',
               'annual_inc',
               'verification_status',
               'loan_status',
               'pymnt_plan',
               'purpose',
               'application_type'})

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


def calc_pay_ratio(data):
    """
    installment * 12 / annual_inc
    :param data:
    :return:
    """
    installment = data['installment']
    annual_inc = data['annual_inc']
    return installment * 12 / annual_inc


def calc_emp_length(data):
    emp_length = data['emp_length']
    if isinstance(emp_length, str) and emp_length > '':
        return emp_length
    return "unknown"


def data_trans():
    df = pd.read_csv(path, low_memory=False, usecols=col_set)
    df = df[df['loan_status'].isin(status_set)]
    df = df[df['annual_inc'] > 0]
    df['risk'] = df.apply(calc_risk, axis=1)
    del df['loan_status']
    df['pay_ratio'] = df.apply(calc_pay_ratio, axis=1)
    del df['installment']
    del df['annual_inc']

    df['emp_length'] = df.apply(calc_emp_length, axis=1)

    df.to_csv(target_path)
    print("save data to {0}".format(target_path))


if __name__ == "__main__":
    data_trans()
