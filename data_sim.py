# -*- coding: utf-8 -*

import pandas as pd

path = "./loan_trans.csv"


def dumm_df(df, col_name):
    dumm = pd.get_dummies(df[col_name])
    return pd.concat([df, dumm], axis=1)


def split_df(df):
    size = (df.shape)[0]
    start = int(size * 0.8)
    mid = int(size * 0.9)
    start_data = df[0:start]
    mid_data = df[start:mid]
    end_data = df[mid:]
    return [start_data, mid_data, end_data]


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
