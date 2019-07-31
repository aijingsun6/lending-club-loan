# -*- coding: utf-8 -*
import pandas as pd


def display_csv_info(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    print(df.head())
    print(df.describe())
    print("distinct(term)", df['term'].unique())
    print("distinct(emp_length)", df['emp_length'].unique())
    print("distinct(verification_status)", df['verification_status'].unique())
    print("distinct(purpose)", df['purpose'].unique())
    print("distinct(application_type)", df['application_type'].unique())
    print("distinct(home_ownership)", df['home_ownership'].unique())
    print("distinct(grade)", df['grade'].unique())
