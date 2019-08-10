import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# 源数据
path = "./loan.csv"
col_set = {'loan_amnt',
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
           'application_type'}

# 我们关注的 status
status_set = ['Fully Paid',
              'Charged Off',
              'Late (16-30 days)',
              'Late (31-120 days)'
              ]

# 不含 nan 的雇佣年限
emp_length_set = ['10+ years',
                  '< 1 year',
                  '1 year',
                  '3 years',
                  '8 years',
                  '9 years',
                  '4 years', '5 years', '6 years', '2 years', '7 years']


def display_col_unique(df):
    for col in col_set:
        print(col, df[col].unique())


def calc_col_conf_int(col, val):
    mean, sigma = np.mean(col), np.std(col)
    conf_int = stats.norm.interval(val, loc=mean, scale=sigma)
    return conf_int


def distplot(df):
    sns.distplot(df["loan_amnt"], bins=100, rug=True)
    plt.show()
    sns.distplot(df["installment"], bins=100, rug=True)
    plt.show()
    sns.distplot(df["annual_inc"], bins=100, rug=True)
    plt.show()


df = pd.read_csv(path, low_memory=False, usecols=col_set)
print("========== data columns unique info before filter")
display_col_unique(df)
print("shape before ", df.shape)
# 对数据进行一次过滤
df = df[df['loan_status'].isin(status_set)]
df = df[df['emp_length'].isin(emp_length_set)]
df = df[df['annual_inc'] > 0]
print("========== data columns unique info after filter")
display_col_unique(df)
print("shape after ", df.shape)

# 做出一些数值类型的分布图
sns.set(color_codes=True)
print("before filter conf int...")
# distplot(df)

loan_amnt_min, loan_amnt_max = calc_col_conf_int(df["loan_amnt"], 0.5)
print("loan_amnt conf int:", loan_amnt_min, loan_amnt_max)

installment_min, installment_max = calc_col_conf_int(df["installment"], 0.68)
print("installment conf int:", installment_min, installment_max)

annual_inc_min, annual_inc_max = calc_col_conf_int(df["annual_inc"], 0.68)
print("annual_inc conf int:", annual_inc_min, annual_inc_max)

df = df[df['loan_amnt'] > loan_amnt_min]
df = df[df['loan_amnt'] < loan_amnt_max]

df = df[df['installment'] > installment_min]
df = df[df['installment'] < installment_max]

df = df[df['annual_inc'] > annual_inc_min]
df = df[df['annual_inc'] < annual_inc_max]
print("shape after calc conf int ", df.shape)
# distplot(df)
df.to_csv("./loan_filter.csv")
