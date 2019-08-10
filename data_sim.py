# -*- coding: utf-8 -*

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def calc_risk(data):
    status = data['loan_status']
    if status == 'Fully Paid':
        return 0
    return 1


def dumm_df(df, col_name):
    dumm = pd.get_dummies(df[col_name])
    df = pd.concat([df, dumm], axis=1)
    del df[col_name]
    return df


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
                  '7 years', '8 years', '9 years', '< 1 year',
                  'MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT', 'Not Verified',
                  'Source Verified', 'Verified', 'n', 'y', 'car', 'credit_card',
                  'debt_consolidation', 'educational', 'home_improvement', 'house',
                  'major_purchase', 'medical', 'moving', 'other', 'renewable_energy',
                  'small_business', 'vacation', 'wedding', 'INDIVIDUAL']
    return param_cols


# 我们之前算的过滤过的 csv
csv_path = "./loan_filter.csv"
df = pd.read_csv(csv_path)
df['risk'] = df.apply(calc_risk, axis=1)
df["intercept"] = 1
df = dumm_df_all(df)
size = (df.shape)[0]
# 需要拟合的数据
sim_data = df[0:int(size * 0.6)]
# 对正负判断优化系数的数据
opt_data = df[int(size * 0.6):int(size * 0.8)]
# 最后验证数据
verify_data = df[int(size * 0.8):size]
print("verify_data", verify_data.shape)
# 拟合结果
fit_result = sm.Logit(sim_data['risk'], sim_data[sim_cols()]).fit()
print(fit_result.summary())


class DataPredict(object):

    def __init__(self, gap_value):
        self._gap_value = gap_value

    def calc_correct(self, data):
        predicted = data['predicted']
        actual = data['actual']
        if actual < 1 and predicted < self._gap_value:
            # 我们预测的值 在混淆矩阵的右下方，预测正确
            return 1
        if actual > 0 and predicted > self._gap_value:
            # 我们预测的值 在混淆矩阵的左上方，预测正确
            return 1
        return 0

    def calc_correct_num(self, data):
        predicted = fit_result.predict(data[sim_cols()])
        compare = pd.DataFrame({'predicted': predicted, 'actual': data['risk']})
        ret_data = compare.apply(self.calc_correct, axis=1)
        return sum(ret_data)


start = 0.3
end = 0.7
gap = 0.01
gap_list = []
correct_num_list = []
curr = start

correct_dic = dict()
max_num = 0
max_num_key = start
while curr < end:
    gap_list.append(curr)
    dp = DataPredict(curr)
    correct_num = dp.calc_correct_num(opt_data)
    correct_dic[curr] = correct_num
    # 找到正确率最大点
    if correct_num > max_num:
        max_num = correct_num
        max_num_key = curr

    correct_num_list.append(correct_num)
    curr = curr + gap

opt_df = pd.DataFrame({
    'gap_value': pd.Series(gap_list),
    'correct_num': pd.Series(correct_num_list)
})

sns.relplot(x="gap_value", y="correct_num", data=opt_df)
plt.show()

print("max_num_key:", max_num_key)

# 使用验证集来验证比较

# 如果使用gap = 0.5
dp = DataPredict(0.5)
correct_num = dp.calc_correct_num(verify_data)
print("on verify_data gap = 0.5, correct_num = ", correct_num)
print("正确率：", correct_num * 5 / size)

dp = DataPredict(max_num_key)
correct_num = dp.calc_correct_num(verify_data)
print("on verify_data gap =,", max_num_key, " correct_num = ", correct_num)
print("正确率：", correct_num * 5 / size)
