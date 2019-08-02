# lending-club-loan
https://www.kaggle.com/wendykan/lending-club-loan-data


### 过程
```
1.安装 venv
python3 -m venv venv

2.进入 venv 虚拟环境
source venv/bin/active

3.安装依赖库
pip install -r requirements.txt

4.数据操作

4.1 将loan.csv下载放到该目录
4.2 对数据进行预处理得到 loan_trans.csv
python data_trans.py
save data to ./loan_trans.csv


4.3 对数据进行拟合
python data_sim.py 

                           Logit Regression Results                           
==============================================================================
Dep. Variable:                   risk   No. Observations:               242697
Model:                          Logit   Df Residuals:                   242653
Method:                           MLE   Df Model:                           43
Date:                Fri, 02 Aug 2019   Pseudo R-squ.:                 0.07668
Time:                        23:26:43   Log-Likelihood:            -1.1671e+05
converged:                      False   LL-Null:                   -1.2640e+05
Covariance Type:            nonrobust   LLR p-value:                     0.000
...


        predicted  actual
242697   0.325262       0
242698   0.286234       0
242699   0.323611       0
242700   0.455596       0
242701   0.465061       1
...           ...     ...
269659   0.254642       1
269660   0.086538       0
269661   0.321354       0
269662   0.096882       0
269663   0.468197       1

[26967 rows x 2 columns]
正确率： 0.715874958282345


```