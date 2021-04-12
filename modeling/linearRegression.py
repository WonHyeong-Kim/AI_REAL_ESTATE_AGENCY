#linear regression에서 ols로 summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

plt.rc('font', family='malgun gothic')  # 한글깨짐 방지

df = pd.read_csv("train_park_daycare.csv")
pd.set_option('display.max_row', 500) # 컬럼 다 보기
pd.set_option('display.max_columns', 100)


#model = smf.ols(formula = 'transaction_real_price ~ exclusive_use_area + day_care_babyTeacher_rate +floor', data = df).fit() 
model = smf.ols(formula = 'transaction_real_price ~ gu + year_of_completion + floor', data = df).fit()
table = sm.stats.anova_lm(model, type=2)
print(table)
result = pairwise_tukeyhsd(df, df.gu)
print(result)
result.plot_simultaneous()
plt.show()
'''
ols 가 자동으로 하는 일 :
 최적의 추세선을 긋기위해 잔차 제곱의 합이 최소가 되는 cost을 찾는다
 편미분을 통해 절편기울기를 구한다
print(model.summary())
'''
# 결과 : P>|t|=0.000 는 p-value=0.000 를 의미함. 모델이 유효하다는 뜻
