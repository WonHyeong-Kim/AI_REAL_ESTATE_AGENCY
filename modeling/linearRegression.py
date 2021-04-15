#linear regression에서 ols로 summary

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import scipy.stats

plt.rc('font', family='malgun gothic')  # 한글깨짐 방지
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("train_add_cctv.csv")
#df = pd.read_csv("train_park_daycare.csv")
pd.set_option('display.max_row', 500) # 컬럼 다 보기
pd.set_option('display.max_columns', 100)

df = df.sample(n=1000, random_state=123)
print(df.head(3))
print(df.columns.unique())
# Index(['apartment_id', 'gu', 'exclusive_use_area', 'year_of_completion',
#        'transaction_year_month', 'transaction_date', 'floor', 'park_area_sum',
#        'day_care_babyTeacher_rate', 'transaction_real_price',
#        'number of cctv'], dtype='object')
df['number_of_cctv'] = df['number of cctv']
print('상관계수 r : \n', df.loc[:, ['transaction_real_price', 'number_of_cctv']].corr())
# 0.318563
 
model = smf.ols(formula = 'transaction_real_price ~ gu + exclusive_use_area + year_of_completion + transaction_year_month + floor + day_care_babyTeacher_rate + number_of_cctv',\
                data = df).fit()
#model.save('linearModel.hdf5')
print(model.summary()) # R-squared : 0.122, p : 1.47e-42
print(model.params)
print('pvalues :', model.pvalues)
print('rsquared :', model.rsquared)
print(df.info())
print(df.describe())
plt.subplot(2,4,1)
plt.boxplot(df.gu)
plt.subplot(2,4,2)
plt.boxplot(df.exclusive_use_area)
plt.subplot(2,4,3)
plt.boxplot(df.year_of_completion)
plt.subplot(2,4,4)
plt.boxplot(df.transaction_year_month)
plt.subplot(2,4,5)
plt.boxplot(df.floor)
plt.subplot(2,4,6)
plt.boxplot(df.day_care_babyTeacher_rate)
plt.subplot(2,4,7)
plt.boxplot(df.number_of_cctv)
plt.subplot(2,4,8)
plt.boxplot(df.park_area_sum)
plt.show()

# 잔차항
print('잔차항')
fitted = model.predict(df)     # 예측값
print(fitted)
'''
0      20.523974
1      12.337855
2      12.307671
'''
residual = df['transaction_real_price'] - fitted # 잔차
print(residual)
scipy.stats.shapiro(residual)
print('선형성')

# 선형성 - 예측값과 잔차가 비슷하게 유지
sns.regplot(fitted, residual, lowess = True, line_kws = {'color':'red'}, scatter_kws={'color': 'orange'})
fig = plt.gcf()         # 이미지 저장 선언
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='grey')
# plt.show() 
plt.xlabel('fitted')
plt.ylabel('residual')
plt.title('선형성')
fig.savefig('선형성.png') # 이미지 저장
plt.close(fig)
print('******************************************************************************')

#정규성- 잔차가 정규분포를 따르는 지 확인
sr = stats.zscore(residual)
fig = plt.gcf()         # 이미지 저장 선언
(x, y), _ = stats.probplot(sr)
sns.scatterplot(x, y, color='orange')
plt.plot([-3, 3], [-3, 3], '--', color="grey")
print('residual test :', stats.shapiro(residual))
# residual test : ShapiroResult(statistic=0.7612664699554443, pvalue=0.0)
# pvalue=1.60833789344866e-21 < 0.05 => 정규성을 만족하지못함.
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('정규성')
plt.show() 
fig.savefig('정규성.png') # 이미지 저장
plt.close(fig)

# 독립성 - 잔차가 자기상관(인접 관측치의 오차가 상관되어 있음)이 있는지 확인
# Durbin-Watson:                   1.631=> 잔차항이 독립성을 만족하는 지 확인. 2에 가까우면 자기상관이 없다.(서로 독립- 잔차끼리 상관관계가 없다)
# 0에 가까우면 양의 상관, 4에 가까우면 음의 상관.
print('******************************************************************************')

# 등분산성 - 잔차의 분산이 일정한지 확인
fig = plt.gcf()         # 이미지 저장 선언
sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess = True, line_kws = {'color':'red'}, scatter_kws={'color': 'orange'})
plt.xlabel('Fitted values')
plt.ylabel('√|Standardized residuals|')
plt.title('등분산성')
plt.show()
fig.savefig('등분산성.png') # 이미지 저장
plt.close(fig)
# 추세선이 수평선을 그리지않으므로 등분산성을 만족하지 못한다.
print('******************************************************************************')
print('다중공선성')
print(df.values)
# 다중공선성 - 독립변수들 간에 강한 상관관계 확인
# VIF(Variance Inflation Factors - 분산 팽창 요인) 값이 10을 넘으면 다중공선성이 발생하는 변수라고 할 수 있다.
from statsmodels.stats.outliers_influence import variance_inflation_factor
# DataFrame으로 보기
vif_df = pd.DataFrame()
vif_df['vid_value'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
print(vif_df)
#     vid_value
# 0    1.046389
# 1    0.000000
# 2    4.661656
# 3    1.205255
# 4    1.030238
# 5    1.007729
# 6    1.071141
# 7    0.000000
# 8    0.000000
# 9    5.086743
# 10   0.000000
# 11   0.000000
# Index(['apartment_id', 'gu', 'exclusive_use_area', 'year_of_completion',
#        'transaction_year_month', 'transaction_date', 'floor', 'park_area_sum',
#        'day_care_babyTeacher_rate', 'transaction_real_price',
#        'number of cctv']
print(df.values)
print(df.shape)
print('******************************************************************************')
table = sm.stats.anova_lm(model, type=2)
print(table)
fig = plt.gcf()         # 이미지 저장 선언

# gu 사후 검정
result = pairwise_tukeyhsd(df.transaction_real_price, df.gu)
fig = result.plot_simultaneous(xlabel='실거래가', ylabel='구')
plt.show()
fig.savefig('PostHoc_gu.png') # 이미지 저장
plt.close(fig)

# year_of_completion 사후 검정
result2 = pairwise_tukeyhsd(df.transaction_real_price, df.year_of_completion)
fig = result2.plot_simultaneous(xlabel='실거래가', ylabel='설립일자')
plt.show()
fig.savefig('PostHoc_completion.png') # 이미지 저장
plt.close(fig)

# floor 사후 검정
result3 = pairwise_tukeyhsd(df.transaction_real_price, df.floor)
fig = result3.plot_simultaneous(xlabel='실거래가', ylabel='층')
plt.show()
fig.savefig('PostHoc_floor.png') # 이미지 저장
plt.close(fig)
print('******************************************************************************')
# exclusive_use_area
model = smf.ols(formula='transaction_real_price ~ exclusive_use_area', data=df).fit()
print(model.summary())
model = stats.linregress(df.exclusive_use_area, df.transaction_real_price)
fig = plt.gcf()         # 이미지 저장 선언
plt.plot(df.exclusive_use_area, df.transaction_real_price, 'o', color='orange', label='data', markersize=10)
plt.plot(df.exclusive_use_area, model.slope * df.exclusive_use_area + model.intercept, 'r', label='추세선')
# sns.regplot(df.exclusive_use_area, df.transaction_real_price, scatter_kws = {'color':'r'})
# pr = LinearRegression()
# polyf = PolynomialFeatures(degree=2) # 특징 행렬 생성
# x_quad = polyf.fit_transform(df.exclusive_use_area)
# pr.fit(x_quad, df.transaction_real_price)
# y_quad_fit = pr.predict(polyf.fit_transform(df.exclusive_use_area))
# plt.plot(df.exclusive_use_area, y_quad_fit, label='quadratic fit', linestyle='-', c='blue')
plt.xlabel('전용면적')
plt.ylabel('실거래가')
plt.title('전용면적 x 실거래가')
plt.legend()
plt.show()
fig.savefig('linear_area.png') # 이미지 저장
plt.close(fig)
print('******************************************************************************')
# park_area_sum
model = smf.ols(formula='transaction_real_price ~ park_area_sum', data=df).fit()
print(model.summary())
model = stats.linregress(df.park_area_sum, df.transaction_real_price)
fig = plt.gcf()         # 이미지 저장 선언
plt.plot(df.park_area_sum, df.transaction_real_price, 'o', color='orange', label='data', markersize=10)
plt.plot(df.park_area_sum, model.slope * df.park_area_sum + model.intercept, 'r', label='추세선')
# sns.regplot(df.exclusive_use_area, df.transaction_real_price, scatter_kws = {'color':'r'})
# pr = LinearRegression()
# polyf = PolynomialFeatures(degree=2) # 특징 행렬 생성
# x_quad = polyf.fit_transform(df.exclusive_use_area)
# pr.fit(x_quad, df.transaction_real_price)
# y_quad_fit = pr.predict(polyf.fit_transform(df.exclusive_use_area))
# plt.plot(df.exclusive_use_area, y_quad_fit, label='quadratic fit', linestyle='-', c='blue')
plt.xlabel('공원 면적')
plt.ylabel('실거래가')
plt.title('구당 공원 면적 x 실거래가')
plt.legend()
plt.show()
fig.savefig('linear_park.png') # 이미지 저장
plt.close(fig)

print('******************************************************************************')
# day_care_babyTeacher_rate
model = smf.ols(formula='transaction_real_price ~ day_care_babyTeacher_rate', data=df).fit()
print(model.summary())
model = stats.linregress(df.day_care_babyTeacher_rate, df.transaction_real_price)
fig = plt.gcf()         # 이미지 저장 선언
plt.plot(df.day_care_babyTeacher_rate, df.transaction_real_price, 'o', color='orange', label='data', markersize=10)
plt.plot(df.day_care_babyTeacher_rate, model.slope * df.day_care_babyTeacher_rate + model.intercept, 'r', label='추세선')
# sns.regplot(df.exclusive_use_area, df.transaction_real_price, scatter_kws = {'color':'r'})
# pr = LinearRegression()
# polyf = PolynomialFeatures(degree=2) # 특징 행렬 생성
# x_quad = polyf.fit_transform(df.exclusive_use_area)
# pr.fit(x_quad, df.transaction_real_price)
# y_quad_fit = pr.predict(polyf.fit_transform(df.exclusive_use_area))
# plt.plot(df.exclusive_use_area, y_quad_fit, label='quadratic fit', linestyle='-', c='blue')
plt.xlabel('교육지수')
plt.ylabel('실거래가')
plt.title('교육지수 x 실거래가')
plt.legend()
plt.show()
fig.savefig('linear_teacher.png') # 이미지 저장
plt.close(fig)

print('******************************************************************************')

'''
ols 가 자동으로 하는 일 :
 최적의 추세선을 긋기위해 잔차 제곱의 합이 최소가 되는 cost을 찾는다
 편미분을 통해 절편기울기를 구한다
print(model.summary())
'''
# 결과 : P>|t|=0.000 는 p-value=0.000 를 의미함. 모델이 유효하다는 뜻
