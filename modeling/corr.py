'''
상관분석 : 두 변수 간에 상관관계의 강도를 분석
이론적 타당성(독립성) 확인. 독립변수 대상 변수들은 서로간에 독립적이어야함
독립변수 대상 변수들은 다중공산성이 발생할 수 있는데 이를 확인


apartment_id 범주
gu 범주
exclusive_use_area 수치
year_of_completion 범주
transaction_year_month 범주
transaction_date 범주
floor 범주
park_area_sum 수치
day_care_babyTeacher_rate 수치
transaction_real_price 수치
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train_add_cctv.csv")
pd.set_option('display.max_row', 500) # 컬럼 다 보기
pd.set_option('display.max_columns', 100)
print(df.head(3))
print(df.describe())
df['number_of_cctv'] = df['number of cctv']
print(df.columns.unique())
print('==================표준편차 뽑아보기===================')
print(np.std(df.transaction_real_price)) # 33868.30

print('==================공분산 보기===================')
print(np.cov(df.apartment_id, df.transaction_real_price))           # -1.97
print(np.cov(df.gu, df.transaction_real_price))                     # 2.33
print(np.cov(df.year_of_completion, df.transaction_real_price))     # -1.54
print(np.cov(df.transaction_year_month, df.transaction_real_price)) # 1.13
print(np.cov(df.transaction_date, df.transaction_real_price))       # 5.59
print(np.cov(df.floor, df.transaction_real_price))                  # 2.85
print(np.cov(df.exclusive_use_area, df.transaction_real_price))     # 6.49
print(np.cov(df.park_area_sum, df.transaction_real_price))          # -1.80
print(np.cov(df.day_care_babyTeacher_rate, df.transaction_real_price)) # 4.94
print(df.cov()) # pandas 이용

print('==================상관계수 보기===================')
print(np.corrcoef(df.transaction_real_price, df.exclusive_use_area))        # 0.6687
print(np.corrcoef(df.transaction_real_price, df.park_area_sum))             # -0.15489336
print(np.corrcoef(df.transaction_real_price, df.day_care_babyTeacher_rate)) # 0.236
# pandas 이용. 괄호안에 method='pearson'피어슨계수가 default
# print('pearson : ',df.corr()) # 변수가 동간, 정규성따름
#print('spearman : ', df.corr(method='spearman')) # 서열척도임. 정규성X
#print('kendall : ',df.corr(method='kendall')) #

plt.plot(df.transaction_real_price, df.exclusive_use_area, 'o')
plt.show()
plt.plot(df.transaction_real_price, df.park_area_sum, 'o')
plt.show()
plt.plot(df.transaction_real_price, df.day_care_babyTeacher_rate, 'o')
plt.show()

#co_re = df.corr(df.exclusive_use_area, df.transaction_real_price)
#print(co_re['transaction_real_price'].sort_values(ascending=False))


print('=====================시각화==================')
df.plot(kind='scatter', x='exclusive_use_area', y='transaction_real_price')
plt.show()
df.plot(kind='scatter', x='day_care_babyTeacher_rate', y='transaction_real_price')
plt.show()
#---------------------------------------------------------------------------------------
#feature label 상관분석
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl 
mpl.rcParams['agg.path.chunksize'] = 10000
pd.set_option('display.max_columns', 500) #모든 열을 볼 수 있다


# df = pd.read_csv('train_park_daycare.csv')

print(df.head(3)) # 데이터 확인

#등간척도 비율척도 변수만 추출
df1 = df[['exclusive_use_area','park_area_sum','day_care_babyTeacher_rate', 'number_of_cctv', 'transaction_real_price' ]] #피어슨 상관계수 쓰려고 


cor1 = df1.corr() #안써주면 defalut pearson

print(cor1)

#시각화
df1_heatmap = sns.heatmap(cor1, cbar=True, annot=True, fmt='.3f', square=True, cmap='Oranges')
plt.show()

print(df.corr())