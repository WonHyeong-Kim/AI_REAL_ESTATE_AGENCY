# -*- coding: utf-8 -*-
"""dahye_ols.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DKll26tlzBdD7YCKXr1CNrbgOPjhxUge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing._data import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler
from tensorflow.keras.models import load_model
import pickle


plt.rc('font', family='malgun gothic')  # 한글깨짐 방지
pd.set_option('display.max_row', 500) # 컬럼 다 보기
pd.set_option('display.max_columns', 100)


df = pd.read_csv("https://raw.githubusercontent.com/WonHyeong-Kim/AI_REAL_ESTATE_AGENCY/main/preprocessing/dataset/train_add_kremap.csv")
#print(df.describe().loc["std"]) #std 값 알아보기 너무 숫자의 범위가 다르다. 



dfx = pd.DataFrame(df, columns = ['gu','exclusive_use_area','year_of_completion','transaction_year_month','floor','park_area_sum','day_care_babyTeacher_rate','cctv_num','k_remap'])
#print(dfx) #transaction_id, apartment_id, transaction_real_price제외
scaler = StandardScaler() #scale 처리를 통해 다중공선성 warning은 안뜸, 그러나 전과 같은 편.
dfx = scaler.fit_transform(dfx) #ndarray의 type으로 바뀜
column_names = ['gu','exclusive_use_area','year_of_completion','transaction_year_month','floor','park_area_sum','day_care_babyTeacher_rate','cctv_num','k_remap']
dfx = pd.DataFrame(dfx, columns = column_names) #그래서 다시 DataFrame type으로 바꿔줌
#print(dfx)

dfy =  pd.DataFrame(df, columns = ['transaction_real_price'])
dfy = scaler.fit_transform(dfy)
dfy = pd.DataFrame(dfy, columns=['transaction_real_price'])
#print(dfy)

df = pd.concat([dfx,dfy], axis=1) #열을 기준으로 x, y 합친다 새로운 데이터프레임 형성
print(df)

model = smf.ols(formula = 'transaction_real_price ~ + gu + exclusive_use_area + year_of_completion + transaction_year_month + floor + park_area_sum + day_care_babyTeacher_rate + cctv_num + k_remap', \
                data = df).fit()

filename = 'ols.h5'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename,'rb'))


print('summary : ',model.summary()) #조정된 결정계수 : 0.544 (54%)
print('p-value값 : \n', model.pvalues)
print('rsqaured값 : ', model.rsquared)

print(loaded_model.rsquared)