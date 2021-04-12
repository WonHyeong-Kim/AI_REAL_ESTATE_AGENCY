'''
전통적인 방법의 모델 정확도 예측하기
- randomforest
- xgboost
- knn

모델간의 과적합 그래프로 판별하기
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics._scorer import accuracy_scorer
# from sklearn import metrics
# from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('train_park_daycare.csv')
print(data.head(3))    # [742285 rows x 10 columns]
print(data.info())
'''
 #   Column                     Non-Null Count   Dtype  
---  ------                     --------------   -----  
 0   apartment_id               742285 non-null  int64  
 1   gu                         742285 non-null  int64  
 2   exclusive_use_area         742285 non-null  float64
 3   year_of_completion         742285 non-null  int64  
 4   transaction_year_month     742285 non-null  int64  
 5   transaction_date           742285 non-null  int64  
 6   floor                      742285 non-null  int64  
 7   park_area_sum              742285 non-null  float64
 8   day_care_babyTeacher_rate  742285 non-null  float64
 9   transaction_real_price     742285 non-null  int64 
'''

#data['year_of_completion'] = datetime.today().year - data['year_of_completion']
#print(data['gu'].unique())
#print(data['gu'].unique().counts())
#aa = pd.DataFrame(OneHotEncoder().fit_transform(data['gu'].values[:, np.newaxis]).toarray(), columns = ['용산구', '양천구', '강동구', '관악구', '노원구', '영등포구', '마포구', '서초구', '성동구', '금천구', '도봉구', '동작구', '강서구', '동대문구', '강북구', '서대문구', '광진구', '구로구', '성북구', '강남구', '종로구', '중구', '중랑구', '송파구', '은평구'],\
#                  index=data.index)
#print(aa)
data['gu'] = OneHotEncoder().fit_transform(data['gu'].values[:, np.newaxis]).toarray()
print(data['gu'])
print(data.head(3))    # [742285 rows x 10 columns]
y_data = data["transaction_real_price"]
x_data = data[["exclusive_use_area" ,'day_care_babyTeacher_rate' , 'floor' ,
           'year_of_completion' , 'park_area_sum', 'apartment_id', 'gu', 
           'transaction_year_month' ]]   

#print(x_data)
#print(y_data)

"""train/test"""
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=123)

"""스케일링"""
sc = StandardScaler()
sc.fit(x_train, x_test) 
x_train = sc.transform(x_train)  
x_test = sc.transform(x_test)  


"""Randomforest"""
#rfmodel = RandomForestRegressor(n_estimators=, criterion='mse', random_state = 123)
rfmodel = RandomForestRegressor(n_estimators=5, criterion='mse')
history = rfmodel.fit(x_train,y_train)
y_rfpred = rfmodel.predict(x_test)
rf_r2 = r2_score(y_test, y_rfpred)
print(' Randomforest 설명력: ',rf_r2) 
#print(y_test == y_rfpred)
print(sum(y_test == y_rfpred))
print(len(y_test))
print('모델정확도1:', (sum(y_test == y_rfpred) / len(y_test))*100)


# 데이터 셋 분할 (by cross validation) 및 분류 정확도 평균 계산
cv = model_selection.cross_val_score(rfmodel,x_data,y_data,cv=6)   
print("cross validation 정확도 평균:",cv.mean())
col = pd.DataFrame(x_train,columns=x_data.columns)
print(col.columns)

varDic = {'var':col.columns,'imp': rfmodel.feature_importances_}
imp = pd.DataFrame(varDic)
imp = imp.sort_values(by='imp', ascending=False)[0:17]
print(imp)  

print("Importance barChart")
importances = rfmodel.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfmodel.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
plt.rc('font', family="malgun gothic")
plt.title("특성 중요도")
plt.bar(range(x_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
x_train = pd.DataFrame(x_train)
plt.xticks(np.arange(x_train.shape[1]), tuple(imp["var"]))
plt.xlim([-1, x_train.shape[1]])
plt.show()

#------------------------------------------------------------------------
# 보스턴집값 예측 거기 sample넣고 돌린코드랑 결과
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection._split import train_test_split
from matplotlib import style

plt.rc('font', family='malgun gothic')  # 한글깨짐 방지

data = pd.read_csv("train_park_daycare_sample.csv")
pd.set_option('display.max_row', 500) # 컬럼 다 보기
pd.set_option('display.max_columns', 100)


dfx = pd.DataFrame(data, columns=['exclusive_use_area' ,'day_care_babyTeacher_rate' , 'floor' ,
           'year_of_completion' , 'park_area_sum' ])
print(dfx.head(3), dfx.shape) # (506, 13)
dfy = pd.DataFrame(data.transaction_real_price, columns=['transaction_real_price'])
print(dfy.head(3))

df = pd.concat([dfx,dfy], axis=1) # column 합치기, axis=1 : 행기준으로 설정 (기본은0 : 열기준)
#print(df.head(3))

print(df.corr()) # 상관관계 확인

# 시각화
cols = ['exclusive_use_area' ,'day_care_babyTeacher_rate' , 'floor' ,
           'year_of_completion' , 'park_area_sum']
sns.pairplot(df[cols])
plt.show()

x = df[['exclusive_use_area']].values
y = df[['transaction_real_price']].values
print(x[:2])
print(y[:2])

print('-------------실습 1- DecisionTreeRegressor------------') 
model1 = DecisionTreeRegressor(max_depth=3).fit(x,y) #모델생성
print('predict : ', model1.predict(x)[:5])
print('real : ',y[:5])
y_pred = model1.predict(x)
r2_1 = r2_score(y, y_pred)
print('결정계수(R2,설명력) : ', r2_1) # 0.6993

print('-------------실습 2- RandomForestRegressor------------') 
model2 = RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=1).fit(x,y) #모델생성
# criterion='mse' : 평균제곱오차
print('predict : ', model2.predict(x)[:5])
print('real : ',y[:5])
y_pred = model2.predict(x)
r2_2 = r2_score(y, y_pred)
print('결정계수(R2,설명력) : ', r2_2) # 0.909

print('-------------- 학습 검정 자료로 분리 ---------------')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
model2.fit(x_train,y_train)

r2_train = r2_score(y_train, model2.predict(x_train))
print('train 설명력 : ', r2_train)
# train 설명력 :  0.9101806653534238

r2_test = r2_score(y_test, model2.predict(x_test))
print('train 설명력 : ', r2_test)
# train 설명력 :  0.8790027615733326

# 시각화
style.use('seaborn-talk')
plt.scatter(x, y, c='lightgrey', label='train data')
plt.scatter(x_test, model2.predict(x_test), c='r',label='predict data, $R^2=%.2f$'%r2_test)
plt.legend()
plt.show()

# 새값으로 예측
print(x_test[:3])
x_new = [[50.11], [26.53], [1.76]]
print('예상 집값 : ',model2.predict(x_new))
# 예상 집값 :  [20431.91671443 12908.9935312  10760.21510714]
# -------------실습 1- DecisionTreeRegressor------------
# 결정계수(R2,설명력) :  0.7320665340205426
# -------------실습 2- RandomForestRegressor------------
# 결정계수(R2,설명력) :  0.9148230003724018
# -------------- 학습 검정 자료로 분리 ---------------
# train 설명력 :  0.9101806653534238
