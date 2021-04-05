import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
baby = pd.read_csv('../dataset/day_care_center.csv')

print(baby.head(4))
#서울 데이터만 추출
baby = baby.loc[baby['city'] == '서울특별시']

#결측치 평균으로 채우기
baby['teacher_num'] = baby['teacher_num'].fillna(baby['teacher_num'].mean())
print('결측치 확인: ', baby['teacher_num'].isnull().sum())

#구별로 매핑
baby['gu_map'] = baby['gu'].map({'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9, '도봉구': 10, '동작구': 11, '강서구': 12, '동대문구': 13, '강북구': 14, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18, '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24})


baby_type = baby['day_care_type'].unique()

print(baby_type)

#기관 별로 가중치 입력
print('결측치 확인', baby['day_care_type'].isnull().sum())

baby['baby/teacher point'] = baby['day_care_type'].map({'국공립':0.4, '사회복지법인': 0.3, '직장':0.3, '법인·단체':0.3, '민간':0.2,'협동':0.1, '가정':0.1})
print('1결측치 확인', baby['baby/teacher point'].isnull().sum())

print(baby.head(5))

# 가중치 곱한 값 입력
baby['day_care_baby_num'] = round(baby['day_care_baby_num'] * baby['baby/teacher point'], 3)
baby['teacher_num'] = round(baby['teacher_num'] * baby['baby/teacher point'], 3)

print(baby.head(5))

#구별 아이 수와 선생 수 합하기
baby_sum = []
teacher_sum = []
gu = []

for i in baby['gu'].unique():
    gu.append(i)
    x = baby.loc[baby['gu'] == i]['day_care_baby_num'].sum()
    y = baby.loc[baby['gu'] == i]['teacher_num'].sum()
    baby_sum.append(x)
    teacher_sum.append(y)

set1 = pd.DataFrame()
set1['gu'] = gu
set1['baby_sum'] = baby_sum
set1['teacher_sum'] = teacher_sum

print(set1.head(4))

set1['baby/teacher point'] = round(set1['baby_sum'] / set1['teacher_sum'],2)

print(set1)

set1.to_csv('../dataset/baby_center.csv', header = True, index = False)