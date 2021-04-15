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

print(baby_type) # ['국공립' '직장' '가정' '민간' '법인·단체' '협동' '사회복지법인']

#기관 별로 가중치 입력
print('결측치 확인', baby['day_care_type'].isnull().sum()) # 0

#baby['baby/teacher point'] = baby['day_care_type'].map({'국공립':1, '사회복지법인': 1, '직장':1, '법인·단체':1, '민간':1,'협동':1, '가정':1})
baby['baby/teacher point'] = baby['day_care_type'].map({'국공립':0.4, '사회복지법인': 0.3, '직장':0.3, '법인·단체':0.3, '민간':0.2,'협동':0.1, '가정':0.1})
print('1결측치 확인', baby['baby/teacher point'].isnull().sum()) # 0

print(baby.head(5))

# 가중치 곱한 값 입력
baby['day_care_baby_num'] = round(baby['day_care_baby_num'] * baby['baby/teacher point'], 3) # 정원 수
baby['teacher_num'] = round(baby['teacher_num'] * baby['baby/teacher point'], 3)             # 보육교직원 수

print(baby.head(5))
#     city    gu day_care_name day_care_type  day_care_baby_num  teacher_num  \
# 0  서울특별시  서대문구        가람어린이집           국공립               28.0          4.4   
# 1  서울특별시  서대문구      가좌제일어린이집           국공립               35.2          8.8   
# 2  서울특별시  서대문구       경찰청어린이집            직장               27.0          6.9   
# 3  서울특별시  서대문구      고운햇살어린이집            가정                1.8          0.5   
# 4  서울특별시  서대문구        고은어린이집           국공립               37.2          8.0   

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
#      gu  baby_sum  teacher_sum
# 0  서대문구    1999.9        404.7
# 1   도봉구    2313.7        445.5
# 2   송파구    4074.0        819.8
# 3   성동구    2482.8        738.8

set1['baby/teacher point'] = round(set1['baby_sum'] / set1['teacher_sum'],2)

print(set1)

set1.to_csv('../dataset/baby_center.csv', header = True, index = False)
#       gu  baby_sum  teacher_sum  baby/teacher point
# 0   서대문구    1999.9      404.700                4.94
# 1    도봉구    2313.7      445.500                5.19
# 2    송파구    4074.0      819.800                4.97
# 3    성동구    2482.8      738.800                3.36
# 4    은평구    3239.9      595.800                5.44
# 5    서초구    3121.3      626.600                4.98
# 6    종로구    1634.0      341.600                4.78
# 7    양천구    3099.3      596.100                5.20
# 8    동작구    2541.2      460.800                5.51
# 9    강서구    8107.0     1501.000                5.40
# 10    중구    1408.9      294.200                4.79
# 11   강북구    2165.3      413.600                5.24
# 12   금천구    2178.7      430.400                5.06
# 13   강동구    3292.2      637.000                5.17
# 14  영등포구    3080.7      613.700                5.02
# 15   중랑구    2823.0      551.700                5.12
# 16   구로구    3327.9      626.500                5.31
# 17   광진구    2196.8      437.700                5.02
# 18   노원구    3181.6      663.676                4.79
# 19   관악구    3090.2      604.400                5.11
# 20   강남구    3301.0      473.801                6.97