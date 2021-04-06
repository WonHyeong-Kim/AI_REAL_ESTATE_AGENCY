import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

park = pd.read_csv('../dataset/park.csv')

#서울시 데이터만 추출
print(park.shape)
park = park.loc[park['city'] == '서울특별시']

print(park.shape)

#구컬럼 값으로 정렬
park = park.sort_values(by = 'gu', ascending=True)
#print(park.head(5))

#park['gu'] = park['city'] + park['gu']

print(park.head(5))

print(park.loc[park['gu'] == '양천구'])

#구별로 
park_gu = park['gu'].unique()

#print(park_gu.shape)

area_sum = {}
#구이름별로 공원 면적 합 매핑
for i in park_gu:
    #print(i)
    area_sum[i] = park.loc[park['gu'] == i]['park_area'].sum()
    
#print(area_sum)
gu = []
sum_area = []
for key, value in area_sum.items():
    gu.append(key)
    sum_area.append(value)
    
gu_sum_area = pd.DataFrame()
gu_sum_area['gu_name'] = gu
gu_sum_area['sum_area'] = sum_area


gu_sum_area['gu_map'] = gu_sum_area['gu_name'].map({'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9, '도봉구': 10, '동작구': 11, '강서구': 12, '동대문구': 13, '강북구': 14, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18, '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24})

print(gu_sum_area)
gu_sum_area.to_csv('../dataset/park_sum.csv', header=True, index = False)