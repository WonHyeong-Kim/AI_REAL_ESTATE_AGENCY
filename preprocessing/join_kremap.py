import pandas as pd

train = pd.read_csv('./dataset/train_add_cctv.csv')
k_remap = pd.read_csv('./dataset/k_remap.csv')

# 구 매핑
gu_dict_num = {'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포': 5, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9, '도봉구': 10, '동작구': 11, '강서구': 12, '동대문': 13, '동대문구': 13, '강북구': 14, '서대문': 15, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18, '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24}
gu_list = []
for i in k_remap['gu_name']:
    gu_list.append(gu_dict_num[i])
k_remap['gu'] = gu_list

# 조인
new_data = pd.merge(train, k_remap, left_on=['gu', 'transaction_year_month'], right_on=['gu', 'year_month'], how='left')

# 불필요한 열 제거
pd.set_option('display.max_columns', None)
new_data.drop(['gu_name', 'year_month'], axis=1, inplace=True)

# 저장
new_data.to_csv("./dataset/train_add_kremap.csv", header=True, index=False)

'''
new_data
<class 'pandas.core.frame.DataFrame'>
Int64Index: 742285 entries, 0 to 742284
Data columns (total 13 columns):
 #   Column                     Non-Null Count   Dtype  
---  ------                     --------------   -----  
 0   transaction_id             742285 non-null  int64  
 1   apartment_id               742285 non-null  int64  
 2   gu                         742285 non-null  int64  
 3   exclusive_use_area         742285 non-null  float64
 4   year_of_completion         742285 non-null  int64  
 5   transaction_year_month     742285 non-null  int64  
 6   transaction_date           742285 non-null  int64  
 7   floor                      742285 non-null  int64  
 8   park_area_sum              742285 non-null  float64
 9   day_care_babyTeacher_rate  742285 non-null  float64
 10  transaction_real_price     742285 non-null  int64  
 11  cctv_num                   742285 non-null  int64  
 12  k_remap                    742285 non-null  float64
dtypes: float64(4), int64(9)
memory usage: 79.3 MB
'''