import pandas as pd

# train dataset, babycenter dataset
# /Users/jk/Downloads/baby_canter.csv
train_data = pd.read_csv('https://raw.githubusercontent.com/WonHyeong-Kim/AI_REAL_ESTATE_AGENCY/main/preprocessing/dataset/train_park.csv')
dayCare_data = pd.read_csv('https://raw.githubusercontent.com/WonHyeong-Kim/AI_REAL_ESTATE_AGENCY/main/preprocessing/dataset/baby_center.csv')
gu_num_dict = {'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9, '도봉구': 10, '동작구': 11, '강서구': 12, '동대문구': 13, '강북구': 14, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18, '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24}
# print(len(gu_num_dict))
# print(dayCare_data.info())
# print(dayCare_data.head(3))

# 전처리 - drop
dayCare_data.drop(['baby_sum', 'teacher_sum'], axis=1, inplace=True)
# 전처리 - gu 매핑
# print(dayCare_data.isnull().any())
gu_list = []
for i in dayCare_data['gu']:
    gu_list.append(gu_num_dict[i])

print(len(gu_list))
dayCare_data['gu_num'] = gu_list
dayCare_data.drop(['gu'], axis=1, inplace=True)
# print(dayCare_data)

# left-join
new_data = pd.merge(train_data, dayCare_data, left_on='gu', right_on='gu_num', how='left')
# print(new_data.info())
# print(new_data.head(3))

new_data.drop(['gu_num'], axis=1, inplace=True)
new_data.fillna(new_data.mean(), inplace=True)
# print(new_data.head(3))
# print(new_data.info())
print(new_data.isnull().any())

# 컬럼 순서 및 이름 변경
new_data = new_data.reindex(columns=[
    'apartment_id', 'gu', 'exclusive_use_area', 'year_of_completion',
    'transaction_year_month', 'transaction_date', 'floor', 'sum_area',
    'baby/teacher point', 'transaction_real_price'])
new_data = new_data.rename(columns={'sum_area': 'park_area_sum', 'baby/teacher point': 'day_care_babyTeacher_rate'})
pd.set_option('display.max_columns', 100)
print(new_data.head(3))
print(new_data.info())

new_data.to_csv('train_park_daycare.csv', header=True, index=False)

