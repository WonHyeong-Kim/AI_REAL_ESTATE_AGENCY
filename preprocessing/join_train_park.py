import pandas as pd

# 데이터 로드
train_data = pd.read_csv("https://raw.githubusercontent.com/WonHyeong-Kim/AI_REAL_ESTATE_AGENCY/main/preprocessing/dataset/train_ver1.csv")
park_sum = pd.read_csv("https://raw.githubusercontent.com/WonHyeong-Kim/AI_REAL_ESTATE_AGENCY/main/preprocessing/dataset/park_sum.csv")
# print(park_sum.info())
# print(park_sum.head(3))

# left_join
new_data = pd.merge(train_data, park_sum, left_on='gu', right_on='gu_name', how='left')
# print(new_data.info())
# print(new_data.isnull().any())
# print(new_data)
new_data.drop(['gu_name'], inplace=True, axis=1)
new_data.fillna(0, inplace=True)
# print(new_data.isnull().any())
# print(new_data.info())

new_data.to_csv('train_park.csv', header=True, index=False)




