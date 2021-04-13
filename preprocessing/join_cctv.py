import pandas as pd

# 데이터 로드
train_data = pd.read_csv('./dataset/train_park_daycare.csv')
cctv = pd.read_csv("./dataset/cctv_origin.csv", encoding="EUC-KR")

## 데이터 전처리
# 데이터 추출
cctv = cctv.iloc[1:, :2]

# 구 매핑
gu_dict_num = {'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포': 5, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9, '도봉구': 10, '동작구': 11, '강서구': 12, '동대문': 13, '동대문구': 13, '강북구': 14, '서대문': 15, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18, '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24}
gu_list = []
for i in cctv['구분']:
    gu_list.append(gu_dict_num[i])
cctv['gu'] = gu_list
cctv.drop(['구분'], axis=1, inplace=True)

# 컬럼 이름 변경
cctv = cctv.rename(columns={'총계': 'number of cctv'})

# 데이터 타입 변경
cctv['number of cctv'] = cctv['number of cctv'].apply(lambda x: "".join(x.split(',')))
cctv['number of cctv'] = pd.to_numeric(cctv['number of cctv'])

# 조인
new_data = pd.merge(train_data, cctv, on='gu', how='left')

print(new_data.info())
# 저장
new_data.to_csv("./dataset/train_add_cctv.csv", header=True, index=False)

