import pandas as pd
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()


train_data = pd.read_csv('./dataset/train_add_cctv.csv')
gu_dict_num = {'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포': 5, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9, '도봉구': 10, '동작구': 11, '강서구': 12, '동대문': 13, '동대문구': 13, '강북구': 14, '서대문': 15, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18, '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24}

# 구 매핑, 구별 평균, 구별 공원, 구별 cctv
gu_dic = {}
gu_name = []
gu_num = []
gu_area = [0 for _ in range(25)]
gu_cctv = [0 for _ in range(25)]
gu_daycare = [0 for _ in range(25)]
chk = [True for _ in range(25)]

# gu_area, gu_cctv, gu_daycare 리스트 제작
for idx, data in train_data.iterrows():
    gu = int(data['gu'])
    # 방문하지 않았으면 chk[gu] = True
    if chk[gu]:
        gu_area[gu] = data['park_area_sum']
        gu_cctv[gu] = int(data['number of cctv'])
        gu_daycare[gu] = data['day_care_babyTeacher_rate']
        chk[gu] = False


# gu_name, gu_num 리스트 제작
for key, value in gu_dict_num.items():
    gu_name.append(key)
    gu_num.append(value)

# gu_mean_price 리스트 제작
gu_mean_price = list(train_data.groupby(['gu'])['transaction_real_price'].mean())

# 중복 데이터 3개 구 추가(동대문 = 동대문구, 서대문 = 서대문구, 등)
for i in [15, 13, 5]:
    gu_area.insert(i, gu_area[i])
    gu_cctv.insert(i, gu_cctv[i])
    gu_daycare.insert(i, gu_daycare[i])
    gu_mean_price.insert(i, gu_mean_price[i])


gu_dic['gu_name'] = gu_name
gu_dic['gu_num'] = gu_num
gu_dic['gu_area'] = gu_area
gu_dic['gu_cctv'] = gu_cctv
gu_dic['gu_daycare'] = gu_daycare
gu_dic['gu_mean_price'] = gu_mean_price
# print(len(gu_dic['gu_num']))    # 28

gu_data = pd.DataFrame(gu_dic, columns=['gu_name', 'gu_num', 'gu_area', 'gu_daycare', 'gu_cctv', 'gu_mean_price'], index=None)


try:
    engine = create_engine("mysql+mysqldb://root:123@127.0.0.1:3306/estate", encoding='utf-8')
    conn = engine.connect()
    gu_data.to_sql(name='gu', con=conn, if_exists='replace', index=False)

except Exception as e:
    print('err : ', e)


