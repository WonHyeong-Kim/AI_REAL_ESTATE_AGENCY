import pandas as pd
import urllib.request as req
from urllib.parse import quote
from bs4 import BeautifulSoup
import pickle

# 데이터 전처리하기
# 원본 파일이 100mb 가 넘어서 해당 데이터는 로컬에서 처리해야됨.
data = pd.read_csv('/Users/jk/study/java/java_academic/파이널프로젝트/파이널2/데이터/dacon/train.csv')
data = data[
    ['transaction_id', 'apartment_id', 'city', 'dong', 'exclusive_use_area', 'year_of_completion', 'transaction_year_month', 'transaction_date',
     'floor', 'transaction_real_price']]

# 결측치 파악
# print(data.isnull().any())      # 이상 무


# 매핑
# 서울 데이터 추출
data = data[data['city'] == '서울특별시']
# print(data.shape) (742285, 9)
data = data.drop(['city'], axis=1)


# transaction_date 매핑(1~10, 11~20, 21~31)
data['transaction_date'] = data['transaction_date'].map(
    {'1~10': 0, '11~20': 1, '21~31': 2, '21~29': 3, '21~30': 4, '21~28': 5})

# print(data.info())

# 구 추가하기. 크롤링.
gu = []
gu_dict = {}
gu_set = set()
for i, j in enumerate(data['dong'].unique()):
    dong = quote("서울특별시 " + j)
    base_url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query="
    url = base_url + dong
    search = req.urlopen(url).read().decode('utf8')
    soup = BeautifulSoup(search, 'lxml')
    addr = soup.select_one('#loc-main-section-root > div > div > div:nth-child(2) > div._32HJW > div > div._1fsxy > span').string
    # print(i, '번 ', j, '  ', addr)
    addr_gu = addr.split()[1]
    gu_dict[j] = addr_gu        # gu_dict = {'신교동': 종로구}
    gu_set.add(addr_gu)         # gu_set = {'종로구', '성북구'}
    print(addr_gu)


# 구 맵핑 코드
gu_dict_num = {'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포': 5, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9, '도봉구': 10, '동작구': 11, '강서구': 12, '동대문': 13, '동대문구': 13, '강북구': 14, '서대문': 15, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18, '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24}
print(gu_dict_num)

# pickle로 저장
# with open('gu_dong_dict.bin', 'wb') as f:
#     pickle.dump(gu_dict, f)
#
# with open('gu_num_dict.bin', 'wb') as f:
#     pickle.dump(gu_dict_num, f)


for m in data['dong']:
    gu.append(gu_dict_num[gu_dict[m]])
# print(gu)
data['gu'] = gu  
data = data.drop(['dong'], axis=1)
print(data.info())  
data.to_csv("./dataset/train_ver1.csv", header=True, index=False)




