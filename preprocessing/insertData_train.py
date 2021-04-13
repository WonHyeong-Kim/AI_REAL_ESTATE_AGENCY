import pandas as pd
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

# 데이터 로드
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
dataset = pd.read_csv('./dataset_pre/train.csv')
print(dataset.head())
print()
#    transaction_id  apartment_id   city dong   jibun       apt  \
# 0               0          7622  서울특별시  신교동    6-13  신현(101동)   
# 1               1          5399  서울특별시  필운동     142    사직파크맨션   
# 2               2          3578  서울특별시  필운동   174-1    두레엘리시안   
# 3               3         10957  서울특별시  내수동      95     파크팰리스   
# 4               4         10639  서울특별시  내수동  110-15      킹스매너   
# 
#              addr_kr  exclusive_use_area  year_of_completion  \
# 0  신교동 6-13 신현(101동)               84.82                2002   
# 1     필운동 142 사직파크맨션               99.17                1973   
# 2   필운동 174-1 두레엘리시안               84.74                2007   
# 3       내수동 95 파크팰리스              146.39                2003   
# 4    내수동 110-15 킹스매너              194.43                2004   
# 
#    transaction_year_month transaction_date  floor  transaction_real_price  
# 0                  200801            21~31      2                   37500  
# 1                  200801             1~10      6                   20000  
# 2                  200801             1~10      6                   38500  
# 3                  200801            11~20     15                  118000  
# 4                  200801            21~31      3                  120000


try:
    engine = create_engine("mysql+mysqldb://root:123@127.0.0.1:4406/estate", encoding='utf-8')
    conn = engine.connect()
    dataset.to_sql(name='dataset', con=conn, if_exists='replace', index=False)
 
except Exception as e:
    print('err : ', e)