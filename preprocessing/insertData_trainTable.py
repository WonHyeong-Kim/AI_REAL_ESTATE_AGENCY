# 데이터 삽입하기
'''
데이터 삽입방법
1. db 삭제후 db 재생성
drop database estate;
create database estate;
use estate;

2.모듈이 설치가 안되신분만 (pymysql, sqlalchemy)
pip install pymysql
pip install sqlalchemy

3.insertData_trainTable.py 실행

주의할점 : 포트번호 및 기타 설정사항을 체크
'''

import pandas as pd
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 데이터 로드
dataset_train = pd.read_csv('./dataset/train_add_kremap.csv')
dataset = pd.read_csv('./dataset_pre/train.csv')

try:
    engine = create_engine("mysql+mysqldb://root:123@127.0.0.1:3306/estate", encoding='utf-8')
    conn = engine.connect()
    dataset_train.to_sql(name='train', con=conn, if_exists='replace', index=False)
    print('dataset_train end')
    dataset.to_sql(name='dataset', con=conn, if_exists='replace', index=False)

except Exception as e:
    print('err : ', e)
print('end')

