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

3.insertData.py 실행

주의할점 : 포트번호 및 기타 설정사항을 체크
'''

import pandas as pd
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

# 데이터 로드
dataset = pd.read_csv('https://raw.githubusercontent.com/WonHyeong-Kim/AI_REAL_ESTATE_AGENCY/main/preprocessing/dataset/train_add_cctv.csv')

try:
    engine = create_engine("mysql+mysqldb://root:123@127.0.0.1:3306/estate", encoding='utf-8')
    conn = engine.connect()
    dataset.to_sql(name='train', con=conn, if_exists='replace', index=False)

except Exception as e:
    print('err : ', e)


