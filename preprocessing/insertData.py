# 데이터 삽입하기
'''
데이터 삽입방법
1. db 삭제후 테이블 재설정

drop database estate;
create database estate;
use estate;
CREATE TABLE train
(
    `apartment_id`               INT      NULL        COMMENT 'apartment_id',
    `gu`                         INT      NULL        COMMENT 'gu',
    `exclusive_use_area`         FLOAT    NULL        COMMENT 'exclusive_use_area',
    `year_of_completion`         INT      NULL        COMMENT 'year_of_completion',
    `transaction_year_month`     INT      NULL        COMMENT 'transaction_year_month',
    `transaction_date`           INT      NULL        COMMENT 'transaction_date',
    `floor`                      INT      NULL        COMMENT 'floor',
    `park_area_sum`              FLOAT    NULL        COMMENT 'park_area_sum',
    `day_care_babyTeacher_rate`  FLOAT    NULL        COMMENT 'day_care_babyTeacher_rate',
    `transaction_real_price`     INT      NULL        COMMENT 'transaction_real_price'
);
ALTER TABLE train COMMENT 'train';


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
dataset = pd.read_csv('/Users/jk/git/acornTeam1_project2/preprocessing/dataset/dataset.csv')

try:
    engine = create_engine("mysql+mysqldb://root:123@127.0.0.1:3306/estate", encoding='utf-8')
    conn = engine.connect()
    dataset.to_sql(name='train', con=conn, if_exists='replace', index=False)

except Exception as e:
    print('err : ', e)


