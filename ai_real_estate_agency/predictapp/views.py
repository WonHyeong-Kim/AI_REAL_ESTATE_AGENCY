from django.shortcuts import render
import pandas as pd
import os
from predictapp.models import Test
from django.http.response import HttpResponse
import json
import numpy as np

# Create your views here.
def MainFunc(request):
    
    
    return render(request, 'index.html')

def DbLoadFunc(request):
    path = os.getcwd()
    #print(os.getcwd())
    pd.set_option('display.max_columns', None)
    day_care_center_df = pd.read_csv('https://raw.githubusercontent.com/WonHyeong-Kim/PYTHON/main/day_care_center.csv')
    print(day_care_center_df)
    
    park_df = pd.read_csv('https://raw.githubusercontent.com/WonHyeong-Kim/PYTHON/main/park.csv')
    print(park_df)
    
    test_df = pd.read_csv(path+'/ai_real_estate_agency/predictapp/static/dataset/test.csv')
    print(test_df)
    print(len(test_df))
    datas = []
    for i in range(100):
    #for i in range(len(test_df)):
        print(i)
        transaction_id = test_df.loc[i].transaction_id;
        apartment_id   = test_df.loc[i].apartment_id;
        city = test_df.loc[i].city;
        dong = test_df.loc[i].dong;
        jibun = test_df.loc[i].jibun;
        apt = test_df.loc[i].apt;
        addr_kr = test_df.loc[i].addr_kr;
        exclusive_use_area = test_df.loc[i].exclusive_use_area;
        year_of_completion = test_df.loc[i].year_of_completion;
        transaction_year_month = test_df.loc[i].transaction_year_month;
        transaction_date = test_df.loc[i].transaction_date;
        floor = test_df.loc[i].floor;
        dict={'transaction_id':transaction_id,
              'apartment_id':apartment_id,
              'city':city,
              'dong':dong,
              'jibun':jibun,
              'apt':apt,
              'addr_kr':addr_kr,
              'exclusive_use_area':exclusive_use_area,
              'year_of_completion':year_of_completion,
              'transaction_year_month':transaction_year_month,
              'transaction_date':transaction_date,
              'floor':floor
              }
        datas.append(dict)
#         Test(
#              transaction_id = test_df.loc[i].transaction_id,
#              apartment_id   = test_df.loc[i].apartment_id,
#              city = test_df.loc[i].city,
#              dong = test_df.loc[i].dong,
#              jibun = test_df.loc[i].jibun,
#              apt = test_df.loc[i].apt,
#              addr_kr = test_df.loc[i].addr_kr,
#              exclusive_use_area = test_df.loc[i].exclusive_use_area,
#              year_of_completion = test_df.loc[i].year_of_completion,
#              transaction_year_month = test_df.loc[i].transaction_year_month,
#              transaction_date = test_df.loc[i].transaction_date,
#              floor = test_df.loc[i].floor
#              ).save()
    print(datas)
    #return HttpResponse(json.dumps(datas), content_type='application/json')
    return render(request, 'index.html', {'datas':datas})

def InfoFunc(request):
    if request.method == 'GET':
        pd.set_option('display.max_columns', None)
        apt_id = request.GET.get('apartment_id')
        apt_id = 10453;
        print(apt_id)
        # apartment_id로 DB의 정보 조회
        path = os.getcwd()
        # 이름
        # 위치
        test_df = pd.read_csv(path+'/ai_real_estate_agency/predictapp/static/dataset/test.csv')
        print(test_df)
        print(test_df.info())
        #print(test_df.loc[apt_id, ['apartment_id']])
        print(test_df.apartment_id == apt_id)
        df = test_df[test_df.apartment_id == apt_id]
        apt = str(df['apt'].values)[2:-2]
        addr_kr = str(df['addr_kr'].values)[2:-2]
        city = str(df['city'].values)[2:-2]
        area = float(df['exclusive_use_area'].vaues)
        area_pyeong = np.floor(area/ 3.305785 * 100)/100
        transaction_year_month = int(df['transaction_year_month'].values)
        floor = int(df['floor'].values)
        transaction_year_month = transaction_year_month/100
        print(floor)
        print(type(floor))
    return render(request, 'info.html', {'apt':apt, 'addr_kr':addr_kr, 'city':city, 'area':area, 'area_pyeong':area_pyeong, 'transaction_year_month':transaction_year_month, 'floor':floor})
    