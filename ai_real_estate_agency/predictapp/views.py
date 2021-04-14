from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.shortcuts import render
import pandas as pd
import os
# from predictapp.models import Test
from django.http.response import HttpResponse
import json
import numpy as np
from predictapp.models import Dataset,News, Train

from predictapp.models import Dataset, News, Gu, Train
from tensorflow.keras.models import load_model
from datetime import datetime

def MainFunc(request):
    '''
    # 네이버 부동산관련 기사 웹크롤링
    import requests
    from bs4 import BeautifulSoup

    # 검색 키워드
    search_word = '부동산'

    # 해당 url의 html문서를 soup 객체로 저장
    url = f'https://m.search.naver.com/search.naver?where=m_news&sm=mtb_jum&query={search_word}'
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    
    search_result = soup.select_one('#news_result_list')
    news_links = search_result.select('.bx > .news_wrap > a')
    
    for i in news_links:
        print(i.get_text())
    for i in news_links:
        print(i['href'])
    '''

    dataset = News.objects.all()
    # print(len(dataset))

    news_datas = []
    for d in dataset:
        # print(d.news_title)
        dict = {'news_id': d.news_id, 'news_title': d.news_title, 'news_link': d.news_link}
        news_datas.append(dict)

    # print(news_datas)

    return render(request, 'index.html', {'news_datas': news_datas})


def ChartFunc(request):
    
    return render(request, 'chart.html')

def GuChart(request):
    #데이터 로드
    dataset = pd.read_csv('https://raw.githubusercontent.com/WonHyeong-Kim/AI_REAL_ESTATE_AGENCY/main/preprocessing/dataset/train_add_cctv.csv')
    response = {}
    #숫자로 매핑되어 있는 구 정보를 다시 구 이름으로 변환
    gu = {'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9, '도봉구': 10, '동작구': 11, '강서구': 12, '동대문구': 13, '강북구': 14, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18, '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24}
    
    gu_name = {}
    for k, v in gu.items():
        gu_name[v] = k
    
    dataset['gu'] = dataset['gu'].map(gu_name)
    #print(dataset['gu'])
    
    #구별 CCTV 추이
    cctv_data = dataset.groupby('gu')['cctv_num'].mean()
    #print(cctv_data)
    #print(list(cctv_data.values))
    response['gu_name'] = list(cctv_data.index)
    response['cctv'] = list(map(str, cctv_data.values))
    
    #구별 교육지수 평균 추이
    edu_data = Train.objects.all()
    #print(len(edu_data))
    
    edu_gu = []
    edu_rate = []
    
    for a in edu_data:
        edu_gu.append(a.gu)
        edu_rate.append(a.day_care_babyteacher_rate)
    
    #print(len(edu_gu))
    edu_data = pd.DataFrame()
    edu_data['gu_name'] = edu_gu
    edu_data['edu_rate'] = edu_rate
    
    edu_data = edu_data.groupby('gu_name')['edu_rate'].mean()
    response['edu_rate'] = list(map(str, edu_data.values))
    
    
    #클라이언트로부터 구 이름 정보를 받고 해당하는 구별 데이터 추출
    dataset = dataset.loc[dataset['gu'] == request.GET.get("gu")]
    #print(dataset[['gu', 'floor']])
    
    #구별 거래년월별 거래액 평균
    transaction_data = dataset.groupby('transaction_year_month')['transaction_real_price'].mean()
    #print(data)
    
    #데이터를 json형식으로 보내기
    response['date'] = list(transaction_data.index)
    response['price'] = list(transaction_data.values)
    
    #print(response)
    
    return HttpResponse(json.dumps(response), content_type = 'application/json')

def PredictFunc(request):
    pd.set_option('display.max_columns', None)

    df = Dataset.objects.all()

    i = 0
    datas = []
    chk = []
    for d in df:
        if d.apartment_id not in chk:
            dict = {'apartment_id': d.apartment_id, 'apt': d.apt, 'addr_kr': d.addr_kr}
            datas.append(dict)
            chk.append(d.apartment_id)
            i = i + 1
            if i == 1000:
                break

    # print(datas)
    return render(request, 'predict.html', {'datas': datas})


def InfoFunc(request):
    if request.method == 'GET':
        pd.set_option('display.max_columns', None)
        apt_id = request.GET.get('apartment_id')
        datasets = Dataset.objects.filter(apartment_id=apt_id)
        train = Train.objects.filter(apartment_id=apt_id)

        # 데이터 페이징 처리
        paginator = Paginator(datasets, 5)
        page = request.GET.get('page')

        try:
            dataset = paginator.page(page)
        except PageNotAnInteger:
            dataset = paginator.page(1)
        except EmptyPage:
            dataset = paginator.page(paginator.num_pages)


        for idx, d in enumerate(dataset):
            apt = d.apt
            addr_kr = d.addr_kr
            city = d.city
            area = float(d.exclusive_use_area)

            area_pyeong = np.floor(area / 3.305785 * 100) / 100
            dataset[idx].exclusive_use_area = np.round(d.exclusive_use_area, 2)
            transaction_year_month = d.transaction_year_month
            floor = int(d.floor)

            transaction_year_month = d.transaction_year_month / 100


        last_transaction = train[len(train)-1].transaction_year_month
        last_transaction_price_sum = 0
        last_transaction_area_sum = 0
        cont = 0
        gu = 20
        for t in train:
            parksum = t.park_area_sum  # 해당 구 공원면적
            bteacherrate = t.day_care_babyteacher_rate  # 해당 구 아기 대비 유치원교사 비율
            area = float(t.exclusive_use_area)
            area_pyeong = np.floor(area / 3.305785 * 100) / 100  # 평수
            year_of_completion = t.year_of_completion  # 완공연도
            
            # CCTV 수

            '''해당지역 최근 거래내역 : 거래액'''
            # print(t.transaction_real_price)
            # print(t.transaction_year_month)
            # print(type(t.transaction_year_month))
            if t.transaction_year_month == last_transaction:
                cont += 1
                last_transaction_price_sum += t.transaction_real_price
                last_transaction_area_sum += area_pyeong

        maxdate_avgcost = last_transaction_price_sum / cont

        '''최근 해당 단지 평당 평균 거래액'''
        avgcost_per_pyeong = maxdate_avgcost / last_transaction_area_sum


        '''평균 거래액(구별)'''
        gu_data = Gu.objects.get(gu_num=train[0].gu)

    return render(request, 'info.html', {'apartment_id': apt_id, 'gu_mean_price': format(gu_data.gu_mean_price, ".1f"), 'dataset': dataset,
                                         'apt': apt, 'addr_kr': addr_kr, 'city': city, 'gu' : gu, 'area': area, 'area_pyeong': area_pyeong, 'transaction_year_month': transaction_year_month,
                                         'floor': floor, 'parksum': parksum, 'bteacherrate': bteacherrate,
                                         'year_of_completion': year_of_completion,
                                         'maxdate_avgcost': round(maxdate_avgcost),
                                         'avgcost_per_pyeong': format(avgcost_per_pyeong, ".1f"),
                                         'gu_cctv': gu_data.gu_cctv})


def ModelFunc(request):
    return render(request, 'model.html')

# def ListFunc(request):
#     dataset = Dataset.objects.filter(apartment_id=apt_id)
#     paginator = Paginator(dataset, 5)
#     page = request.GET.get('page')
# 
#     try:
#         data = paginator.page(page)
#     except PageNotAnInteger:
#         data = paginator.page(1)
#     except EmptyPage:
#         data = paginator.page(paginator.num_pages)
# 
#     return render(request, 'info.html', {'data': data})

def LoadingFunc(request):
    return render(request, 'loading.html')

def FeaturePriceFunc(request):
    transaction_id              = 1
    apartment_id                = int(request.GET.get('apartment_id'))  
    gu                          = int(request.GET.get('gu'))
    exclusive_use_area          = float(request.GET.get('exclusive_use_area'))
    year_of_completion          = int(request.GET.get('year_of_completion'))
    transaction_year_month      = int(request.GET.get('year'))
    transaction_date            = 0
    floor                       = int(request.GET.get('floor'))
    park_area_sum               = float(request.GET.get('park_area_sum'))
    day_care_babyTeacher_rate   = float(request.GET.get('day_care_babyTeacher_rate'))
    cctv_num                    = int(request.GET.get('cctv_num'))
    print(transaction_year_month, apartment_id, gu, exclusive_use_area, year_of_completion)
    print(floor, park_area_sum, day_care_babyTeacher_rate, cctv_num)
    
    transaction_year_month = datetime.today().year * 100 + transaction_year_month * 100 + 1
    print(transaction_year_month)
    
    data = []
    
    path = os.getcwd()
    model = load_model(path+'/ai_real_estate_agency/predictapp/static/model/tensormodel.h5')
    print(model.summary())
    print(type(transaction_id), type(apartment_id), type(gu), type(exclusive_use_area), type(year_of_completion), type(transaction_year_month))
    print(type(floor), type(park_area_sum), type(day_care_babyTeacher_rate), type(cctv_num))
# 0   transaction_id             742285 non-null  int64  
# 1   apartment_id               742285 non-null  int64  
# 2   gu                         742285 non-null  int64  
# 3   exclusive_use_area         742285 non-null  float64
# 4   year_of_completion         742285 non-null  int64  
# 5   transaction_year_month     742285 non-null  int64  
# 6   transaction_date           742285 non-null  int64  
# 7   floor                      742285 non-null  int64  
# 8   park_area_sum              742285 non-null  float64
# 9   day_care_babyTeacher_rate  742285 non-null  float64
# 10  transaction_real_price     742285 non-null  int64  
# 11  cctv_num                   742285 non-null  int64 

    new_x = [[transaction_id, apartment_id, gu, exclusive_use_area, year_of_completion, transaction_year_month, transaction_date, floor, park_area_sum, day_care_babyTeacher_rate, cctv_num]]
    print(new_x)
    featurePrice = model.predict(new_x)
    print(featurePrice)
    dict ={"featurePrice":abs(int(featurePrice))}
    print(dict)
    data.append(dict)
    print(data)
    return HttpResponse(json.dumps(data), content_type = 'application/json')


