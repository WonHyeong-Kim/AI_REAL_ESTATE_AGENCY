from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.shortcuts import render
import pandas as pd
import os
# from predictapp.models import Test
from django.http.response import HttpResponse
import json
import numpy as np
from predictapp.models import Dataset, News, Gu, Train


def MainFunc(request):
    dataset = News.objects.all()
    # print(len(dataset))

    news_datas = []
    for d in dataset:
        # print(d.news_title)
        dict = {'news_id': d.news_id, 'news_title': d.news_title, 'news_link': d.news_link}
        news_datas.append(dict)

    return render(request, 'index.html', {'news_datas': news_datas})


def ChartFunc(request):
    return render(request, 'chart.html')


def GuChart(request):
    # 데이터 로드
    dataset = pd.read_csv('../../../preprocessing/dataset/train_add_cctv.csv')
    response = {}
    # 숫자로 매핑되어 있는 구 정보를 다시 구 이름으로 변환
    gu = {'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9,
          '도봉구': 10, '동작구': 11, '강서구': 12, '동대문구': 13, '강북구': 14, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18,
          '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24}

    gu_name = {}
    for k, v in gu.items():
        gu_name[v] = k

    dataset['gu'] = dataset['gu'].map(gu_name)
    # print(dataset['gu'])

    # 구별 CCTV 추이
    cctv_data = dataset.groupby('gu')['cctv_num'].mean()
    # print(cctv_data)
    # print(list(cctv_data.values))
    response['gu_name'] = list(cctv_data.index)
    response['cctv'] = list(map(str, cctv_data.values))

    # 구별 교육지수 평균 추이
    edu_data = Train.objects.all()
    # print(len(edu_data))

    edu_gu = []
    edu_rate = []

    for a in edu_data:
        edu_gu.append(a.gu)
        edu_rate.append(a.day_care_babyteacher_rate)

    # print(len(edu_gu))
    edu_data = pd.DataFrame()
    edu_data['gu_name'] = edu_gu
    edu_data['edu_rate'] = edu_rate

    edu_data = edu_data.groupby('gu_name')['edu_rate'].mean()
    response['edu_rate'] = list(map(str, edu_data.values))

    # 클라이언트로부터 구 이름 정보를 받고 해당하는 구별 데이터 추출
    dataset = dataset.loc[dataset['gu'] == request.GET.get("gu")]
    # print(dataset[['gu', 'floor']])

    # 구별 거래년월별 거래액 평균
    transaction_data = dataset.groupby('transaction_year_month')['transaction_real_price'].mean()
    # print(data)

    # 데이터를 json형식으로 보내기
    response['date'] = list(transaction_data.index)
    response['price'] = list(transaction_data.values)

    # print(response)

    return HttpResponse(json.dumps(response), content_type='application/json')



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

        last_transaction = train[len(train) - 1].transaction_year_month
        last_transaction_price_sum = 0
        last_transaction_area_sum = 0
        cont = 0

        # 구 이름 얻기
        train_gu = train[0].gu
        gu_dict = {'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9,
                   '도봉구': 10, '동작구': 11, '강서구': 12, '동대문구': 13, '강북구': 14, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18,
                   '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24}
        gu_name = {}
        for k, v in gu_dict.items():
            gu_name[v] = k
        gu_name = gu_name[train_gu]
        # print(train_gu)
        # print(gu_name)

        for t in train:
            parksum = t.park_area_sum  # 해당 구 공원면적
            bteacherrate = t.day_care_babyteacher_rate  # 해당 구 아기 대비 유치원교사 비율
            area = float(t.exclusive_use_area)
            area_pyeong = np.floor(area / 3.305785 * 100) / 100  # 평수
            year_of_completion = t.year_of_completion  # 완공연도
            k_remap = t.k_remap  # 부동산 활성화 지수

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

    return render(request, 'info.html',
                  {'apartment_id': apt_id, 'gu_mean_price': format(gu_data.gu_mean_price, ".1f"), 'dataset': dataset,
                   'apt': apt, 'addr_kr': addr_kr, 'city': city, 'gu_name': gu_name, 'gu': train_gu, 'area': area,
                   'area_pyeong': area_pyeong, 'transaction_year_month': transaction_year_month,
                   'floor': floor, 'parksum': parksum, 'bteacherrate': bteacherrate,
                   'year_of_completion': year_of_completion,
                   'maxdate_avgcost': round(maxdate_avgcost),
                   'avgcost_per_pyeong': format(avgcost_per_pyeong, ".1f"),
                   'gu_cctv': gu_data.gu_cctv,
                   'k_remap': k_remap})


def ModelFunc(request):
    return render(request, 'model.html')


def LoadingFunc(request):
    return render(request, 'loading.html')

