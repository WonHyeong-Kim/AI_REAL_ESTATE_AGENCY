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
                                         'apt': apt, 'addr_kr': addr_kr, 'city': city, 'area': area, 'area_pyeong': area_pyeong, 'transaction_year_month': transaction_year_month,
                                         'floor': floor, 'parksum': parksum, 'bteacherrate': bteacherrate,
                                         'year_of_completion': year_of_completion,
                                         'maxdate_avgcost': round(maxdate_avgcost),
                                         'avgcost_per_pyeong': format(avgcost_per_pyeong, ".1f"),
                                         'gu_cctv': gu_data.gu_cctv})


def ModelFunc(request):
    '''
    상관분석 : 두 변수 간에 상관관계의 강도를 분석
    이론적 타당성(독립성) 확인. 독립변수 대상 변수들은 서로간에 독립적이어야함
    독립변수 대상 변수들은 다중공산성이 발생할 수 있는데 이를 확인
    
    apartment_id 범주
    gu 범주
    exclusive_use_area 수치
    year_of_completion 범주
    transaction_year_month 범주
    transaction_date 범주
    floor 범주
    park_area_sum 수치
    day_care_babyTeacher_rate 수치
    transaction_real_price 수치
    '''
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    path = os.getcwd()
    df = pd.read_csv(path+'/ai_real_estate_agency/predictapp/static/dataset/train_park_daycare.csv')
    #df = pd.read_csv("train_park_daycare.csv")
    
    pd.set_option('display.max_row', 500) # 컬럼 다 보기
    pd.set_option('display.max_columns', 100)
    #print(df.head(3))
    #print(df.describe())
    
    print('==================표준편차 뽑아보기===================')
    print(np.std(df.transaction_real_price)) # 33868.30
    
    print('==================공분산 보기===================')
    print(np.cov(df.apartment_id, df.transaction_real_price)) # -1.97
    print(np.cov(df.gu, df.transaction_real_price)) # 2.33
    print(np.cov(df.year_of_completion, df.transaction_real_price)) # -1.54
    print(np.cov(df.transaction_year_month, df.transaction_real_price)) # 1.13
    print(np.cov(df.transaction_date, df.transaction_real_price)) # 5.59
    print(np.cov(df.floor, df.transaction_real_price)) # 2.85
    print(np.cov(df.exclusive_use_area, df.transaction_real_price)) # 6.49
    print(np.cov(df.park_area_sum, df.transaction_real_price)) # -1.80
    print(np.cov(df.day_care_babyTeacher_rate, df.transaction_real_price)) # 4.94
    print(df.cov()) # pandas 이용
    
    print('==================상관계수 보기===================')
    print(np.corrcoef(df.transaction_real_price, df.exclusive_use_area)) # 0.6687
    print(np.corrcoef(df.transaction_real_price, df.day_care_babyTeacher_rate)) # 0.236
    # pandas 이용. 괄호안에 method='pearson'피어슨계수가 default
    # print('pearson : ',df.corr()) # 변수가 동간, 정규성따름
    #print('spearman : ', df.corr(method='spearman')) # 서열척도임. 정규성X
    #print('kendall : ',df.corr(method='kendall')) #
    
    
    co_re = df.corr()
    print(co_re['transaction_real_price'].sort_values(ascending=False))
    
    
    print('=====================시각화==================')
    fig = plt.gcf()         # 이미지 저장 선언
    df.plot(kind='scatter', x='exclusive_use_area', y='transaction_real_price')
    #plt.show()
    fig.savefig(path+'/ai_real_estate_agency/predictapp/static/model/model1.png') # 이미지 저장
    plt.close(fig)
    
    fig = plt.gcf()         # 이미지 저장 선언
    df.plot(kind='scatter', x='day_care_babyTeacher_rate', y='transaction_real_price')
    #plt.show()
    fig.savefig(path+'/ai_real_estate_agency/predictapp/static/model/model2.png') # 이미지 저장
    plt.close(fig)
    
    #---------------------------------------------------------------------------------------
    #feature label 상관분석
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib as mpl 
    mpl.rcParams['agg.path.chunksize'] = 10000
    pd.set_option('display.max_columns', 500) #모든 열을 볼 수 있다
    
    
    df = pd.read_csv('train_park_daycare.csv')
    
    #print(df.head(3)) 데이터 확인
    
    #등간척도 비율척도 변수만 추출
    df1 = df[['exclusive_use_area','park_area_sum','day_care_babyTeacher_rate','transaction_real_price' ]] #피어슨 상관계수 쓰려고 
    
    
    cor1 = df1.corr() #안써주면 defalut pearson
    
    print(cor1)
    
    #시각화
    fig = plt.gcf()         # 이미지 저장 선언
    df1_heatmap = sns.heatmap(cor1, cbar=True, annot=True, fmt='.3f', square=True, cmap='Blues')
    #plt.show()
    fig.savefig(path+'/ai_real_estate_agency/predictapp/static/model/model3.png') # 이미지 저장
    plt.close(fig)
    print(df.corr())
    '''
    '''
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import optimizers 
    from sklearn.metrics import r2_score
    from tensorflow.python.keras.callbacks import TensorBoard
    from sklearn.preprocessing._data import MinMaxScaler, minmax_scale,\
        StandardScaler, RobustScaler
    
    
    df = pd.read_csv('./train_park_daycare_sample.csv')
    
    dataset = df.values
    x = dataset[:, [2]]
    y = dataset[:, [-1]]
    #print(x[100])
    #print(y[100])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
    print('x_train.shape : ',x_train.shape) #(8505, 9)
    print(x_test.shape) #(3645, 9)
    print(y_train.shape) #(8505, 1)
    print(y_test.shape) #(3645, 1)
    
    
    
    print('-------------------표준화 : (요소값-평균) / 표준편차----------------')
    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)
    print(x_train[:2])
    
    def build_model():
        model = Sequential()
        model.add(Dense(64, activation='linear', input_shape=(x_train.shape[1], )))
        model.add(Dense(32, activation='linear'))
        model.add(Dense(1, activation='linear')) # layer 3개
        
        model.compile(loss='mse',optimizer='adam',metrics=['mse'])
        return model
    
    model = build_model()
    print(model.summary())
    
    print('------------------------------ train/test-------------------------------')
    history = model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=0,
                        validation_split=0.3)
    mse_history = history.history['mse'] # loss, mse, val_loss, val_mse 중에서 mse 값만 보기
    print('mse_history: ',mse_history)
    val_history = history.history['val_mse']
    
    
    # 시각화
    plt.plot(mse_history,'r')
    plt.plot(val_history, 'b--') # 두개의 선이 비슷해야함
    plt.xlabel('epoch')
    plt.ylabel('mse, val_mse')
    plt.show()
    
    
    print('설명력 : ',r2_score(y_test, model.predict(x_test))) #설명력 :  0.76281
    '''
    return render(request, 'model.html')
# def model():

# 
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