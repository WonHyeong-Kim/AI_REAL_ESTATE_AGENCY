from django.shortcuts import render
import pandas as pd
import os
#from predictapp.models import Test
from django.http.response import HttpResponse
import json
import numpy as np
from predictapp.models import Dataset
from predictapp.models import Gu

from predictapp.models import Train


def MainFunc(request):
    return render(request, 'index.html')
    
def PredictFunc(request):
#     path = os.getcwd()
    #print(os.getcwd())
    pd.set_option('display.max_columns', None)
    #day_care_center_df = pd.read_csv('https://raw.githubusercontent.com/WonHyeong-Kim/PYTHON/main/day_care_center.csv')
    #print(day_care_center_df)
    
    #park_df = pd.read_csv('https://raw.githubusercontent.com/WonHyeong-Kim/PYTHON/main/park.csv')
    #print(park_df)
    #df = pd.read_csv(path+'/ai_real_estate_agency/predictapp/static/dataset/test.csv')
    df = Dataset.objects.all()
    print(len(df))
    print(df)
    print(type(df))
    i = 0
    datas = []
    for d in df:
        print(d.apartment_id)
        dict ={'apartment_id':d.apartment_id, 'apt':d.apt, 'addr_kr':d.addr_kr}
        datas.append(dict)
        i = i + 1
        if i == 1000:
            break
    #print(df)
    #print(len(df))
    print(datas)
    '''
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
    '''
    #return HttpResponse(json.dumps(datas), content_type='application/json')
    #return render(request, 'predict.html')
    return render(request, 'predict.html', {'datas':datas})

def InfoFunc(request):
    global city
    if request.method == 'GET':
        pd.set_option('display.max_columns', None)
        apt_id = request.GET.get('apartment_id')
        print(apt_id)
        dataset = Dataset.objects.filter(apartment_id = apt_id)
        dataset_Train = Train.objects.filter(apartment_id=apt_id)
        # apartment_id로 DB의 정보 조회
        #path = os.getcwd()
        #test_df = pd.read_csv(path+'/ai_real_estate_agency/predictapp/static/dataset/test.csv')
        #print(test_df)
        #print(test_df.info())
        #print(test_df.loc[apt_id, ['apartment_id']])
        #print(test_df.apartment_id == apt_id)
        #df = test_df[test_df.apartment_id == apt_id]
        #print(d)
        for d in dataset:
            apt = d.apt
            addr_kr = d.addr_kr
            city = d.city
            area = float(d.exclusive_use_area)
            area_pyeong = np.floor(area/ 3.305785 * 100)/100
            transaction_year_month = d.transaction_year_month
            floor = int(d.floor)
            transaction_year_month = d.transaction_year_month/100

        '''
        apt = str(df['apt'].values)[2:-2]
        addr_kr = str(df['addr_kr'].values)[2:-2]
        city = str(df['city'].values)[2:-2]
        area = float(df['exclusive_use_area'].values)
        area_pyeong = np.floor(area/ 3.305785 * 100)/100
        transaction_year_month = int(df['transaction_year_month'].values)
        floor = int(df['floor'].values)
        transaction_year_month = transaction_year_month/100
        '''

        # 구 평균 거래가
        print(dataset_Train[0].gu)
        gu_data = Gu.objects.get(gu_num=dataset_Train[0].gu)
        print(gu_data.gu_mean_price)

    return render(request, 'info.html', {'gu_mean_price': gu_data.gu_mean_price,  'dataset': dataset, 'apt':apt, 'addr_kr':addr_kr, 'city':city, 'area':area, 'area_pyeong':area_pyeong, 'transaction_year_month':transaction_year_month, 'floor':floor})

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
#def model():
    