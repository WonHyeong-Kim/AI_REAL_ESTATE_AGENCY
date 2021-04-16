from django.shortcuts import render
import pandas as pd
import os
from django.http.response import HttpResponse
import json
from predictapp.models import Dataset, News, Gu, Train
from tensorflow.keras.models import load_model
from datetime import datetime


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


def FeaturePriceFunc(request):
    transaction_id = 1
    apartment_id = int(request.GET.get('apartment_id'))
    gu = int(request.GET.get('gu'))
    exclusive_use_area = float(request.GET.get('exclusive_use_area'))
    year_of_completion = int(request.GET.get('year_of_completion'))
    transaction_year_month = int(request.GET.get('year'))
    transaction_date = 0
    floor = int(request.GET.get('floor'))
    park_area_sum = float(request.GET.get('park_area_sum'))
    day_care_babyTeacher_rate = float(request.GET.get('day_care_babyTeacher_rate'))
    cctv_num = int(request.GET.get('cctv_num'))
    k_remap = float(request.GET.get('k_remap'))
    print(transaction_year_month, apartment_id, gu, exclusive_use_area, year_of_completion)
    print(floor, park_area_sum, day_care_babyTeacher_rate, cctv_num, k_remap)

    transaction_year_month = datetime.today().year * 100 + transaction_year_month * 100 + 1  # 미래 년도 산출
    # print(transaction_year_month)

    data = []

    path = os.getcwd()
    # model = load_model(path + '/ai_real_estate_agency/predictapp/static/model/tensormodel.h5')
    model = load_model('/Users/jk/git/acornTeam1_project2/ai_real_estate_agency/predictapp/static/model/tensormodel.h5')
    # print(model.summary())
    # print(model.info())
    # print(type(transaction_id), type(apartment_id), type(gu), type(exclusive_use_area), type(year_of_completion), type(transaction_year_month))
    # print(type(floor), type(park_area_sum), type(day_care_babyTeacher_rate), type(cctv_num))
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

    new_x = [[apartment_id, gu, exclusive_use_area, year_of_completion, transaction_year_month, transaction_date, floor,
              park_area_sum, day_care_babyTeacher_rate, cctv_num, k_remap]]
    print(new_x)
    featurePrice = model.predict(new_x)  # 가격 예측
    print(featurePrice)
    data.append({"featurePrice": abs(int(featurePrice))})
    # print(data)
    return HttpResponse(json.dumps(data), content_type='application/json')


def predict_price(request):
    return render(request, 'predict_price.html')


def predict_modeling(request):
    path = os.getcwd()

    gu = int(request.GET.get("gu_name"))
    exclusive_use_area = float(request.GET.get("ex_area"))
    year_of_completion = int(request.GET.get("year_complition"))
    transaction_year_month = int(request.GET.get("trans_year_month"))
    floor = int(request.GET.get("floor"))
    print(gu, exclusive_use_area, year_of_completion, transaction_year_month, floor)
    # 구 이름 매핑
    #     gu = {'용산구': 0, '양천구': 1, '강동구': 2, '관악구': 3, '노원구': 4, '영등포구': 5, '마포구': 6, '서초구': 7, '성동구': 8, '금천구': 9, '도봉구': 10, '동작구': 11, '강서구': 12, '동대문구': 13, '강북구': 14, '서대문구': 15, '광진구': 16, '구로구': 17, '성북구': 18, '강남구': 19, '종로구': 20, '중구': 21, '중랑구': 22, '송파구': 23, '은평구': 24}
    #     name_gu = 0
    #     for i in gu.keys():
    #         if i == gu_name:
    #             name_gu = gu[i]
    # print(name_gu)

    # 구에 해당하는 공원면적, 교육지수,cctv개수, kremap(평균)가져오기
    # dataset = pd.read_csv(path + "/ai_real_estate_agency/predictapp/static/dataset/train_add_kremap.csv")
    dataset = pd.read_csv("../../../preprocessing/dataset/train_add_kremap.csv")
    # dataset = pd.read_csv("https://raw.githubusercontent.com/WonHyeong-Kim/AI_REAL_ESTATE_AGENCY/main/preprocessing/dataset/train_add_kremap.csv")
    gu_data = dataset.loc[dataset['gu'] == gu]
    park_area_sum = float(gu_data.groupby('gu')['park_area_sum'].mean().values[0])
    day_care_babyTeacher_rate = int(gu_data.groupby('gu')['day_care_babyTeacher_rate'].mean().values[0])
    cctv_num = int(gu_data.groupby('gu')['cctv_num'].mean().values[0])
    k_remap = float(gu_data.groupby('gu')['k_remap'].mean().values[0])

    # 새로운 데이터 모델에 삽입
    new_data = pd.DataFrame(
        {'gu': [gu], 'exclusive_use_area': [exclusive_use_area], 'year_of_completion': [year_of_completion],
         'transaction_year_month': [transaction_year_month], \
         'floor': [floor], 'park_area_sum': [park_area_sum], 'day_care_babyTeacher_rate': [day_care_babyTeacher_rate],
         'cctv_num': [cctv_num], 'k_remap': [k_remap]})
    # ['gu','exclusive_use_area','year_of_completion','transaction_year_month',
    # 'floor','park_area_sum','day_care_babyTeacher_rate','cctv_num','k_remap'])
    print(new_data)
    # print(new_data.info())
    # new_data = [[1, 7777, int(name_gu), ex_area, year_complition, trans_year_month,0, floor, park_area,teacher_rate,cctv_num]]
    # new_data = [[20, ex_area, year_complition, trans_year_month, floor, park_area,teacher_rate,cctv_num, k_remap]]
    '''
    print(type(int(name_gu)))
    print(new_data)
    print(type(ex_area))
    print(type(year_complition))
    print(type(trans_year_month))
    print(type(floor))
    print(type(park_area))
    print(type(teacher_rate))
    print(type(cctv_num))
    print(type(k_remap))
    '''
    # new_df = pd.DataFrame(new_data)
    # print(new_df.info())
    import pickle
    # from sklearn.externals import joblib
    model = pickle.load(open(r'/Users/jk/git/acornTeam1_project2/ai_real_estate_agency/predictapp/static/model/ols.h5', 'rb'))
    # model = joblib.load(path + '/ai_real_estate_agency/predictapp/static/model/olsmodel.pkl')
    # model = load_model(r'C:\work\psou\ai_real_estate_agency\predictapp\static\model\olsmodel.h5')
    # model = load_model(path + '/ai_real_estate_agency/predictapp/static/model/olsmodel.hdf5')
    # model = load_model(path + "/ai_real_estate_agency/predictapp/static/model/olsmodel.hdf5")
    # model = load_model("C:/Users/SH/Documents/ai_real_estate_agency/ai_real_estate_agency/predictapp/static/model/tensormodel.h5")
    pred_y = int(model.predict(new_data)[0])
    pred_y = pred_y * 10000
    # pred_y = int(model.predict(new_data).flatten())
    pred = str(abs(pred_y))
    print(pred_y)

    '''
    df = pd.read_csv("https://raw.githubusercontent.com/WonHyeong-Kim/AI_REAL_ESTATE_AGENCY/main/preprocessing/dataset/train_add_kremap.csv")
    #print(df.describe().loc["std"]) #std 값 알아보기. 너무 숫자의 범위가 다르다. scale해봤는데 결과가 더 이상하게 나와서 그냥 숫자의 범위가 달라도 무시하기로함
    print(df.info())
    dfx = pd.DataFrame(df, columns = ['gu','exclusive_use_area','year_of_completion','transaction_year_month','floor','park_area_sum','day_care_babyTeacher_rate','cctv_num','k_remap'])
    #print(dfx) #transaction_id, apartment_id, transaction_real_price제외
    
    dfy =  pd.DataFrame(df, columns = ['transaction_real_price'])
    #print(dfy)
    
    df = pd.concat([dfx,dfy], axis=1) #열을 기준으로 x, y 합친다 새로운 데이터프레임 형성df = pd.concat([dfx,dfy], axis=1) #열을 기준으로 x, y 합친다 새로운 데이터프레임 형성
    
    model = smf.ols(formula = 'transaction_real_price ~ gu + exclusive_use_area + year_of_completion + transaction_year_month + floor + park_area_sum + day_care_babyTeacher_rate + cctv_num + k_remap', \
                    data = df)
    olsmodel = model.fit()
    '''
    # olsmodel.save('olsmodel.h5')

    # loaded_model = load_model('olsmodel.h5')

    # print(loaded_model.history)
    # print('보고서용 : ',olsmodel.summary()) #조정된 결정계수 : 0.544 (54%)

    # pred = olsmodel.predict(new_df)
    # print(pred)

    return HttpResponse(json.dumps({"pred": pred}), content_type='application/json')
