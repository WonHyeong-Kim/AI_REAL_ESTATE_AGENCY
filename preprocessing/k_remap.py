import pandas as pd
import pymysql
pymysql.install_as_MySQLdb()


k_remap_origin = pd.read_csv("/Users/jk/git/acornTeam1_project2/preprocessing/dataset/k_remap_origin.csv")

# 서울특별시만 뽑기.
k_remap_seoul = k_remap_origin.iloc[20:45, :-1]

# 컬럼명 rename
rename_dic = {}
for i in k_remap_seoul:
    if i == '지역명':
        rename_dic[i] = 'gu_name'
    else:
        rename_dic[i] = "".join(i.split('-'))

k_remap_seoul = k_remap_seoul.rename(columns=rename_dic)


# 결측치 추가. 200801 ~ 201008까지 추가
for i in range(32):     # 12 + 12 + 8 개월
    a = i // 12     # 몫
    b = i % 12      # 나머지
    year = str(2008 + a)
    month = str(1 + b)
    if len(month) == 1:
        month = '0' + month
    year_month = year + month
    k_remap_seoul[year_month] = k_remap_seoul['201009']


# print(k_remap_seoul.columns)        # 200801~ 202102, length=159
col_reindex = ['gu_name', '200801', '200802', '200803', '200804', '200805', '200806', '200807', '200808', '200809', '200810', '200811', '200812', '200901', '200902', '200903', '200904', '200905', '200906', '200907', '200908', '200909', '200910', '200911', '200912', '201001', '201002', '201003', '201004', '201005', '201006', '201007', '201008', '201009', '201010', '201011', '201012', '201101', '201102', '201103', '201104', '201105', '201106', '201107', '201108', '201109', '201110', '201111', '201112', '201201', '201202', '201203', '201204', '201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201301', '201302', '201303', '201304', '201305', '201306', '201307', '201308', '201309', '201310', '201311', '201312', '201401', '201402', '201403', '201404', '201405', '201406', '201407', '201408', '201409', '201410', '201411', '201412', '201501', '201502', '201503', '201504', '201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512', '201601', '201602', '201603', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612', '201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901', '201902', '201903', '201904', '201905', '201906', '201907', '201908', '201909', '201910', '201911', '201912', '202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008', '202009', '202010', '202011', '202012', '202101', '202102']
k_remap_seoul = k_remap_seoul.reindex(columns=col_reindex)


# new_data 만들기. 새로운 dataframe 으로.
gu_name_list = list(k_remap_seoul['gu_name'].unique())
year_month_list = list(k_remap_seoul.columns[1:])
gu_name_list_cp = []
year_month_list_cp = []
for i in gu_name_list:
    for _ in range(158):
        gu_name_list_cp.append(i)
for _ in range(25):
    for i in year_month_list:
        year_month_list_cp.append(i)

data = {'gu_name': gu_name_list_cp, 'year_month': year_month_list_cp}       # 3950

new_data = pd.DataFrame(data)
new_data['k_remap'] = 0


# 대망의 데이터 넣기
for idx, remap in new_data.iterrows():
    gu_name = remap.gu_name
    year_month = remap.year_month
    # 해당 행의 조건을 만족하는 값 찾기.
    value = k_remap_seoul[year_month][k_remap_seoul['gu_name'] == gu_name]
    # 각 행의 k-remap
    new_data.iloc[idx, -1] = value.iloc[0]

new_data.to_csv('./dataset/k_remap.csv', header=True, index=False)
print(new_data.info())
print(new_data)



