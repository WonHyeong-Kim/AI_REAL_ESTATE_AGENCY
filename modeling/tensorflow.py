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
from sklearn.model_selection import train_test_split

df = pd.read_csv('train_park_daycare_sample.csv')

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