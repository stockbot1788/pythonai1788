import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from env import MarketEnv

env = MarketEnv("data")
import tensorflow as tf
import keras
from keras.utils import np_utils

from keras.layers import Dense, Conv2D, Flatten, Input
from keras.models import Model
from keras.optimizers import RMSprop
from os import path

print("preparing model")
input_layer = Input(shape=(4, 10, 1))
cnn1 = Conv2D(10, kernel_size=(1,8), strides=(1,1), activation='relu')(input_layer)
cnn2 = Conv2D(20, kernel_size=(1,2), strides=(1,1), activation='relu')(cnn1)
flat = Flatten()(cnn2)
fc1 = Dense(56, activation='relu')(flat)
pred = Dense(3, name='pred', activation='softmax')(fc1)
model = Model(inputs=[input_layer], outputs=[pred], name='fc')

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

_path = "model2.h5"
if _path and path.isfile(_path):
    try:
       print("try load weight")
       model.load_weights(_path)
       print("load weight success")
    except Exception as ex:
       print("error",ex)
       sys.exit("Error message")

print("preparing data")
inputX = []
outputY = []
for data in env.data:
    for i in range(20,len(data.m_data)-1-15):
        _all=[]
        for j in range(i-10,i):
            _h = data.m_data[j]._high
            _l = data.m_data[j]._low
            _c = data.m_data[j]._close
            _o = data.m_data[j]._open
            tmp = [_h,_l,_c,_o]
            _all.append(tmp)
        #_all = list(reversed(_all))

        Max = -99999999999
        Min = 999999999999
        for j in range(0,i):
            h = data.m_data[j]._high
            l = data.m_data[j]._low
            h = np.float(h)
            l = np.float(l)
            Max = max([h,Max])
            Min = min([l,Min])                    
        X = np.array(_all)
        X = np.expand_dims(X, axis=0)
        X = X.reshape(-1)
        X = X.astype(np.float)
        X = [Max-np.float(i) for i in X]
        X = [100*i/max(X) for i in X]
        dif = Max-Min
        X =  np.array(X)
        X = np.reshape(X, (4, 10,1))
        inputX.append(X)
        #print(X)
        #work for Y value 
        LongPrice = data.m_data[i]._high
        LongPrice = np.float(LongPrice)
        ShortPrice = data.m_data[i]._low
        ShortPrice = np.float(ShortPrice)
        MaxEarn = -99999
        MaxLoss = 99999
        status = 0
        isLong = 0
        for j in range(1,15):
            tmpPriceHigh = data.m_data[i+j]._high
            tmpPriceHigh = np.float(tmpPriceHigh)

            tmpPriceLow = data.m_data[i+j]._low
            tmpPriceLow = np.float(tmpPriceLow)
            
            earnLong = tmpPriceHigh - LongPrice
            lossLong = tmpPriceLow - LongPrice
            if lossLong <-25 and status == 0:
                status = 1  #long fail
                break
            elif earnLong >25 and status == 0:
                status = 2  #long win
                break
        if status != 2:
            status = 0
            for j in range(1,15):
                tmpPriceHigh = data.m_data[i+j]._high
                tmpPriceHigh = np.float(tmpPriceHigh)

                tmpPriceLow = data.m_data[i+j]._low
                tmpPriceLow = np.float(tmpPriceLow)

                earnShort = ShortPrice - tmpPriceLow
                lossShort = ShortPrice - tmpPriceHigh
                if lossShort <-25 and status == 0:
                    status = 1
                    break
                elif earnShort >25 and status == 0:
                    status = 3
                    break
        if status !=2 and status !=3:
            status = 1
        status = status - 1
        # if status == 1:
        #     status = [1,0,0]
        # elif status == 2 :
        #     status = [0,1,0]
        # elif status == 3 :
        #     status = [0,0,1]
        outputY.append(status)


# print("prepare data complete")
# print("---")
outputY = np_utils.to_categorical(outputY, 3)
inputX = np.array(inputX)
#print(np.array(inputX).shape)

model.fit(inputX, outputY, 
          batch_size=5000, epochs=100, verbose=1)

model.save_weights("model2.h5")
model.save_weights("model2_bk.h5")
# for step in range(10001):
#     cost = model.train_on_batch(inputX, outputY)
#     if step % 100 == 0:
#         print('train cost: ', cost)
#         model.save_weights("model2.h5")
#         model.save_weights("model2_bk.h5")

    