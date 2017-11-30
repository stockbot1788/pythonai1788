import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from env import MarketEnv
from dataformator import DataFormator
env = MarketEnv("data")
import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import concatenate
from numpy import array
import sys
from keras.optimizers import RMSprop
from keras.layers import Input
from os import path
print("preparing model")


ls1Ip = Input(shape=(15,5))
ls11 = LSTM(5)(ls1Ip)
ls12 = Dense(2, activation='tanh')(ls11)

ls1IpR = Input(shape=(2,))
ls1IpR2 = Dense(2, activation='tanh')(ls1IpR)

M1 = concatenate([ls12, ls1IpR2])


ls2Ip2 = Input(shape=(15,4))
ls21 = LSTM(5)(ls2Ip2)
ls22 = Dense(2, activation='tanh')(ls21)

ls2IpR = Input(shape=(2,))
ls2IpR2 = Dense(2, activation='tanh')(ls2IpR)

M2 = concatenate([ls22, ls2IpR2])

ls3Ip3 = Input(shape=(1,))
ls31 = Dense(2, activation='tanh')(ls3Ip3)

merge_one = concatenate([M1, M2, ls31])

output = Dense(5, activation='tanh')(merge_one)
output2 = Dense(1, activation='sigmoid')(output)
model = Model(inputs=[ls1Ip,ls1IpR,ls2Ip2,ls2IpR,ls3Ip3], outputs=output2)

#model = Model(inputs=[ls1],outputs=)
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

print("preparing data")
PrepareData = DataFormator(env.data)
inputX,inputXStr,inputX2,inputX2Str,Position,outputY = PrepareData.formatData()


inputX = array(inputX)
inputXStr = array(inputXStr)
inputX2 = array(inputX2)
inputX2Str = array(inputX2Str)
Position = array(Position)
outputY = array(outputY)


_path = "lstm.h5"
if _path and path.isfile(_path):
    try:
        print("try load weight")
        model.load_weights(_path)
        print("load weight success")
    except Exception as ex:
        print("error",ex)
        sys.exit("Error message")

print("predicting")
dataA = model.predict([inputX,inputXStr,inputX2,inputX2Str,Position])
guess = 0
guess1 = 0
for i in range(0,len(dataA)):
    print(dataA[i][0])
    if dataA[i][0] > 0.6 and outputY[i]==1:
        print(dataA[i][0])
        guess = guess + 1
    if dataA[i][0] > 0.6:
        guess1 = guess1 + 1

print(guess)
print(guess1)
print("complete")

#print(inputX.shape)
#model.fit([inputX,inputXStr,inputX2,inputX2Str,Position], outputY, batch_size=50, epochs=100, verbose=1)
#model.save_weights("lstm.h5")

# for i in range(200):
#     model.fit(np.array(inputX), outputY, batch_size=5000, epochs=100, verbose=1)
#     model.save_weights("model233.h5")
#     model.save_weights("model233_bk.h5")

