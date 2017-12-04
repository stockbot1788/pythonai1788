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
from keras.layers import Input,Dropout
from keras import optimizers
from os import path
print("preparing model")


ls1Ip = Input(shape=(30,5))
ls11 = LSTM(25,dropout=0.5, recurrent_dropout=0.3)(ls1Ip)
ls12 = Dense(10, activation='sigmoid')(ls11)
ls12d = Dropout(.3)(ls12)
ls13 = Dense(5, activation='tanh')(ls12d)

ls1IpR = Input(shape=(2,))
ls1IpR2 = Dense(5, activation='tanh')(ls1IpR)

M1 = concatenate([ls13, ls1IpR2])


ls2Ip2 = Input(shape=(30,4))
ls21 = LSTM(25,dropout=0.5, recurrent_dropout=0.3)(ls2Ip2)
ls22 = Dense(20, activation='sigmoid',dropout=0.3)(ls21)
ls23 = Dense(10, activation='tanh',dropout=0.3)(ls22)
ls24 = Dense(5, activation='tanh',dropout=0.3)(ls23)

ls2IpR = Input(shape=(2,))
ls2IpR2 = Dense(5, activation='tanh')(ls2IpR)

M2 = concatenate([ls24, ls2IpR2])

ls3Ip3 = Input(shape=(1,))
ls31 = Dense(5, activation='tanh')(ls3Ip3)

merge_one = concatenate([M1, M2, ls31])

output = Dense(20, activation='tanh')(merge_one)
output1 = Dense(5, activation='tanh',dropout=0.3)(output)
output2 = Dense(1, activation='sigmoid',dropout=0.3)(output1)
model = Model(inputs=[ls1Ip,ls1IpR,ls2Ip2,ls2IpR,ls3Ip3], outputs=output2)

#model = Model(inputs=[ls1],outputs=)
sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',  optimizer=sgd ,metrics=['accuracy'])
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

# _path = "lstm.h5"
# if _path and path.isfile(_path):
#     try:
#         print("try load weight")
#         model.load_weights(_path)
#         print("load weight success")
#     except Exception as ex:
#         print("error",ex)
#         sys.exit("Error message")

#print(inputX.shape)
# for step in range(30):
#     cost = model.train_on_batch([inputX,inputXStr,inputX2,inputX2Str,Position], outputY) 
#     print (cost)
model.fit([inputX,inputXStr,inputX2,inputX2Str,Position], outputY, batch_size=300, epochs=800, verbose=1)
model.save_weights("lstm2.h5")

# for i in range(200):
#     model.fit(np.array(inputX), outputY, batch_size=5000, epochs=100, verbose=1)
#     model.save_weights("model233.h5")
#     model.save_weights("model233_bk.h5")

