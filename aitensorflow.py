import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
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
from keras.layers import Input,Dropout,Conv2D,MaxPooling2D,Flatten
from keras import optimizers
from os import path

print("preparing model")



ls1Ip = Input(shape=(30, 5, 1))
ls11 = Conv2D(64, (1, 1), padding='same', activation='relu')(ls1Ip)
ls12 = Conv2D(64, (2, 2), padding='same', activation='relu')(ls11)
ls13 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(ls12)
out = Flatten()(ls13)
output1 = Dense(10, activation='tanh')(out)
output2 = Dense(5, activation='sigmoid')(output1)


ls1IpR = Input(shape=(2,))
ls1IpR2 = Dense(5, activation='tanh')(ls1IpR)

M1 = concatenate([output2, ls1IpR2])


ls2Ip = Input(shape=(30, 4, 1))
ls21 = Conv2D(64, (1, 1), padding='same', activation='relu')(ls2Ip)
ls22 = Conv2D(64, (2, 2), padding='same', activation='relu')(ls21)
ls23 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(ls22)
out2 = Flatten()(ls23)
output21 = Dense(10, activation='tanh')(out2)
output22 = Dense(5, activation='sigmoid')(output21)

ls2IpR = Input(shape=(2,))
ls2IpR2 = Dense(5, activation='tanh')(ls2IpR)

M2 = concatenate([output22, ls2IpR2])

ls3Ip3 = Input(shape=(1,))
ls31 = Dense(5, activation='tanh')(ls3Ip3)

merge_one = concatenate([M1, M2, ls31])

output = Dense(20, activation='tanh')(merge_one)
output1 = Dense(5, activation='tanh')(output)
output2 = Dense(1, activation='sigmoid')(output1)
model = Model(inputs=[ls1Ip,ls1IpR,ls2Ip,ls2IpR,ls3Ip3], outputs=output2)

sgd = optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
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

_path = "lstm.h5"
if _path and path.isfile(_path):
    try:
        print("try load weight")
        model.load_weights(_path)
        print("load weight success")
    except Exception as ex:
        print("error",ex)
        sys.exit("Error message")


model.fit([inputX,inputXStr,inputX2,inputX2Str,Position], outputY, batch_size=600, epochs=800, verbose=1)
model.save_weights("lstm2.h5")
