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
print("preparing model")


ls1Ip = Input(shape=(15,5))
ls11 = LSTM(25)(ls1Ip)
ls11d = Dropout(.3)(ls11)
ls12 = Dense(10, activation='tanh')(ls11d)
ls13 = Dense(5, activation='tanh')(ls12)

ls1IpR = Input(shape=(2,))
ls1IpR2 = Dense(5, activation='tanh')(ls1IpR)

M1 = concatenate([ls13, ls1IpR2])


ls2Ip2 = Input(shape=(15,4))
ls21 = LSTM(25)(ls2Ip2)
ls21d = Dropout(.3)(ls21)
ls22 = Dense(10, activation='tanh')(ls21d)
ls23 = Dense(5, activation='tanh')(ls22)

ls2IpR = Input(shape=(2,))
ls2IpR2 = Dense(5, activation='tanh')(ls2IpR)

M2 = concatenate([ls23, ls2IpR2])

ls3Ip3 = Input(shape=(1,))
ls31 = Dense(5, activation='tanh')(ls3Ip3)

merge_one = concatenate([M1, M2, ls31])

output = Dense(10, activation='tanh')(merge_one)
outputd = Dropout(.3)(output)
output1 = Dense(5, activation='tanh')(outputd)
output2 = Dense(1, activation='sigmoid')(output1)
model = Model(inputs=[ls1Ip,ls1IpR,ls2Ip2,ls2IpR,ls3Ip3], outputs=output2)

#model = Model(inputs=[ls1],outputs=)
sgd = optimizers.SGD(lr=0.00001)
model.compile(loss='mean_squared_error',  optimizer=sgd)
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


#print(inputX.shape)
# for step in range(30):
#     cost = model.train_on_batch([inputX,inputXStr,inputX2,inputX2Str,Position], outputY) 
#     print (cost)
model.fit([inputX,inputXStr,inputX2,inputX2Str,Position], outputY, batch_size=300, epochs=400, verbose=1)
model.save_weights("lstm.h5")

# for i in range(200):
#     model.fit(np.array(inputX), outputY, batch_size=5000, epochs=100, verbose=1)
#     model.save_weights("model233.h5")
#     model.save_weights("model233_bk.h5")

