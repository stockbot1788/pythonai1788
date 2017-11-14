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
from keras.layers import merge,Conv2D, concatenate,Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
from keras.layers.advanced_activations import LeakyReLU
import sys
from keras.optimizers import RMSprop
print("preparing model")

B = Input(shape=[10, 4, 1])
h = Conv2D(2048, (5, 1), padding = 'same')(B)
h = LeakyReLU(0.001)(h)

h = Flatten()(h)
h = Dense(512)(h)
h = LeakyReLU(0.001)(h)
b = Dense(3, activation = "relu")(h)

inputs = [B]
merges = [b]

S = Input(shape=[10, 5, 1])
inputs.append(S)
h = Conv2D(2048, (3, 1), padding = 'same')(S)
h = LeakyReLU(0.001)(h)
h = Conv2D(2048, (5, 1), padding = 'same')(S)
h = LeakyReLU(0.001)(h)
# h = Conv2D(2048, (10, 1), padding = 'same')(S)
# h = LeakyReLU(0.001)(h)
# h = Conv2D(2048, (20, 1), padding = 'same')(S)
# h = LeakyReLU(0.001)(h)
# h = Conv2D(2048, (40, 1), padding = 'same')(S)
# h = LeakyReLU(0.001)(h)


h = Flatten()(h)
h = Dense(512)(h)
h = LeakyReLU(0.001)(h)
merges.append(h)

h = Conv2D(2048, (10, 1), padding = 'same')(S)
h = LeakyReLU(0.001)(h)

h = Flatten()(h)
h = Dense(512)(h)
h = LeakyReLU(0.001)(h)
merges.append(h)


m = concatenate(merges, axis = 1)
m = Dense(1024)(m)
m = LeakyReLU(0.001)(m)
m = Dense(512)(m)
m = LeakyReLU(0.001)(m)
m = Dense(256)(m)
m = LeakyReLU(0.001)(m)
V = Dense(3, activation = 'softmax')(m)
model = Model(inputs = inputs, outputs = V)
rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.summary()

# _path = "model233.h5"
# if _path and path.isfile(_path):
#     try:
#        print("try load weight")
#        model.load_weights(_path)
#        print("load weight success")
#     except Exception as ex:
#        print("error",ex)
#        sys.exit("Error message")

print("preparing data")
PrepareData = DataFormator(env.data)
inputX ,outputY= PrepareData.formatData()

#0. long earn 
#1. nothing 
#2. short earn 
#print(outputY) 
outputY = np_utils.to_categorical(outputY, 3)
#print(outputY)

#[1,0,0] long earn 
#[0,1,0] nothing 
#[0,0,1] short earn 

#print(inputX.shape)
model.fit(inputX, outputY, batch_size=5000, epochs=100, verbose=1)


# for i in range(200):
#     model.fit(np.array(inputX), outputY, batch_size=5000, epochs=100, verbose=1)
#     model.save_weights("model233.h5")
#     model.save_weights("model233_bk.h5")








# input_layer = Input(shape=(90, ))
# fc = Dense(100, activation='relu')(input_layer)
# fc = Dense(300, activation='relu')(fc)
# fc = Dense(200, activation='relu')(fc)
# fc = Dense(100, activation='relu')(fc)
# fc = Dense(45, activation='relu')(fc)
# fc = Dense(16, activation='relu')(fc)
# fc = Dense(10, activation='relu')(fc)
# pred = Dense(3, name='pred', activation='softmax')(fc)
# model = Model(inputs=[input_layer], outputs=[pred], name='fc')