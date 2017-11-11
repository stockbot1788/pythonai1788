import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from env import MarketEnv
from dataformator import DataFormator
env = MarketEnv("data")
import tensorflow as tf
import keras
from keras.utils import np_utils

from keras.layers import Dense, Conv2D, Flatten, Input
from keras.models import Model
from keras.optimizers import RMSprop
from os import path
import sys
print("preparing model")
# input_layer = Input(shape=(10, 9, 1))
# cnn1 = Conv2D(10, kernel_size=(2,2), strides=(2,1), activation='relu')(input_layer)
# cnn2 = Conv2D(20, kernel_size=(2,2), strides=(2,1), activation='relu')(cnn1)
# flat = Flatten()(cnn2)
# fc1 = Dense(56, activation='relu')(flat)
# pred = Dense(3, name='pred', activation='softmax')(fc1)
# model = Model(inputs=[input_layer], outputs=[pred], name='fc')

input_layer = Input(shape=(90, ))
fc = Dense(100, activation='relu')(input_layer)
fc = Dense(300, activation='relu')(fc)
fc = Dense(200, activation='relu')(fc)
fc = Dense(100, activation='relu')(fc)
fc = Dense(45, activation='relu')(fc)
fc = Dense(16, activation='relu')(fc)
fc = Dense(10, activation='relu')(fc)
pred = Dense(3, name='pred', activation='softmax')(fc)
model = Model(inputs=[input_layer], outputs=[pred], name='fc')

rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

_path = "model233.h5"
if _path and path.isfile(_path):
    try:
       print("try load weight")
       model.load_weights(_path)
       print("load weight success")
    except Exception as ex:
       print("error",ex)
       sys.exit("Error message")

print("preparing data")
PrepareData = DataFormator(env.data)
inputX ,outputY= PrepareData.formatData()

#0. long earn 
#1. long earn 
#2. short earn 
#print(outputY) 
outputY = np_utils.to_categorical(outputY, 3)
#print(outputY)

#[1,0,0] long earn 
#[0,1,0] long earn 
#[0,0,1] short earn 

#model.fit(np.array(inputX), outputY, batch_size=5000, epochs=100, verbose=1)
for i in range(200):
    model.fit(np.array(inputX), outputY, batch_size=5000, epochs=100, verbose=1)
    model.save_weights("model233.h5")
    model.save_weights("model233_bk.h5")


    