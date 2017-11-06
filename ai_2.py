import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from env import MarketEnv
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D
from keras.layers import LSTM, Dropout, Activation, Convolution2D,Convolution1D, MaxPooling2D, Flatten,GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.utils import np_utils

env = MarketEnv("data/20150917.txt")

epoch = 1000000
epsilon = .5

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    
s = (200,5)
Y = np.ones(s)

X_train, Y_train = X[:160], Y[:160] 
model = Sequential()
model.add(Dense(units=5, input_dim=1)) 
model.add(Activation('relu'))

adam = Adam(lr=1e-4)
model.compile(loss='mse', optimizer='sgd')
#model.compile(optimizer=adam,              
              #loss='categorical_crossentropy',
              #metrics=['accuracy'])

learn = True
while learn:
    cost = model.train_on_batch(X_train, Y_train)
    W, b = model.layers[0].get_weights()
    print('Weights=', W, '\nbiases=', b)