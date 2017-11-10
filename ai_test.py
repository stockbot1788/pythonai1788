import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from env_test import MarketEnv
# import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D
from keras.layers import LSTM, Dropout, Activation, Convolution2D,Convolution1D, MaxPooling2D, Flatten,GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
from expMetrix import ExperienceReplay
from model import NerualModel
from keras.optimizers import RMSprop
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.models import Model
from keras.optimizers import RMSprop
import sys

env = MarketEnv("data")
epoch = 1000000
epsilon = 0.0
batch_size = 30



input_layer = Input(shape=(42, ))
fc = Dense(32, activation='relu')(input_layer)
fc = Dense(16, activation='relu')(fc)
fc = Dense(8, activation='relu')(fc)
pred = Dense(3, name='pred', activation='softmax')(fc)
model = Model(inputs=[input_layer], outputs=[pred], name='fc')

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
W ,b= model.layers[1].get_weights()
#print(W)
print("--------")
try:
    print("try load weight")
    model.load_weights("model2.h5")
    print("load weight success")
except Exception as ex:
    print("error",ex)
    sys.exit("Error message")

W ,b= model.layers[1].get_weights()
#print(W)

# exp_replay = ExperienceReplay()


for e in range(20):
    #loss = 0.
    game_over = False
    cur_reward = 0
    winGame = 0
    lostGame = 0
    input_t = env.reset()
    print("run status {} ",format(e))
    while not game_over:
        #print(env.stepNumber)
        env.defineState()
        input_v = env.state
        #print(input_v)
        q = model.predict(np.array([input_v]))
        action = np.argmax(q[0])
        #print(action)
        if action == 1:
            #long
            buyPrice = env.StockDataSingleDay.m_data[env.stepNumber]._high
            status = 0
            for i in range(1,15):
                tmpPriceHigh = env.StockDataSingleDay.m_data[env.stepNumber+i]._high
                tmpPriceLow = env.StockDataSingleDay.m_data[env.stepNumber+i]._low
                earn = float(tmpPriceHigh) - float(buyPrice)
                loss = float(tmpPriceLow) - float(buyPrice)
                if loss < env.biggestLost and status == 0:
                    status = 1
                    break
                if earn > env.biggestEarn and status == 0:
                    status = 2
                    break
            if status == 1:
                cur_reward = cur_reward - 25
                lostGame = lostGame + 1
            elif status == 2:
                winGame = winGame + 1
                cur_reward = cur_reward + 25

        elif action ==2:
            #short
            buyPrice = env.StockDataSingleDay.m_data[env.stepNumber]._low
            status = 0
            for i in range(1,15):
                tmpPriceHigh = env.StockDataSingleDay.m_data[env.stepNumber+i]._high
                tmpPriceLow = env.StockDataSingleDay.m_data[env.stepNumber+i]._low
                earn = float(buyPrice) - float(tmpPriceLow)
                loss = float(buyPrice) - float(tmpPriceHigh)
                if loss < env.biggestLost and status == 0:
                    status = 1
                    break
                if earn > env.biggestEarn and status == 0:
                    status = 2
                    break
            if status == 1:
                lostGame = lostGame + 1
                cur_reward = cur_reward - 25
            elif status == 2:
                winGame = winGame + 1
                cur_reward = cur_reward + 25

        env.stepNumber = env.stepNumber + 1
        if env.stepNumber > len(env.StockDataSingleDay.m_data)-1-15:
                game_over = True
                print("win Game",winGame)
                print("Lost Game",lostGame)
                print("----")

