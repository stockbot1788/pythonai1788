import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from env import MarketEnv
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


env = MarketEnv("data")
epoch = 1000000
epsilon = 0.5
batch_size = 30



Neural = NerualModel()
model = Neural.getModel()


rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
exp_replay = ExperienceReplay()


for e in range(epoch):
    #loss = 0.
    game_over = False
    input_t = env.reset()
    print("run status {}",format(e))
    while not game_over:
        input_prev = input_t
        #print(input_prev)
        isRandom = False
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
            isRandom = True
        else:
            q = model.predict(np.array([input_prev]))
            action = np.argmax(q[0])

        input_t, reward, game_over, info = env.step(action)
        if game_over == True:
           print("total reward : " ,env.cur_reward)
        exp_replay.remember([input_prev, action, reward, input_t], game_over)
        batch = exp_replay.get_batch(model, batch_size=batch_size)
        loss = model.train_on_batch(batch[0], batch[1])
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        model.save_weights("model_bk.h5")


# input_t = env.reset()
# q = model.predict(np.array([input_t]))
# action = np.argmax(q[0])
# print("action {}",format(action))