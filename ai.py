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
from expMetrix import ExperienceReplay
from model import NerualModel
from keras.optimizers import RMSprop
env = MarketEnv("data/20150917.txt")

epoch = 1000000
epsilon = .5
batch_size = 128

Neural = NerualModel()
model = Neural.getModel()


rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
exp_replay = ExperienceReplay()


input_t = env.reset()
action = np.random.randint(0, env.action_space.n, size=1)[0]
print("-----")
#print(len(input_t.reshape(-1)))

print("-----")
for e in range(epoch):
    #loss = 0.
    game_over = False
    input_t = env.reset()
    model.save_weights("modelrnn.h5", overwrite=True)
    while not game_over:
        input_prev = input_t
        isRandom = False
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
            isRandom = True
        else:
            data = input_t.reshape(-1)
            q = model.predict(np.array([data]))
            action = np.argmax(q[0])
        input_t, reward, game_over, info = env.step(action)
        if isRandom != True:
            print("action {}  reward {}".format(action,reward))
        exp_replay.remember([input_prev, action, reward, input_t], game_over)
        batch = exp_replay.get_batch(model, batch_size=batch_size)
        loss = model.train_on_batch(batch[0], batch[1])
