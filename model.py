from keras.models import Sequential
from keras.layers import Dense,Conv2D
from keras.layers import LSTM, Dropout, Activation, Convolution2D,Convolution1D, MaxPooling2D, Flatten,GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
from os import path
import sys
from keras.models import model_from_json
from keras.models import Model
from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import add,concatenate
class NerualModel(object):
    def __init__(self):
        print("init model")

    def getModel(self,_path="model.h5",_modelpath="model.json"):
        model = self.buildModel()
        # if _modelpath and path.isfile(_modelpath):
        #     # load json and create model
        #     json_file = open(_modelpath, 'r')
        #     loaded_model_json = json_file.read()
        #     json_file.close()
        #     loaded_model = model_from_json(loaded_model_json)
        #     model = loaded_model
        #     print("get model from json")
        # else:
        #     print("get model from json fail")
        if _path and path.isfile(_path):
            try:
                print("try load weight")
                model.load_weights(_path)
                print("load weight success")
            except Exception as ex:
                print("error",ex)
                sys.exit("Error message")
        return model

    def buildModel(self):

        
        B = Input(shape = (3,))
        b = Dense(5, activation = "relu")(B)
        inputs = [B]
        merges = [b]
        S = Input(shape=[2, 60, 1])
        inputs.append(S)
        h = Conv2D(2048, (3, 1), padding="same")(S)
        h = LeakyReLU(0.001)(h)
        merges.append(h)
        m = concatenate(merges, axis = 1)
        m = Dense(1024)(m)
        m = LeakyReLU(0.001)(m)
        V = Dense(2, activation = 'softmax')(m)
        model = Model(input = inputs, output = V)
        model.summary()

        model = Sequential()
        model.add(Dense(70, input_shape=(42,), kernel_initializer="lecun_uniform"))
        model.add(Activation('relu'))
        
        model.add(Dense(28))
        model.add(Activation('relu'))


        model.add(Dense(12))
        model.add(Activation('relu'))

        # model.add(Dense(2000))
        # model.add(Dropout(0.2))
        # model.add(Activation('relu'))


        # model.add(Dense(5000))
        # model.add(Dropout(0.2))
        # model.add(Activation('relu'))


        model.add(Dense(3, kernel_initializer="lecun_uniform"))
        model.add(Activation('linear'))
        return model