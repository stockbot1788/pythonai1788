import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D
from keras.layers import LSTM, Dropout, Activation, Convolution2D,Convolution1D, MaxPooling2D, Flatten,GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
from os import path

class NerualModel(object):
    def __init__(self):
        print("init model")

    def getModel(self,_path="model.h5"):
        model = self.buildModel()
        if _path and path.isfile(_path):
            try:
                print("try load weight")
                model.load_weights(_path)
                print("load weight success")
            except Exception:
                print("error")
        return model

    def buildModel(self):
        model = Sequential()
        model.add(Conv2D(
            64, (3, 4),
            input_shape=(1, 10, 4),
            padding='same',
        ))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',    # Padding method
            data_format='channels_first',
        ))

        model.add(Flatten())
        model.add(Dense(10))
        model.add(Activation('relu'))


        model.add(Dense(500))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))


        model.add(Dense(1000))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        # model.add(Dense(2000))
        # model.add(Dropout(0.2))
        # model.add(Activation('relu'))


        # model.add(Dense(5000))
        # model.add(Dropout(0.2))
        # model.add(Activation('relu'))



        model.add(Dense(5))
        model.add(Activation('softmax'))
        return model