from keras.layers import Dense,Input,Reshape,Flatten,Dropout
from keras.layers import BatchNormalization,Activation,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
from keras.models import Sequential,Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class WGAN():
    def __Init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3

        optimizer = Adam(0.0002,0.5)



    def generator(self,x):
        model = Sequential()

        model.add(Conv2D(32,kernel_size=3,padding="same")
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32, kernel_size=3, padding="same")
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(1, kernel_size=3, padding="same")
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.summary()

        x = Input(shape=(256,256,3))
        img = model(x)

        return Model(img)

    def discriminator(self):
        



    def train(self):

if __name__ == "__main__":
