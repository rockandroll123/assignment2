# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:37:28 2019

@author: autpx
"""

from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras import models
from keras import Model
from keras import layers
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

#load resnet50 without top layers
model_resn = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=1000)
print(len(model_resn.layers)) 
model_resn.layers
model_resn.summary()

for layer in model_resn.layers[:-7]:
    layer.trainable = False
for layer in model_resn.layers:
    print(layer, layer.trainable)

#data prep
batch_size = 4
num_classes = 10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
'''
datagen = ImageDataGenerator(rescale=1. / 255,
                         horizontal_flip=True,
                         rotation_range=360)
'''

## augment training data 
## use generator
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=False)


top_model = Sequential()
top_model.add(model_resn)
top_model.add(Flatten())
top_model.add(Dense(512, activation='relu'))#
top_model.add(Dropout(0.5))#
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

top_model.summary()

top_model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

top_model.fit(x_train, y_train, epochs=50, batch_size = 1024)


def show_r(model, testx, testlab):
    
    test_loss, test_acc = model.evaluate(testx, testlab)
    print('test_acc:', test_acc)

    history = model.history
    #history_dict = history.history

    acc = history.history['acc']
    print('train acc:', acc[-1])
    #val_acc = history.history['val_acc']
    loss = history.history['loss']
    #val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    #plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'go', label='Training acc')
    # b is for "solid blue line"
    #plt.plot(epochs, val_acc, 'g', label='Validation loss')
    plt.title('Training acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()

    plt.show()

show_r(top_model, x_test, y_test)



