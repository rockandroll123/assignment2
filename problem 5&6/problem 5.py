#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:08:17 2019

@author: autumnbreed
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
model1 = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=1000)
print(len(model1.layers)) 
model1.summary()
model1.layers
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


## pass Training_data through resnet50 conv layers and get bottleneck_features restored
#bottleneck_features_train = model1.predict_generator(train_generator, 50000)

bottleneck_features_train = []
new_y = []

def add_to_bottle_f(fx):
    """Add conv training data to a list for each batch"""
    bottleneck_features_train.append(fx)
def add_to_label(ya):
    """Add label to a list for each batch"""
    new_y.append(ya)
    
for index,(x,y) in enumerate(train_generator):
    if index == 50000:
        break
    fx = model1.predict(x)[:,0,0,:]
    np.apply_along_axis(add_to_bottle_f, axis=1, arr=fx)
    np.apply_along_axis(add_to_label, axis=1, arr=y)

train_f = np.array(bottleneck_features_train)
train_t = np.array(new_y)
train_t = keras.utils.to_categorical(train_t, 10)
#train_f = model1.predict_generator(train_generator, 50000)
np.save("train_f.npy",train_f)
np.save("train_t.npy",train_t)

train_f = np.load("train_f.npy")
train_t = np.load("train_t.npy")


## pass Test_data through resnet50 conv layers and get bottleneck_features restored
test_f = model1.predict(x_test)[:,0,0,:]
np.save("test_f.npy",test_f)
test_f = np.load("test_f.npy")

# convert label to one-hot
y_test = keras.utils.to_categorical(y_test, 10)


# build MLP model 

def para_test(bsize, nlay, deep, lr, ActFun, DropRate, ep, traindata, trainy):
    model = models.Sequential()
    model.add(layers.Dense(nlay, activation=ActFun, input_shape=(2048,)))
    model.add(Dropout(DropRate))
    if deep:
        model.add(layers.Dense(nlay, activation=ActFun))
        model.add(Dropout(DropRate))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    sgd = optimizers.SGD(lr=lr)
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(traindata, trainy, epochs=ep, batch_size = bsize)
    return (model)

mb = para_test(bsize=8192, nlay=512, deep=True, lr=0.05, ActFun='relu', DropRate=0.05, ep=100, traindata = train_f, trainy = train_t)

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

show_r(mb, test_f, y_test)
###############################################################################
# visualize
test_loss, test_acc = mb.evaluate(test_f, y_test)
print('test_acc:', test_acc)

history = mb.history
acc = history.history['acc']
print('train acc:', acc[-1])

import seaborn as sns
sns.set()
ax = sns.heatmap(test_f[0:2000,:])
plt.show()
###############################################################################

# try simple network
# keep the 49 layer of resnet

intermediate_layer_model = Model(inputs=model1.input, output=model1.get_layer(None,-8).output)

#Add the Maxpooling and flatten layers
model_post = models.Sequential()
model_post.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None, input_shape=(4,4,512)))
model_post.add(Flatten())

x1, y1 = train_generator.next()
x1f = intermediate_layer_model.predict(x1)

x1fin = model_post.predict(x1f)

bottleneck_features_train = []
new_y = []

def add_to_bottle_f(fx):
    """Add conv training data to a list for each batch"""
    bottleneck_features_train.append(fx)
def add_to_label(ya):
    """Add label to a list for each batch"""
    new_y.append(ya)
    
for index,(x,y) in enumerate(train_generator):
    if index == 50000:
        break
    fx = intermediate_layer_model.predict(x)
    fx_post = model_post.predict(fx)
    np.apply_along_axis(add_to_bottle_f, axis=1, arr=fx_post)
    np.apply_along_axis(add_to_label, axis=1, arr=y)

train_f2 = np.array(bottleneck_features_train)
train_t2 = np.array(new_y)
train_t2 = keras.utils.to_categorical(train_t2, 10)

np.save("train_f2.npy",train_f2)
np.save("train_t2.npy",train_t2)

train_f2 = np.load("train_f2.npy")
train_t2 = np.load("train_t2.npy")

## pass Test_data through resnet50 conv layers and get bottleneck_features restored
test_f = intermediate_layer_model.predict(x_test)
test_f_post = model_post.predict(test_f)
np.save("test_f2.npy",test_f_post)
test_f_post = np.load("test_f2.npy")

model_C10 = para_test(bsize=4096, nlay=128, deep=True, lr=0.05, ActFun='sigmoid', DropRate=0.5, ep=100, traindata= train_f2, trainy = train_t2)

show_r(model_C10, test_f_post, y_test)

history = model_C10.history
acc = history.history['acc']
print('train acc:', acc[-1])




