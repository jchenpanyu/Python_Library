#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 21:39:07 2018

Demo of convnet

@author: root
"""

from keras import layers
from keras import models

# build DCNN model via keras
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

# print DCNN model structure
model.summary()

# conect final concnet to full-connect layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# print DCNN model structure
model.summary()

# train the DCNN model with MNIST data
from keras.utils import to_categorical
import numpy as np

# load MINST data
minst_npz = "/mnt/d/Document/linux/python/data/mnist.npz"
minst_data = np.load(minst_npz)
train_images = minst_data["x_train"]
train_labels = minst_data["y_train"]
test_images  = minst_data["x_test"]
test_labels  = minst_data["y_test"]
# or
"""
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
"""

# reshape the data
train_images = train_images.reshape((train_images.shape[0],
                                     train_images.shape[1],
                                     train_images.shape[2], 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((test_images.shape[0],
                                   test_images.shape[1],
                                   test_images.shape[2], 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)

# set up model training flow
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
"""
Epoch 1/5
60000/60000 [==============================] - 64s 1ms/step - loss: 0.1697 - acc: 0.9473
Epoch 2/5
60000/60000 [==============================] - 65s 1ms/step - loss: 0.0483 - acc: 0.9850
Epoch 3/5
60000/60000 [==============================] - 66s 1ms/step - loss: 0.0341 - acc: 0.9892
Epoch 4/5
60000/60000 [==============================] - 66s 1ms/step - loss: 0.0266 - acc: 0.9920
Epoch 5/5
60000/60000 [==============================] - 64s 1ms/step - loss: 0.0213 - acc: 0.9939
"""

# evaluate the DCNN model with test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(test_loss, test_accuracy)
"""
10000/10000 [==============================] - 3s 250us/step
0.04359047726951467 0.9882
"""