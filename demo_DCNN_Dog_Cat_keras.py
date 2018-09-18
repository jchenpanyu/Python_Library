#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:28:24 2018

DCNN to classify cat vs dog

@author: root
"""

# organize data set
import numpy as np
import random
import os, shutil

dataset_dir = "/mnt/d/Document/linux/python/data/dogs_vs_cas/train"

random.seed(0)
train_index = np.random.choice(np.arange(12500),
                               size=1000, replace=False)
val_index   = np.random.choice(list(set(range(12500)) - set(train_index)),
                               size=500, replace=False)
test_index  = np.random.choice(list(set(range(12500)) - set(train_index) - set(val_index)),
                               size=500, replace=False)

sub_dataset_dir = "/mnt/d/Document/linux/python/data/dogs_vs_cas/sub_set"
os.mkdir(sub_dataset_dir)
train_dir = os.path.join(sub_dataset_dir, "train")
os.mkdir(train_dir)
validation_dir = os.path.join(sub_dataset_dir, "validation")
os.mkdir(validation_dir)
test_dir = os.path.join(sub_dataset_dir, "test")
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, "cats")
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, "dogs")
os.mkdir(train_dogs_dir)
val_cats_dir = os.path.join(validation_dir, "cats")
os.mkdir(val_cats_dir)
val_dogs_dir = os.path.join(validation_dir, "dogs")
os.mkdir(val_dogs_dir)
test_cats_dir = os.path.join(test_dir, "cats")
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, "dogs")
os.mkdir(test_dogs_dir)

file_names = ['cat.{}.jpg'.format(i) for i in train_index]
for f in file_names:
    src = os.path.join(dataset_dir, f)
    dst = os.path.join(train_cats_dir, f)
    shutil.copyfile(src, dst)

file_names = ['dog.{}.jpg'.format(i) for i in train_index]
for f in file_names:
    src = os.path.join(dataset_dir, f)
    dst = os.path.join(train_dogs_dir, f)
    shutil.copyfile(src, dst)

file_names = ['cat.{}.jpg'.format(i) for i in val_index]
for f in file_names:
    src = os.path.join(dataset_dir, f)
    dst = os.path.join(val_cats_dir, f)
    shutil.copyfile(src, dst)

file_names = ['dog.{}.jpg'.format(i) for i in val_index]
for f in file_names:
    src = os.path.join(dataset_dir, f)
    dst = os.path.join(val_dogs_dir, f)
    shutil.copyfile(src, dst)

file_names = ['cat.{}.jpg'.format(i) for i in test_index]
for f in file_names:
    src = os.path.join(dataset_dir, f)
    dst = os.path.join(test_cats_dir, f)
    shutil.copyfile(src, dst)

file_names = ['dog.{}.jpg'.format(i) for i in test_index]
for f in file_names:
    src = os.path.join(dataset_dir, f)
    dst = os.path.join(test_dogs_dir, f)
    shutil.copyfile(src, dst)


# construct DCNN model
from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1,   activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# data pre-process
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen   = ImageDataGenerator(rescale=1.0/255)

train_datagen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
val_datagen = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# model training with DataGenerator
history = model.fit_generator(
        train_datagen,
        steps_per_epoch=100,
        epochs=30,
        validation_data=val_datagen,
        validation_steps=50)


# save the model
model.save("cats_and_dogs_small_1.h5")

# plot the history
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.figure()
plt.plot(epochs, acc, 'bo', label="Training Acc")
plt.plot(epochs, val_acc, 'b', label="Validation Acc")
plt.title("Training and Validation Accurary")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


