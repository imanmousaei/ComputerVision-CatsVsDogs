import os
import random
import glob
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from keras.applications.vgg16 import VGG16

'''
os.chdir('../res/dogs-vs-cats/train')  # changes directory

os.makedirs('train/cats')
os.makedirs('train/dogs')
os.makedirs('validation/cats')
os.makedirs('validation/dogs')
os.makedirs('test/cats')
os.makedirs('test/dogs')


for image in random.sample(glob.glob('cat*'), 500):  # moves 500 random cat images to train/cats
    shutil.move(image, '../train/cat')
for image in random.sample(glob.glob('dog*'), 500):
    shutil.move(image, '../train/dog')
for image in random.sample(glob.glob('cat*'), 100):
    shutil.move(image, '../validation/cat')
for image in random.sample(glob.glob('dog*'), 100):
    shutil.move(image, '../validation/dog')
for image in random.sample(glob.glob('cat*'), 50):
    shutil.move(image, '../test/cat')
for image in random.sample(glob.glob('dog*'), 50):
    shutil.move(image, '../test/dog')
'''

train_path = '../res/dogs-vs-cats/train'
validation_path = '../res/dogs-vs-cats/validation'
test_path = '../res/dogs-vs-cats/test'

# target_size : for normalizing all images to 224x224
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
validation_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=validation_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10,
                         shuffle=False)

# probability:[cat,dog] i.e. [1. 0.]=cat, [0. 1.]=dog


vgg16_model = VGG16()  # importing a pre-trained CNN
# vgg16_model.summary()

CNN_model = Sequential()

for layer in vgg16_model.layers[:-1]:
    CNN_model.add(layer)

for layer in CNN_model.layers:
    layer.trainable = False
CNN_model.add(Dense(units=2, activation='softmax'))

CNN_model.summary()

# or binary_crossentropy(one node in output layer e.g. 0:cat,1:dog ) & last activation='sigmoid'
CNN_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

CNN_model.fit(
    x=train_batches,  # no need for y, because the generator contains the labels
    validation_data=validation_batches,
    epochs=5,
    verbose=2
)
print('training finished')
