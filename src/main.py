import os
import random
import glob
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

train_path = 'res/dogs-vs-cats/train'
validation_path = 'res/dogs-vs-cats/validation'
test_path = 'res/dogs-vs-cats/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
validation_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=validation_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10, shuffle=False)
