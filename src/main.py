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

images, labels = next(train_batches)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_images(images)  # cuz of the preprocessing_function vgg16, colors are a little fucked up
print(labels)  # probability:[cat,dog] i.e. [1. 0.]=cat, [0. 1.]=dog

CNN_model = Sequential([
    # padding='same' : no padding , input shape 3 : color channels (RGB in our case)
    # 2nd layer in CNN(ergo needs shape of input layer) :
    Conv2D(filters=32, kernel_size=(3, 3), activation='sigmoid', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dropout(0.5),  # dropout to avoid overfitting
    Dense(units=2, activation='softmax')
])

# or binary_crossentropy(one node in output layer e.g. 0:cat,1:dog ) & last activation='sigmoid'
CNN_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

CNN_model.fit(
    x=train_batches,  # no need for y, because the generator contains the labels
    validation_data=validation_batches,
    epochs=10,
    verbose=2
)

print('training finished')

