import os
import random
import glob
import shutil

os.chdir('../res/dogs-vs-cats/train')  # changes directory


os.makedirs('train/cats')
os.makedirs('train/dogs')
os.makedirs('validation/cats')
os.makedirs('validation/dogs')
os.makedirs('test/cats')
os.makedirs('test/dogs')


for image in random.sample(glob.glob('cat*'), 500):  # moves 500 random cat images to train/cats
    shutil.move(image, '../train/cats')
for image in random.sample(glob.glob('dog*'), 500):
    shutil.move(image, '../train/dogs')
for image in random.sample(glob.glob('cat*'), 100):
    shutil.move(image, '../validation/cats')
for image in random.sample(glob.glob('dog*'), 100):
    shutil.move(image, '../validation/dogs')
for image in random.sample(glob.glob('cat*'), 50):
    shutil.move(image, '../test/cats')
for image in random.sample(glob.glob('dog*'), 50):
    shutil.move(image, '../test/dogs')

