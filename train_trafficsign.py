import Augmentor

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers.core import Dense, Flatten, Dropout, Lambda
import pickle
import numpy as np
from keras.utils import to_categorical
import keras
from keras.layers.convolutional import MaxPooling2D

import numpy as np

from PIL import Image

np.random.seed(315)



training_file = 'data/train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

train_x, train_y = train['features'], train['labels']

n_train = len(train_y)
n_classes = len(np.unique(train_y))

np.random.seed(seed=315)

percentage = 0.8



training_size = int(percentage*n_train)
mask=np.random.permutation(np.arange(n_train))[:training_size]


x_train, y_train = train_x[mask], train_y[mask]
val_x, val_y = np.delete(train_x, mask,0), np.delete(train_y, mask,0)

y_train = Augmentor.Pipeline.categorical_labels(y_train)
y_val = Augmentor.Pipeline.categorical_labels(val_y)



p = Augmentor.Pipeline()

p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.status()

num_classes = 43
input_shape = (32, 32, 3)

model = Sequential()

model.add(Conv2D(3, kernel_size=(1, 1),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.save('TrafficIdenfier_2.h5')

batch_size = 128

g = p.keras_generator_from_array(x_train, y_train, batch_size=500)
g_val = p.keras_generator_from_array(val_x, y_val, batch_size=500)

model.fit_generator(g, steps_per_epoch=len(x_train)/batch_size, epochs=15, verbose=2,validation_data=g_val,validation_steps=len(val_x)/batch_size)

model.save_weights('TrafficIdenfitifier_usingAugmentor_2.h5')
