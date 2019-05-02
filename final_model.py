import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv3D, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization

from generator_class import DataGenerator

#loads partition, labels dictionary
from build_dataset import *

#set parameters for Data Generator
params = {'dim': (64, 64, 64),
          'batch_size': 32,
          'n_classes': 7,
          'n_channels': 18,
          'shuffle': True}

# Generators
training_generator = DataGenerator('train', partition['train'], labels, **params)
validation_generator = DataGenerator('validation', partition['validation'], labels, **params)
test_generator = DataGenerator('test', partition['test'], labels, **params)

#build the model
model = Sequential()

#first main convolutional layer, implementing strides, batch norm between convolution and activation
model.add(Conv3D(256, kernel_size=(5, 5, 5), strides=(2, 2, 2), input_shape=(64, 64, 64, 18), data_format="channels_last"))
model.add(BatchNormalization())
model.add(Activation("relu"))

#second convolutional layer
model.add(Conv3D(128, kernel_size=(3, 3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))

#third convo layer
model.add(Conv3D(64, kernel_size=(3, 3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))


#dropout to reduce reliance on single features, 'spread the weights'
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=10,
                    verbose=2,
                   )
model.save('enzyme_cnn.h5')
