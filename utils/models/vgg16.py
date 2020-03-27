import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np


def vgg16(num_of_classes):
    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv1'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv1'))
    model.add(Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv1'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv2'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv1'))
    model.add(Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv2'))
    model.add(Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1'))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (3, 3),activation='relu',padding='same', name='block5_conv2'))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv3'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Head section
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax', name='predictions'))