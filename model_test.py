import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Flatten, BatchNormalization, Dropout, Dense, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import cv2
import numpy as np


height = 256
width = 256
channels = 3


inputs = Input(shape=(height, width, channels))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


x = Flatten()(x)

x1 = Dense(256, kernel_regularizer=l2(0.001))(x)
x1 = LeakyReLU()(x1)
x1 = Dropout(0.7)(x1)
x1 = BatchNormalization()(x1)

out_1 = Dense(15, activation="softmax", name='output_1')(x1)

x2 = Dense(256, kernel_regularizer=l2(0.001))(x)
x2 = LeakyReLU()(x2)
x2 = Dropout(0.7)(x2)
x2 = BatchNormalization()(x2)

x2 = Dense(32)(x2)
x2 = LeakyReLU()(x2)
x2 = Dropout(0.7)(x2)
x2 = BatchNormalization()(x2)


out_2 = Dense(3, activation="softmax", name='output_2')(x2)

model = Model(inputs=inputs, outputs=[out_1, out_2])

model.load_weights("weights.h5")

