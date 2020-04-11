import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import cv2

def x_ray_prediction(filepath):
    img = cv2.resize(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB), (256, 256))
    img = img / 255
    global sess1
    sess1 = tf.Session()
    keras.backend.set_session(sess1)
    global model

    model = keras.models.Sequential()
    model.add(keras.layers.Convolution2D(16, 3, input_shape=(256, 256, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(16, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))
    model.add(keras.layers.Convolution2D(32, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(32, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))
    model.add(keras.layers.Convolution2D(64, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(64, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))
    model.add(keras.layers.Convolution2D(128, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(128, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))

    # Classification layer
    model.add(keras.layers.Convolution2D(128, 4))

    ##average pooling
    model.add(keras.layers.Flatten())

    ##Dropout(0.3)
    model.add(keras.layers.Dropout(0.3))

    # output
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.load_weights('model.h5')

    # model = keras.models.load_model('dense.h5')

    global graph1
    graph1 = tf.get_default_graph()
    with graph1.as_default():
        keras.backend.set_session(sess1)
        y_p = model.predict(np.reshape(img, (1, 256, 256, 3)))
        y_p = np.around(y_p, decimals=2).T
        return y_p[3][0]