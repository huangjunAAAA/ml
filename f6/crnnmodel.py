import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class ModelConfig:
    width = 0
    height = 0
    output_cat = 52


dr = 0.25


def make(mc):
    ips = None

    if K.image_data_format() == 'channels_last':
        ips = (mc.height, mc.width, 3)
    else:
        ips = (3, mc.height, mc.width)

    inputs = tf.keras.Input(shape=ips)
    cov1 = tf.keras.layers.Conv2D(filters=104, kernel_size=(3, 3), activation='relu', padding="same")(inputs)
    m1 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(cov1)
    bn1 = tf.keras.layers.BatchNormalization()(m1)

    cov2 = tf.keras.layers.Conv2D(filters=156, kernel_size=(3, 3), activation='relu', padding="same")(bn1)
    m2 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(cov2)
    bn2 = tf.keras.layers.BatchNormalization()(m2)

    cov3 = tf.keras.layers.Conv2D(filters=208, kernel_size=(3, 3), activation='relu', padding="same")(bn2)
    m3 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(cov3)
    bn3 = tf.keras.layers.BatchNormalization()(m3)

    cov4 = tf.keras.layers.Conv2D(filters=260, kernel_size=(3, 3), activation='relu', padding="same")(bn3)
    m4 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(cov4)
    bn4 = tf.keras.layers.BatchNormalization()(m4)

    fms = tf.reshape(bn4, [-1, mc.width, 260])

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=260, return_sequences=True, dropout=0.5))(fms)
    gap = tf.reduce_mean(lstm, axis=2)
    gap1 = tf.reshape(gap, [-1, mc.width, 1])

    outputs = tf.keras.layers.Dense(mc.output_cat, activation="softmax")(gap1)
    print(outputs.shape)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_layer_output(model, x, index=-1):
    """
    get the computing result output of any layer you want, default the last layer.
    :param model: primary model
    :param x: input of primary model( x of model.predict([x])[0])
    :param index: index of target layer, i.e., layer[23]
    :return: result
    """
    layer = K.function([model.input], [model.layers[index].output])
    return layer([x])[0]


y1 = 1


def printx(x):
    # global y1
    # y1 = x * x
    # print("previous x")
    # tf.print(y1)
    #
    # print("now x")
    # tf.print(x, 'stdout')
    #
    # y1 += x
    return x
