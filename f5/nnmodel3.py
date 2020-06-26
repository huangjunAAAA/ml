import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class ModelConfig:
    width = 0
    height = 0
    output_cat = 52


dr = 0.25

def more_char_acc(y_true, y_pred):
    num_char = y_true.get_shape()[1]//26
    y2 = tf.reshape(y_true, [-1, num_char, 26])
    y2max = tf.math.argmax(y2, axis=2)
    y1 = tf.reshape(y_pred, [-1, num_char, 26])
    y1max = tf.math.argmax(y1, axis=2)
    r1 = tf.equal(y1max, y2max)
    r2 = tf.cast(r1, tf.float32)
    k = tf.reduce_mean(r2)
    return k

def make(mc):
    ips = None

    if K.image_data_format() == 'channels_last':
        ips = (mc.height, mc.width, 3)
    else:
        ips = (3, mc.height, mc.width)

    seq = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2D(filters=104, kernel_size=(3, 3), activation='relu', input_shape=ips, padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    seq.add(tf.keras.layers.Lambda(printx))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))

    seq.add(tf.keras.layers.Conv2D(filters=156, kernel_size=(3, 3), activation='relu', padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))

    seq.add(tf.keras.layers.Conv2D(filters=208, kernel_size=(3, 3), activation='relu', padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))

    seq.add(tf.keras.layers.Conv2D(filters=260, kernel_size=(3, 3), activation='relu', padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))


    seq.add(tf.keras.layers.GlobalAveragePooling2D())
    seq.add(tf.keras.layers.Dense(mc.output_cat, activation='sigmoid'))

    return seq


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
