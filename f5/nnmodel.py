import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class ModelConfig:
    width = 0
    height = 0
    output_cat = 52


dr = 0.25

def more_char_acc(y_true, y_pred):
    y1 = tf.reshape(y_pred, [26, y_pred.get_shape()[1]//26])
    y1max = tf.math.argmax(y1, axis=0)
    y2 = tf.reshape(y_true, [26, y_true.get_shape()[1]//26])
    y2max = tf.math.argmax(y2, axis=0)
    r1 = tf.equal(y1max, y2max)
    k = tf.numpy_function(countTrue, [r1], tf.float32)
    return k

def countTrue(r1):
    return np.mean(r1)

def make(mc):
    ips = None

    if K.image_data_format() == 'channels_last':
        ips = (mc.height, mc.width, 3)
    else:
        ips = (3, mc.height, mc.width)

    seq = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=ips, padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    seq.add(tf.keras.layers.Lambda(printx))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))

    seq.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))

    seq.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))

    seq.add(tf.keras.layers.Flatten())
    seq.add(tf.keras.layers.Dense(1024, activation='relu'))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))
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
