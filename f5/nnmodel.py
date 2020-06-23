import tensorflow as tf
from keras import backend as K


class ModelConfig:
    width = 0
    height = 0
    output_cat = 52


dr = 0.25


def make(mc):
    ips = None

    if K.image_data_format() == 'channels_last':
        ips = (mc.width, mc.height, 3)
    else:
        ips = (3, mc.width, mc.height)

    seq = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=ips, padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    seq.add(tf.keras.layers.Lambda(printx))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))

    seq.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))

    seq.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    seq.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))

    seq.add(tf.keras.layers.Flatten())
    seq.add(tf.keras.layers.Dense(256, activation='relu'))
    seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.Dropout(dr))
    seq.add(tf.keras.layers.Dense(mc.output_cat, activation='softmax'))

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
