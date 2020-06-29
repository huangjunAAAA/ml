import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class ModelConfig:
    width = 0
    height = 0
    output_cat = 52

def show_shape(y_true, y_pred):
    # print(y_true.shape)
    # print(y_pred.shape)
    return 0.1

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

    outputs = tf.keras.layers.Dense(mc.output_cat+1, activation='softmax')(gap1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def ctc_loss(y_true, y_pred):
    y_idxlst = tf.argmax(y_true, axis=2)
    tf.print('y_idxlst',y_idxlst)
    y_label_length = tf.reduce_sum(y_true, axis=[1, 2])
    label_length = tf.reshape(y_label_length, [y_label_length.shape[0], 1])
    input_length = tf.fill([label_length.shape[0], 1], y_pred.shape[1])
    return tf.keras.backend.ctc_batch_cost(y_true=y_idxlst, y_pred=y_pred, input_length=input_length, label_length=label_length)

def ctc_acc(y_true, y_pred):
    # batch_size = y_true.shape[0]
    # y_idxlst = tf.argmax(y_true, axis=2)
    # max_len = y_true.shape[1]
    # input_length = tf.fill([batch_size], y_pred.shape[1])
    # y_pred_decoded = tf.keras.backend.ctc_decode(y_pred, input_length,greedy=False)
    # print('y_pred_decoded')
    # print(y_pred_decoded)

    return 0.1

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
