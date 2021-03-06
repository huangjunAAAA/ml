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
    b1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding="same")(inputs)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding="same")(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(b1)
    b1 = tf.keras.layers.Dropout(dr)(b1)

    b1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding="same")(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding="same")(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(b1)
    b1 = tf.keras.layers.Dropout(dr)(b1)

    b1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding="same")(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding="same")(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(b1)
    b1 = tf.keras.layers.Dropout(dr)(b1)

    b1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding="same")(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding="same")(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(b1)
    b1 = tf.keras.layers.Dropout(dr)(b1)

    fms = tf.reshape(b1, [-1, mc.width, 256])

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, dropout=0.5))(fms)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, dropout=0.5))(lstm)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, dropout=0.5))(lstm)

    outputs = tf.keras.layers.Dense(mc.output_cat + 1, activation='softmax')(lstm)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def ctc_loss(y_true, y_pred):
    y_idxlst = tf.argmax(y_true, axis=2)
    y_label_length = tf.reduce_sum(y_true, axis=[1, 2])
    label_length = tf.reshape(y_label_length, [y_label_length.shape[0], 1])
    input_length = tf.fill([label_length.shape[0], 1], y_pred.shape[1])
    return tf.keras.backend.ctc_batch_cost(y_true=y_idxlst, y_pred=y_pred, input_length=input_length,label_length=label_length)


def ctc_acc(y_true, y_pred):
    batch_size = y_true.shape[0]
    input_length = tf.fill([batch_size], y_pred.shape[1])
    y_pred_decoded, y_log_probability = tf.keras.backend.ctc_decode(y_pred=y_pred, input_length=input_length,
                                                                    greedy=False)

    y_idxlst_float = tf.argmax(y_true, axis=2)
    y_idxlst_32 = tf.cast(y_idxlst_float, tf.int32)
    y_label_length = tf.reduce_sum(y_true, axis=[1, 2])

    correct_bool_list = tf.map_fn(fn=limited_eq, elems=(y_pred_decoded[0], y_idxlst_32, y_label_length), dtype=tf.bool)
    correct_v = tf.cast(correct_bool_list, tf.int32)
    correct_items = tf.reduce_sum(correct_v)

    return correct_items/batch_size


def limited_eq(t):
    pred, y, length = t
    length_32 = tf.cast(length, tf.int32)
    rlst = tf.range(start=0, limit=length_32, dtype=tf.int32)
    pred_64 = tf.gather(params=pred, indices=rlst)
    pred_32 = tf.cast(pred_64, tf.int32)
    y1 = tf.gather(params=y, indices=rlst)
    limited_result = tf.equal(y1, pred_32)
    limited_32 = tf.cast(limited_result, tf.int32)
    limited_vlen = tf.reduce_sum(limited_32)
    return tf.equal(limited_vlen, length_32)