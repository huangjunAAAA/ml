
import numpy as np
a1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
a2 = np.array([[1, 2, 1], [2, 1, 0], [1, 0, 0]])
# print(a1)
# a2 = np.pad(a1,pad_width=2)
# print(a2)

import loadsamples as ld
import crnnmodel as nnm
import onehot as oh
import tensorflow.keras as keras
from datetime import datetime
import os

import tensorflow as tf


def para_equal(t):
    e1, e2 = t
    r1 = tf.equal(e1, e2)
    r3 = tf.reduce_all(r1)
    if r3:
        return tf.constant(1)
    else:
        return tf.constant(0)

ta1 = tf.convert_to_tensor(a1)
print(ta1)

ta2 = tf.convert_to_tensor(a2)
print(ta2)

_result = tf.map_fn(fn=para_equal, elems=(ta1, ta2), dtype=tf.int32)
print(_result)

# mta1=tf.argmax(ta1,axis=2)
# # print(mta1)
# p2=tf.reduce_sum(ta1,axis=[1,2])
# # print(p2)
# p3 = tf.reshape(p2,[p2.get_shape()[0],1])
# # print(p3)
#
# x=tf.constant([[2,2],[2,1]],name='a')
# print(x)

