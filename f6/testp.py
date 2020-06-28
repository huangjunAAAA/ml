import loadsamples as ld
import crnnmodel as nnm
import onehot as oh
import tensorflow.keras as keras
from datetime import datetime
import os


y, x = ld.loadsample("../samples/valV")
vy, vx = ld.loadsample("../samples/valV")
# tx, ty, vx, vy = ld.extravset(x, y)

cfg = nnm.ModelConfig()
cfg.height = x.shape[1]
cfg.width = x.shape[2]
cfg.num_of_sample = y.shape[0]
cfg.output_cat = 26
cnn = None

cnn = nnm.make(cfg)
# cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(cnn.summary())