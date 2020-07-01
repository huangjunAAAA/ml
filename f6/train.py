import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
LOADCKT=False
TRAIN=True

import loadsamples as ld
import crnnmodel2 as nnm
import tensorflow.keras as keras

from datetime import datetime


ty, tx = ld.loadsample("../samples/valV_tmp")

cfg = nnm.ModelConfig()
cfg.height = tx.shape[1]
cfg.width = tx.shape[2]
cfg.num_of_sample = ty.shape[0]
cfg.output_cat = 26
crnn = None
ckpath = "../checkpoints/f6-ckt"
if LOADCKT and os.path.exists(ckpath):
    crnn = keras.models.load_model(ckpath, compile=False)
    print("loaded model from checkpoint.")
else:
    crnn = nnm.make(cfg)
    print("create new model.")
print(crnn.summary())


logdir = "../tflogs/f6/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logcb = keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=2, histogram_freq=1, write_grads=True)
chkcb = keras.callbacks.ModelCheckpoint(filepath=ckpath, verbose=1)
if TRAIN:
    y, x = ld.loadsample("../samples/outputV_20w")
    crnn.compile(optimizer='adam', loss=nnm.ctc_loss)
    crnn.fit(x=x, y=y, batch_size=100, epochs=100, validation_data=(tx, ty), callbacks=[logcb, chkcb])
else:
    vy, vx = ld.loadsample("../samples/valV")
    crnn.compile(optimizer='adam', loss=nnm.ctc_loss, metrics=[nnm.ctc_acc])
    crnn.fit(x=tx, y=ty, batch_size=10, epochs=1, validation_data=(vx, vy))




