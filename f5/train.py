import loadsamples as ld
import nnmodel3 as nnm
import tensorflow.keras as keras
from datetime import datetime
import os

ENABLECKT=False

y, x = ld.loadsample("../samples/output4_6w")
vy, vx = ld.loadsample("../samples/val4")
# tx, ty, vx, vy = ld.extravset(x, y)

cfg = nnm.ModelConfig()
cfg.height = x.shape[1]
cfg.width = x.shape[2]
cfg.num_of_sample = y.shape[0]
cfg.output_cat = y.shape[1]

cnn = None
ckpath = "../checkpoints/f5-ckt"
if ENABLECKT and os.path.exists(ckpath):
    cnn = keras.models.load_model(ckpath, custom_objects={'more_char_acc': nnm.more_char_acc})
else:
    cnn = nnm.make(cfg)
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=[nnm.more_char_acc])
print(cnn.summary())


logdir = "../tflogs/f5/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logcb = keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=2, histogram_freq=1, write_grads=True)
chkcb = keras.callbacks.ModelCheckpoint(filepath=ckpath, verbose=1)
cnn.fit(x=x, y=y, batch_size=100, epochs=50, validation_data=(vx,vy), callbacks=[logcb, chkcb])



