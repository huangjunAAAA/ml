import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import loadsamples as ld
import crnnmodel as nnm
import tensorflow.keras as keras

from datetime import datetime

y, x = ld.loadsample("../samples/output")
vy, vx = ld.loadsample("../samples/output")
# tx, ty, vx, vy = ld.extravset(x, y)

cfg = nnm.ModelConfig()
cfg.height = x.shape[1]
cfg.width = x.shape[2]
cfg.num_of_sample = y.shape[0]
cfg.output_cat = 26
crnn = None
ckpath = "../checkpoints/f6-ckt"
# if os.path.exists(ckpath):
#     crnn = keras.models.load_model(ckpath)
# else:
crnn = nnm.make(cfg)
crnn.compile(optimizer='adam', loss=nnm.ctc_loss, metrics=[nnm.ctc_acc])
print(crnn.summary())


logdir = "../tflogs/f6/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logcb = keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=2, histogram_freq=1, write_grads=True)
chkcb = keras.callbacks.ModelCheckpoint(filepath=ckpath, verbose=1)
crnn.fit(x=x, y=y, batch_size=5, epochs=1, validation_data=(vx, vy), callbacks=[logcb, chkcb])



