import loadsamples as ld
import nnmodel as nnm
import onehot as oh
import tensorflow.keras as keras
from datetime import datetime


y, x = ld.loadsample("../samples/output1_24w")
vy, vx = ld.loadsample("../samples/val1")
# tx, ty, vx, vy = ld.extravset(x, y)

cfg = nnm.ModelConfig()
cfg.height = x.shape[1]
cfg.width = x.shape[2]
cfg.num_of_sample = y.shape[0]
cfg.output_cat = y.shape[1]
cnn = nnm.make(cfg, True)
print(cnn.summary())

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


logdir = "..\\tflogs\\f4\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=2, histogram_freq=1, write_grads=True)
cnn.fit(x=x, y=y, batch_size=1000, epochs=50, validation_split=0.1, callbacks=[tensorboard_callback], verbose=1)

result2 = cnn.predict(vx)
print("predict result:")
for i in result2:
    print(oh.fromonehot(i), end='')

print("\nactual result:")
for i in vy:
    print(oh.fromonehot(i), end='')

