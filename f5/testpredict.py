import loadsamples as ld
import nnmodel as nnm
import onehot as oh
import tensorflow.keras as keras

ckpath = "../checkpoints/f5"
cnn = keras.models.load_model(ckpath)
vy, vx = ld.loadsample("../samples/output")

result2 = cnn.predict(vx)
print("predict result:")
for i in result2:
    print(oh.fromonehot(i), end='|')

print("\nactual result:")
for i in vy:
    print(oh.fromonehot(i), end='|')