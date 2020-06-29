import polarizeImage as pl
import onehot as oh
import random
import numpy as np


def loadsample(sdir):

    imgs = []
    ys = []
    max_ylen = -1
    for line in open(sdir + "/train.txt"):
        parts = line.split(" ")
        d = parts[0].split("/")
        df = d[len(d) - 1]
        img = pl.polarizeimage(sdir + "/" + df)
        if img is None:
            print("img is none")
            continue
        imgs.append(img)
        y = ''
        for j in range(1,len(parts)):
            y = y + (parts[j])
        y = y.replace('\n', '').replace('\r', '')
        if max_ylen < len(y):
            max_ylen = len(y)
        try:
            y2 = oh.fromalphabat(y)
            ys.append(y2)
        except IndexError:
            print("error !:", y)

    for i in range(len(ys)):
        pad_num = max_ylen - len(ys[i])
        if pad_num > 0:
            for k in range(pad_num):
                ys[i].append(np.zeros([26], int))

    imgs = np.array(imgs)
    imgs = np.reshape(imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2], 3))

    return np.array(ys), imgs


def extravset(x, y):
    n = len(y)
    t = n//10
    vl = random.sample(range(n), t)
    tx = []
    ty = []
    vx = []
    vy = []
    for i in range(n):
        if i in vl:
            vx.append(x[i])
            vy.append(y[i])
        else:
            tx.append(x[i])
            ty.append(y[i])
    return tx, ty, vx, vy


if __name__ == "__main__":

    vy, vx = loadsample("../samples/output")
    print(vy)