import polarizeImage as pl
import onehot as oh
import random
import numpy as np


def loadsample(sdir):

    imgs = []
    ys = []
    for line in open(sdir + "/train.txt"):
        parts = line.split(" ")
        d = parts[0].split("/")
        df = d[len(d) - 1]
        img = pl.polarizeimage(sdir + "/" + df)
        if img is None:
            print("img is none")
            continue
        imgs.append(img)
        y = parts[1].replace('\n', '').replace('\r', '')
        try:
            y2 = oh.fromalphabat(y)
            ys.append(y2)
        except IndexError:
            print("error !:", y)

    ys = np.array(ys)
    imgs = np.array(imgs)
    imgs = np.reshape(imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2], 3))

    return ys, imgs


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

    vy, vx = loadsample("../samples/output4_2k")
    for i in vy:
        print(oh.fromonehot(i), end='|')