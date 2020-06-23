import numpy as np

def fromalphabat(a):
    i = ord(a)
    if i > 96:
        i -= 6
    i -= 65
    onehot = np.zeros(shape=[52], dtype=int)
    onehot[i] = 1
    return onehot


def fromonehot(o):
    midx = 0
    mx = 0
    for i in range(0, len(o)):
        if mx == 0 or o[i] > mx:
            mx = o[i]
            midx = i

    if midx > 26:
        return chr(midx + 71)
    else:
        return chr(midx + 65)



if __name__ == "__main__":
    x1 = fromalphabat('y')
    print("x1=", x1)
    x2 = fromonehot(x1)
    print("x2=", x2)

