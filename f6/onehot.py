import numpy as np

def fromalphabat(l):
    alst=list(l)
    ilst=[]
    for a in alst:
        i = ord(a)
        if i > 96:
            i -= 6
        i -= 65
        if i >= 26:
            i -= 26
        ilst.append(i)

    onehot = np.zeros(shape=[len(ilst), 26], dtype=int)
    for i in range(len(ilst)):
        onehot[i, ilst[i]] = 1

    return onehot


def fromonehot(o):
    r = []
    for j in range(o.shape[0]):
        midx = 0
        mx = 0
        for i in range(0, o.shape[1]):
            if mx == 0 or o[j][i] > mx:
                mx = o[j][i]
                midx = i
        if midx > 26:
            r.append(chr(midx + 71))
        else:
            r.append(chr(midx + 65))

    return r



if __name__ == "__main__":
    x1 = fromalphabat('JyyzzZta')
    print("x1=", len(x1))
    x2 = fromonehot(x1)
    print("x2=", x2)


