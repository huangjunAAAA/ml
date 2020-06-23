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

    onehot = np.zeros(shape=[26*len(ilst)], dtype=int)
    for i in range(len(ilst)):
        onehot[i*26 + ilst[i]] = 1

    return onehot


def fromonehot(o):
    rlst = []
    midx = -1
    mx = 0
    for i in range(0, len(o)):
        if i % 26 == 0:
            mx = 0
            if midx != -1:
                rlst.append(chr(midx % 26 + 65))
            midx = -1
        if mx == 0 or o[i] > mx:
            mx = o[i]
            midx = i

    if midx != -1:
        rlst.append(chr(midx % 26 + 65))

    return ''.join(rlst)



if __name__ == "__main__":
    x1 = fromalphabat('Jyya')
    print("x1=", x1)
    x2 = fromonehot(x1)
    print("x2=", x2)


