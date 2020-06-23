import cv2 as cv2
import skimage.filters as skif
import numpy as np
import skimage.morphology as mp
import os


def polarizeimage(imf):
#    print(imf)
    if not os.path.exists(imf):
        return None
 #   print("imf=",imf)
    img = cv2.imread(imf, cv2.IMREAD_COLOR)
#    threshold = skif.threshold_otsu(img);
#    img = img > threshold
#    mp.remove_small_objects(img, 25, connectivity=1, in_place=True)

    # img = img * np.uint8(255)
    # cv2.imshow("kk", img)
    # cv2.waitKey(0)
    return img
