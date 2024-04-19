import skimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def load(path):
    img = skimage.io.imread(path)
    print(img.shape)
    print(img.dtype)
    return resize(img)

def threshold(img, threshold):
    if type(threshold) == int:
        img[img > threshold] = True
        img[img < threshold] = False
        return img
    else:
        thresh = skimage.filters.threshold_mean(img)
        img[img > thresh] = True
        img[img < thresh] = False
        return img

def hist_eq(img):
    if img.dtype == np.uint8:
        return skimage.exposure.equalize_hist(img, nbins=256)
    else:
        return skimage.exposure.equalize_hist(img, nbins=65536)

def window(img):
    thresh = np.max(img)-255
    img -= thresh
    res = np.zeros(img.shape, dtype=np.uint8)
    targets = img < 256
    res[targets] = img[targets]
    print(res.dtype)
    print(np.min(res), np.max(res))
    return res

def resize(img):
    height, width = img.shape
    return skimage.transform.resize(img, (int(height / 2), int(width / 2)), preserve_range=True).astype(img.dtype)


if __name__ == '__main__':
    PATH = "C://Users\lok20\OneDrive\_Master\MAIA-ERASMUS//2 Semester\Interdiscipilanry Project AIA_ML_DL\GRAZPEDWRI-DX"
    breaker = False
    for dirpath, dirnames, filenames in os.walk(PATH):
        for fp in filenames:
            if fp.endswith(".png"):
                img = load(os.path.join(dirpath, fp))

                # regular
                cv2.imshow('img', img)
                img_eq = hist_eq(img)
                cv2.imshow('hist eq', img_eq)
                binary = threshold(img_eq, 'otsu')
                cv2.imshow('binary', binary)

                # windowed
                win = window(img)
                cv2.imshow('windowed', win)
                win_eq = hist_eq(win)
                cv2.imshow('win_eq', win_eq)
                win_binary = threshold(win_eq, 'otsu')
                cv2.imshow('win bin', win_binary)
                k = cv2.waitKey(0)
                if k != 27:
                    continue
                else:
                    breaker = True
                    break
        if breaker:
            break

