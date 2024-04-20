import skimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def load(path):
    img = skimage.io.imread(path)
    print('----------------')
    print('Loaded image:', path)
    print(img.shape)
    print(img.dtype)
    print('----------------')
    return resize(img)

def threshold(img, threshold):
    if type(threshold) != str:
        img[img > threshold] = 65535
        img[img < threshold] = 0
        return img
    else:
        thresh = skimage.filters.threshold_mean(img)
        img[img > thresh] = 65535
        img[img < thresh] = 0
        return img

def hist_eq(img, binary=None):
    if binary is None:
        if img.dtype == np.uint8:
            res = skimage.exposure.equalize_hist(img, nbins=256)*255
            return res.astype(np.uint8)
        else:
            res = skimage.exposure.equalize_hist(img, nbins=65536)*65535
            return res.astype(np.uint16)
    else:
        if img.dtype == np.uint8:
            res = skimage.exposure.equalize_hist(img, nbins=256, mask=binary) * 255
            return res.astype(np.uint8)
        else:
            img[np.invert(binary)] = 0
            res = skimage.exposure.equalize_hist(img, nbins=65536) * 65535
            return res.astype(np.uint16)

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

def watershed(img): # basically a pipline lmao
    # equalize histogram so it spans the full 16 bit unsigned range
    img = hist_eq(img)
    # get threshold that yields the brightest percent of pixels in the img
    thresh = hist_percentage_threshold(img, 20)
    # segment to get seeds in rough
    binary = threshold(img, thresh)
    cv2.imshow('watershed pipeleine: Step 1 - binarize', binary)
    # dilate and invert to get bg seed
    dilated = skimage.morphology.binary_dilation(binary, footprint=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,17))).astype(img.dtype)*65535
    background_seed = np.invert(dilated)
    cv2.imshow('watershed pipeleine: Step 2 - dilate', background_seed)
    # erode to get bones
    eroded = skimage.morphology.binary_erosion(binary, footprint=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))).astype(img.dtype)*65535
    cv2.imshow('watershed pipeleine: Step 3 - erosion', eroded)
    return

def hist_percentage_threshold(img, percentage):
    '''
    Method to calculate the threshold that returns the brightest #percentage# pixels in the image img
    :param img: the image to consider
    :param percentage: the percentage of brightest pixels desired
    :return: the threshold that will return the brightest percentage pixels in the img
    '''
    percentage = percentage/100
    # only returns the top percentage non zero pixels
    hist, bins = np.histogram(img, bins=65536, range=(0, 65535))
    #non_zero = hist[hist>0]
    #populated_bins = len(non_zero)
    #return populated_bins*(1-percentage)
    # just shift the threshold by percentage from the peak
    peak = np.max(img)
    print(peak)
    shift = 65535*(percentage)
    return peak-shift

if __name__ == '__main__':
    PATH = "C://Users\lok20\OneDrive\_Master\MAIA-ERASMUS//2 Semester\Interdiscipilanry Project AIA_ML_DL\GRAZPEDWRI-DX"
    breaker = False
    for dirpath, dirnames, filenames in os.walk(PATH):
        for fp in filenames:
            if fp.endswith(".png"):
                img = load(os.path.join(dirpath, fp))

                '''
                # regular
                cv2.imshow('img', img)
                img_eq = hist_eq(img).astype(np.uint16)
                hist, bins = np.histogram(img_eq, bins=65536, range=(0, 65535))
                hist[hist >= 500] = 500
                plt.plot(hist)
                plt.show()
                cv2.imshow('hist eq', img_eq)
                binary = threshold(img_eq, 'otsu')
                cv2.imshow('binary', binary)
                eq_binary = hist_eq(img, binary=threshold(img, 'otsu').astype(bool))
                cv2.imshow('eq_binary', eq_binary)
                '''

                '''
                # windowed
                win = window(img)
                cv2.imshow('windowed', win)
                win_eq = hist_eq(win)
                cv2.imshow('win_eq', win_eq)
                win_binary = threshold(win_eq, 'otsu')
                cv2.imshow('win bin', win_binary)
                '''

                # watershed
                watershed(img)
                k = cv2.waitKey(0)
                if k != 27:
                    continue
                else:
                    breaker = True
                    break
        if breaker:
            break

