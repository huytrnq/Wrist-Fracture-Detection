import skimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def load(path):
    img = skimage.io.imread(path)
    #img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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
    cv2.imshow('raw', img)
    # denoise
    img = skimage.filters.gaussian(img)
    cv2.imshow('denoised', img)
    # equalize histogram so it spans the full 16 bit unsigned range
    img = hist_eq(img)
    cv2.imshow('equalized', img)
    # get threshold that yields the brightest percent of pixels in the img
    thresh = hist_percentage_threshold(img, 20)
    print('Threshold: ', thresh)
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

def mean_shift(img):
    h,w = img.shape
    chan3 = np.zeros((h,w,3), dtype=np.uint8)
    chan3[:,:,0] = img
    chan3[:,:,1] = img
    chan3[:,:,2] = img
    #img = img.astype(np.uint8)

    res = cv2.pyrMeanShiftFiltering(img, 20, 20, 0)
    return res

def kmeans(Z, k):
    Z = Z.reshape((-1, 1))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def hist_percentage_threshold(img, percentage, mode='absolute'):
    '''
    Method to calculate the threshold that returns the brightest #percentage# pixels in the image img
    :param img: the image to consider
    :param percentage: the percentage of brightest pixels desired
    :return: the threshold that will return the brightest percentage pixels in the img
    '''
    percentage = percentage/100

    if mode == 'populated':
        # only returns the top percentage non zero pixels
        hist, bins = np.histogram(img, bins=65536, range=(0, 65535))
        non_zero = hist[hist>0]
        populated_bins = len(non_zero)
        cutoff = round(populated_bins*(percentage))
        hist = np.flip(hist)
        count = 0
        for i in range(len(hist)):
            if hist[i] > 0:
                count += 1
            elif count == cutoff:
                return 65535 - i
        return 'error'

    elif mode == 'absolute':
        # just shift the threshold by percentage from the peak
        peak = np.max(img)
        shift = 65535*(percentage)
        return peak-shift
    else:
        print('Invalid mode')

def lin_stretch(img):
    min_intensity = img.min()
    max_intensity = img.max()
    img = (img - min_intensity) / (max_intensity - min_intensity) * 65535
    return img

if __name__ == '__main__':
    PATH = "C://Users\lok20\OneDrive\_Master\MAIA-ERASMUS//2 Semester\Interdiscipilanry Project AIA_ML_DL\GRAZPEDWRI-DX"
    breaker = False
    for dirpath, dirnames, filenames in os.walk(PATH):
        for fp in filenames:
            if fp.endswith(".png"):
                # load
                img = load(os.path.join(dirpath, fp))
                cv2.imshow('raw', img)
                std = np.std(img)
                mean = np.mean(img)
                snr = np.where(std == 0, 0, mean/std)
                print('Standard Deviation', std,'MEan',mean,'SNR', snr)
                if True:
                    # histogram equalization
                    equalized = (hist_eq(img))
                    eq_copy = equalized.copy() # copy to have untainted equalized, somehow thresholding affects the input and i dont get why
                    cv2.imshow('raw equalized', equalized)
                    # get binary mask of brightest percentage pixels
                    binary = threshold(equalized, hist_percentage_threshold(equalized, 60, 'absolute'))
                    #cv2.imshow('brightest percentage binary', binary)
                    # get final image by init 0 and filling it with the brightest pixels in equalized copy
                    binary = binary.astype(bool)
                    dmo = np.zeros(img.shape, dtype=np.uint16)
                    dmo[binary] = eq_copy[binary]
                    dmo2 = dmo.copy()
                    cv2.imshow('darkest masked out', dmo)
                    # otsu mask of final for reasons
                    ot = threshold(img, 'otsu')
                    cv2.imshow('dmo otsu', ot)
                    # re equalize
                    dmo_eq = hist_eq(dmo)
                    ot_cop = dmo_eq.copy()
                    cv2.imshow('dmo equalized', dmo_eq)
                    # segment agian lol
                    fin_ot = threshold(ot_cop, 'otsu')
                    cv2.imshow('demo equalized otsu', fin_ot)
                    # linstret instead
                    dmo_ls = lin_stretch(dmo2)
                    ls_cop = dmo_ls.copy()
                    cv2.imshow('dmo lin_stretch', dmo_ls)
                    # segment
                    ls_ot = threshold(ls_cop, 'otsu')
                    cv2.imshow('demo lin_stretch ots', ls_ot)
                    # execute k means
                    cv2.imshow('Kmeans dmo_eq', kmeans(dmo_eq, 2))
                    cv2.imshow('Kmeans dmo_ls', kmeans(dmo_ls, 2))
                k = cv2.waitKey(0)
                if k != 27:
                    continue
                else:
                    breaker = True
                    break
        if breaker:
            break

