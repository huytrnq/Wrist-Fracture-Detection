import cv2
import numpy as np


def ml_prepro_experimental(img):
    """
    Experimental preprocessing step. Not in any pipeline or anything, just for sparsely checking if it works.
    Prepro flow:
        - Background stripping using otsu thresholding
        - Histogram equalization of the foreground
        - Return processed image of the foreground
    :param img: input image grayscale
    :return: processed image
    """
    # Otsu's thresholding on the Grayscale input
    # We are working on xrays, so high density objects show up as bright. Therefore the background should always be dark
    _, bg_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mask = bg_mask.astype(bool) # convert to bool, to utilize numpy builin slicers
    # set bg to 0 for equalization
    result = np.zeros(img.shape, dtype=np.uint8)
    result[bg_mask]=img[bg_mask] # creat all zeros image and then replace the pixels that are true in the mask with the contents of the original image
    # Histogram Equalization for the foreground
    result = cv2.equalizeHist(result)

    return result


if __name__ == '__main__':
    # load image
    img = cv2.imread("C://Users\lok20\OneDrive\_Master\MAIA-ERASMUS//2 Semester\Interdiscipilanry Project AIA_ML_DL\GRAZPEDWRI-DX\images_part1//0001_1297860395_01_WRI-L1_M014.png", cv2.IMREAD_GRAYSCALE)
    # show input
    cv2.imshow('Original Image', img)
    # do prepro and show
    processed = ml_prepro_experimental(img)
    cv2.imshow('Processed', processed)
    # try threshold segmentation and show
    _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Segmented', binary)
    # it looks better but is still imperfect. also sample size is one lol
    cv2.waitKey(0)