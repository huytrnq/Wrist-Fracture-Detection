from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import sklearn
import pandas as pd
import numpy as np
import skimage
import matplotlib.pyplot as plt
import cv2

def get_surviving_sample_coords(tile, percentile): # generate sampling survivor coordinates
  coords = np.argwhere(tile) # get all coords that include pixels
  count, _ = coords.shape
  id_survivors = np.random.choice(np.arange(count), int(count*percentile))
  return coords[id_survivors, :]

def probability_sampling(mask, heatmap,
                         tile_size=0.05):  # takes a binary mask of seed points and resamples it according to fracture probability sampling happens in tiles of the image shaped by percentile
    # sampling happens in 5 levels: level 1: include 20%, 2: include 40%, 3: include 60%,´4: include 80%, 5: include 100% samples are randomly selected
    h, w = mask.shape
    heatmap = skimage.transform.resize(heatmap, (h, w), anti_aliasing=True)  # resize mask to target size
    # get dimensions of tiles. we round down here, accepting a little inaccuarcy and misses at the borders but nothing happens there anyway
    tile_w = int(h * tile_size)
    tile_h = int(h * tile_size)
    # make probability thresholds
    p_min = heatmap.min()
    p_max = heatmap.max()
    prob_range = p_max - p_min
    threshold = prob_range / 5
    # iterate over tiles
    output = np.zeros((h, w), dtype=bool)
    tile_count = int(1 / tile_size)
    for row in range(4):
        for col in range(4):
            # make coords for slicing (numpy du bist so ne geile sau omg)
            h_start = row * tile_h
            h_end = (row + 1) * tile_h
            w_start = col * tile_w
            w_end = (col + 1) * tile_w
            mask_tile = mask[h_start:h_end, w_start:w_end]  # slice mask
            if mask_tile.astype(bool).max():  # check if there are seeds present in tile, if not skip
                peak_prob = heatmap[h_start:h_end,
                            w_start:w_end].max()  # get the peak value from the tile to assign the sampling level
                # here we dont need to check for lower boundry fulfillment bc of the if elif else order and function.
                if peak_prob <= threshold:  # first level sampling 20%
                    survivors = get_surviving_sample_coords(mask_tile, 0.2)
                    # correct tile offset
                    survivors[:, 0] += h_start
                    survivors[:, 1] += w_start
                    # update output
                    output[survivors] = True

                elif peak_prob <= threshold * 2:  # second level sampling 40%
                    survivors = get_surviving_sample_coords(mask_tile, 0.4)
                    # correct tile offset
                    survivors[:, 0] += h_start
                    survivors[:, 1] += w_start
                    # update output
                    output[survivors] = True

                elif peak_prob <= threshold * 3:  # third level sampling 60%
                    survivors = get_surviving_sample_coords(mask_tile, 0.6)
                    # correct tile offset
                    survivors[:, 0] += h_start
                    survivors[:, 1] += w_start
                    # update output
                    output[survivors] = True

                elif peak_prob <= threshold * 4:  # fourth level sampling 80%
                    survivors = get_surviving_sample_coords(mask_tile, 0.8)
                    # correct tile offset
                    survivors[:, 0] += h_start
                    survivors[:, 1] += w_start
                    # update output
                    output[survivors] = True

                else:  # fifth level sampling 100%
                    survivors = np.argwhere(mask_tile)
                    # correct tile offset
                    survivors[:, 0] += h_start
                    survivors[:, 1] += w_start
                    # update output
                    output[survivors] = True
    return output

if __name__ == "__main__":
    mask = np.ones((256, 256), dtype=np.uint8)
    heatmap = mask*0.5
    mask = probability_sampling(mask, heatmap)
    print(mask.dtype)
    cv2.imshow('sfd', mask.astype(np.uint8)*255)
    cv2.waitKey(0)
