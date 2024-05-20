import numpy as np
import skimage
import cv2
import os
import pandas as pd
from skimage.filters import unsharp_mask

def load_img(name, test=False): # loads an image
  if test:
    path = 'test/fracture/'+name
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img
  else:
    path = 'train/fracture/'+name
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def load_bbox(name): # loads the image labels
  path = 'txt/'+name+'.txt'
  bboxes = np.loadtxt(path, delimiter=' ')
  bboxes = bboxes[bboxes[:,0]==3,:]
  return bboxes

def convert_coords(h, w, bbox): # convert coords from relative to absolute
  # the yolo file format sucks ass so here we are fixing that shit
  x_c, y_c, b_w, b_h = bbox[1:]
  y_c = int(h*y_c)
  x_c = int(w*x_c)
  b_w = int((h*b_w)/2)
  b_h = int((w*b_h)/2)
  return x_c-b_w, y_c-b_h, x_c+b_w, y_c+b_h

def draw_img_with_bbox(img, bbox): # draws bboxes into an image makes the image color tho
  h, w = img.shape
  shape = bbox.shape
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  if len(shape) > 1:
    for i in range(shape[0]):
      x, y, x_e, y_e = convert_coords(h, w, bbox[i,:])
      cv2.rectangle(img, (x,y), (x_e,y_e), (0,0,255), 3)
  else:
    x, y, x_e, y_e = convert_coords(h, w, bbox)
    cv2.rectangle(img, (x,y), (x_e,y_e), (0,0,255), 3)
  return img

def slice_bbox_mask(img, bbox): # creates a binary mask for everything inside the bbox and an inverted version of the mask for everything that is outside. returns tuple if multiple bbox in one img
  h, w = img.shape
  shape = bbox.shape
  if len(shape) == 2:
    inside = np.zeros(img.shape, dtype=bool)
    for i in range(shape[0]):
      x, y, x_e, y_e = convert_coords(h, w, bbox[i,:])
      mask = np.zeros(img.shape)
      cv2.rectangle(mask, (x,y), (x_e,y_e), 1, -1)
      mask = mask.astype(bool)
      inside = np.bitwise_or(inside, mask)
    return inside
  else:
    x, y, x_e, y_e = convert_coords(h, w, bbox)
    mask = np.zeros(img.shape)
    cv2.rectangle(mask, (x,y), (x_e,y_e), 1, -1)
    mask = mask.astype(bool)
    return mask

def get_hit_ratio(img, bbox): # takes a binary mask and checks how many true elemnts are outside the bbox compared to inside
  mask = slice_bbox_mask(img, bbox)
  inliers = np.sum(np.bitwise_and(mask, img))
  outliers = np.sum(np.bitwise_and(np.invert(mask), img))
  if inliers > 0:
    ratio = outliers/inliers
    return True, ratio
  else:
    return False, -1

def get_lbp_stack(img): # gets three different lbps for the image to be later processed with masks and regions
  h, w = img.shape
  output = np.zeros((h, w, 3))
  output[:,:,0] = skimage.feature.local_binary_pattern(img, 8, 1, 'ror') # according to the manual ror is rotation and grayscale invariant, maybe yields better res
  output[:,:,1] = skimage.feature.local_binary_pattern(img, 12, 2, 'ror')
  output[:,:,2] = skimage.feature.local_binary_pattern(img, 16, 3, 'ror')
  return output


def get_surviving_sample_coords(tile, percentile): # generate sampling survivor coordinates
  coords = np.argwhere(tile) # get all coords that include pixel
  count, _ = coords.shape
  id_survivors = np.random.choice(np.arange(count), int(count*percentile))
  return coords[id_survivors, :]

def probability_sampling(mask, heatmap, tile_size=0.05): # takes a binary mask of seed points and resamples it according to fracture probability sampling happens in tiles of the image shaped by percentile
  # sampling happens in 5 levels: level 1: include 20%, 2: include 40%, 3: include 60%,´4: include 80%, 5: include 100% samples are randomly selected
  h, w = mask.shape
  heatmap = skimage.transform.resize(heatmap, (h, w), anti_aliasing=True) # resize mask to target size
  if mask.shape != heatmap.shape:
    return
  # get dimensions of tiles. we round down here, accepting a little inaccuarcy and misses at the borders but nothing happens there anyway
  tile_w = int(w*tile_size)
  tile_h = int(h*tile_size)
  # make probability thresholds
  p_min = heatmap.min()
  p_max = heatmap.max()
  prob_range = p_max - p_min
  threshold = prob_range/5
  # iterate over tiles
  output = np.zeros((h, w), dtype=bool)
  tile_count = int(1/tile_size)
  for row in range(tile_count-1):
    for col in range(tile_count-1):
      # make coords for slicing (numpy du bist so ne geile sau omg)
      h_start = int(row*tile_h)
      h_end = int((row+1)*tile_h)
      w_start = int(col*tile_w)
      w_end = int((col+1)*tile_w)
      mask_tile = mask[h_start:h_end, w_start:w_end] # slice mask
      if np.max(mask_tile): # check if there are seeds present in tile, if not skip
        peak_prob = np.max(heatmap[h_start:h_end, w_start:w_end]) # get the peak value from the tile to assign the sampling level
        # here we dont need to check for lower boundry fulfillment bc of the if elif else order and function.
        if peak_prob <= threshold: # first level sampling 20%
          survivors = get_surviving_sample_coords(mask_tile, 0.2)

        elif peak_prob <= threshold*2: # second level sampling 40%
          survivors = get_surviving_sample_coords(mask_tile, 0.4)


        elif peak_prob <= threshold*3: # third level sampling 60%
          survivors = get_surviving_sample_coords(mask_tile, 0.6)


        elif peak_prob <= threshold*4: # fourth level sampling 80%
          survivors = get_surviving_sample_coords(mask_tile, 0.8)

        else: # fith level sampling 100%
          survivors = np.argwhere(mask_tile)

        # Update output
        for pos in range(survivors.shape[0]):
          output[survivors[pos, 0]+h_start, survivors[pos, 1]+w_start] = True
  return output

def get_local_hist(lbp, conf):
  if conf == 8:
    hist, _ = np.histogram(img, bins=57, range=(0, 256), density=True)
    return hist

  elif conf == 12:
    hist, _ = np.histogram(img, bins=135, range=(0, 4069), density=True)
    return hist

  elif conf == 16:
    hist, _ = np.histogram(img, bins=243, range=(0, 65536), density=True)
    return hist

  else:
    print('invalid config')
    return None


def generate_feature_vectors(img, mask, feature_stack, n_rad=25, bbox=None, roi=None, heatmap=None): # generates a dataset of (labeled if bbox provided) data to be used in training or classification
  if roi is not None: # if no roi provided just use entire img
    #roi = np.ones(mask.shape).astype(bool)
    mask = np.bitwise_and(roi, mask) # cuts all edges outside the roi out of the mask
  if heatmap is not None: # if a heatmap is provided do the sampling
    mask = probability_sampling(mask, heatmap)
  coords = np.argwhere(mask) # get all the coords for 'True' elements in the array, basically generates an array to walk along to extract features for every seed point
  feat_per_layer = (n_rad*2+1)**2 # get the amount of entries per feature layer depending on the neighborhood size (1=3x3, 2=5x5, usw.)
  h, w, layers = feature_stack.shape # get the amount of layers in the feature stack. Feature stack should always be a 3d array with axes 0 and 1 as the img dimensions and ax 3 as the feature layers
  samples = len(coords) # amount of vectors to be generated
  variables = 57+135+243+1296 # amount of features in a obesrvation (last part is for the hog and has to be changed later to make it variable to fit input shape)
  if bbox is not None: # initializes return of labels array if a bbox is provided, if not just returns the features
    labels = np.zeros(samples)
    bbox_mask = slice_bbox_mask(mask, bbox) # gets the mask of the bbox, if an observation lies inside the true part, it gets labeled as fracture, if not as normal
  output = np.zeros((samples, variables)) # the output array init
  i_s = 0 # index_sample
  for position in coords: # walks through every seed point
    h_c, w_c = position # gets the coords, height_coord, widht_coord
    i_f = 0 # index_feature
    h_start = h_c-n_rad
    h_end = h_c+n_rad
    w_start = w_c-n_rad
    w_end = w_c + n_rad
    output[i_s, 0:57] = get_local_hist(feature_stack[h_start:h_end, w_start:w_end, 0], 8)
    output[i_s, 57:192] = get_local_hist(feature_stack[h_start:h_end, w_start:w_end, 1], 12)
    output[i_s, 192:435] = get_local_hist(feature_stack[h_start:h_end, w_start:w_end, 2], 16)
    output[i_s, 435:] = skimage.feature.hog(img[h_start:h_end, w_start:w_end])
    if bbox is not None: # label data, if bbox present
      if bbox_mask[h_c, w_c]:
        labels[i_s] = 1
      else:
        labels[i_s] = 0
    i_s += 1 # increments the sample position index in the output array
  if bbox is not None:
    return output, labels
  else:
    return output, coords

def exclude_edges(mask, h_perc, w_perc): # removes percentile of border from the mask. gonna be useless when roi is implemented
  h, w = mask.shape
  boundary_exclusion_height = round(h*h_perc)# amount of pixel layers to be discarded at the borders
  boundary_exclusion_width = round(w*w_perc)
  edge_mask_small = edge_mask[boundary_exclusion_height:-boundary_exclusion_height, boundary_exclusion_width:-boundary_exclusion_width]
  edge_mask_pad=np.zeros((h,w))
  edge_mask_pad[boundary_exclusion_height:-boundary_exclusion_height, boundary_exclusion_width:-boundary_exclusion_width]=edge_mask_small
  edge_mask_pad = edge_mask_pad.astype(bool)
  return edge_mask_pad

def hist_match(img, hist, L): # histogram matchin of image to target average histogram
  # Calculate source CDF
  hist_s = cv2.calcHist([img], [0], None, [L], [0, L])
  cdf_s = hist_s.cumsum()
  cdf_s = np.ma.masked_equal(cdf_s, 0)
  cdf_s = (cdf_s - cdf_s.min()) * 255 / (cdf_s.max() - cdf_s.min())
  cdf_s = np.ma.filled(cdf_s, 0).astype('uint8')

  # Calculate target CDF
  cdf_t = hist.cumsum()
  cdf_t = np.ma.masked_equal(cdf_t, 0)
  cdf_t = (cdf_t - cdf_t.min()) * 255 / (cdf_t.max() - cdf_t.min())
  cdf_t = np.ma.filled(cdf_t, 0).astype('uint8')

  # Calculate transform
  LUT = np.zeros(L, dtype='uint8')
  for i in range(L):
      diff = np.abs(cdf_s[i] - cdf_t[0])
      for j in range(1, L):
          new_diff = np.abs(cdf_s[i] - cdf_t[j])
          if new_diff < diff:
              diff = new_diff
              LUT[i] = j

  # Apply transform
  result = LUT[img]

  return result

def draw_result(img, bbox, coords, prediction): # draws the result into the img
  h, w = img.shape
  shape = bbox.shape
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  print(prediction==1.0)
  coords = coords[prediction==1.0,:]

  for elem in coords:
    img[elem[0], elem[1], :] = [0,255,0]
  if len(shape) > 1:
    for i in range(shape[0]):
      x, y, x_e, y_e = convert_coords(h, w, bbox[i,:])
      cv2.rectangle(img, (x,y), (x_e,y_e), (0,0,255), 3)
  else:
    x, y, x_e, y_e = convert_coords(h, w, bbox)
    cv2.rectangle(img, (x,y), (x_e,y_e), (0,0,255), 3)
  return img


if __name__ == '__main__':
    # for loop to walk thorugh directory
    # loads images, generates ds for each image
    # and saves


    ref = np.load('AverageHist.npy')  # ref image for hist matching
    heatmap = np.load('heatmap.npy')  # heatmap for sampling

    total_mis = 0
    samples = 0
    hitrates = 0
    sizesum = 0
    dataset_super = []

    for dirpath, dirnames, filenames in os.walk('train/fracture'):
        for fp in filenames:
            if fp.endswith(".png"):
                img = load_img(fp)  # load img

                bbox = load_bbox(fp.split('.')[0])  # load bbox
                img = hist_match(img, ref, 256)  # histogram mathich to template for stability
                img = cv2.GaussianBlur(img, (5, 5), 0)  # denoise image

                img = ((unsharp_mask(img, radius=2, amount=1)) * 255).clip(0, 255).astype(
                    np.uint8)  # unsharp mask to enhance edges

                edge_mask = cv2.Canny(img, 30, 100)  # generate canny mask ##tune these parameters

                # probab denisty
                edge_mask = exclude_edges(edge_mask, 0.2,0.15)  # cut canny mask edges that have low probability of fracutre according to the fracture heatmap to further enhance perform and avoid out of bounds
                edge_mask = probability_sampling(edge_mask.astype(bool), heatmap)
                # update this to probability sampling density maybe
                suc, hitrate = get_hit_ratio(edge_mask, bbox)  # get the hit ratio of tp:fp
                feature_stack = get_lbp_stack(
                    img)  # generate feature stack: lbp in r different configs, maybe expand later
                dataset, labels = generate_feature_vectors(img, edge_mask, feature_stack, 25,
                                                           bbox)  # generate feature vectors
                # print(dataset.shape, labels.reshape(-1,1).shape)
                dataset = np.concatenate((dataset, labels.reshape(-1, 1)), axis=1)
                #cv2_imshow(np.concatenate((draw_img_with_bbox(img, bbox),
                                           #draw_img_with_bbox(edge_mask.astype(np.uint8) * 255, bbox), feature_stack),
                                          #axis=1))  # show image
                print('Image name:', fp)
                print('Hitrate 1:{:.0f}'.format(hitrate))

                if suc:  # save all stuff to get insight of whole set for a given config
                    hitrates += hitrate
                else:
                    total_mis += hitrate
                sizesum += (dataset.nbytes / 1000000)

                if samples == 0:  # concatenate all the feature vectors to make a complete dataset
                    dataset_super = dataset
                else:
                    dataset_super = np.concatenate((dataset_super, dataset), axis=0)

                samples += 1

                if samples == 101:
                    break

    print('avg hitrate', hitrates / samples)
    print('total misses:', total_mis)
    print('total images processed:', samples)

    cols = list(np.arange(dataset_super.shape[1]))
    cols[-1] = 'class'

    pd.DataFrame(dataset_super, columns=cols).to_csv('generated_data_prob_samp.csv')  # save the whole thing with the pandas data frame
