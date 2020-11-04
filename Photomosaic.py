import glob
import json
import os.path
import time
import operator
import shutil

import numpy as np
import cv2
from annoy import AnnoyIndex
from scipy import spatial
from PIL import Image
from skimage import color
from skimage import io

from sklearn.metrics import jaccard_score

from matplotlib import pyplot

# src file, NPZ file


def create_mosaic(image, npz_file, rows=0, num_kanji=1000, use_canny=False):
    # -----------Open image using PIL and convert to gray scale using luminosity--------------
    img = Image.open(image).convert('L')
    
    # -----------Loading Kanji dataset and setting kanji size--------------
    npz_data = np.load(npz_file)['arr_0.npy']
    ksize = npz_data[0].shape[0]
    
    # -----------Resize the image based on the number of rows desired--------------
    if(rows == 0):
        rows = img.size[0]//ksize
    ratio = img.size[1]/img.size[0]
    width = int(ksize*rows)
    height = int(width*ratio)
    img = img.resize((width, height))
    data = np.array(img)
    # -----------Create the outline image using Canny--------------
    if(use_canny):
        data = auto_canny(data)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        data = cv2.dilate(data, kernel, iterations=1)
    pyplot.imshow(data, cmap='gray', vmin=0, vmax=255)
    pyplot.show()
        
    # -----------Debugging to ensure kanji looks correct--------------
    # pyplot.imshow(sample_kanji, cmap='gray', vmin=0, vmax=255)
    # pyplot.show()

    # ------------Testing how to concatenate two arrays. However this is no efficient-------------
    # concat = np.hstack((sample_kanji, sample_kanji2))
    # pyplot.imshow(concat, cmap='gray', vmin=0, vmax=255)
    # pyplot.show()
    # concat = np.vstack((sample_kanji, sample_kanji2))
    # pyplot.imshow(concat, cmap='gray', vmin=0, vmax=255)
    # pyplot.show()

    # ------------- Quickly testing distances/scores -----------
    # sample_slice = data[:ksize,:ksize]
    # a = sample_slice.flatten()
    # b = sample_kanji.flatten()
    # c = sample_kanji2.flatten()
    # d = np.array([[0, 1, 1], [1, 1, 0]])
    # e = np.array([[1, 1, 1], [1, 0, 0]])

    # distance = jaccard_score(c, b, average='macro')
    # print('Jaccard score between sample: %f' % distance)
    # distance = jaccard_score(c, b, average='micro')
    # print('Jaccard score between sample: %f' % distance)
    # distance = jaccard_score(c, b, average='weighted')
    # print('Jaccard score between sample: %f' % distance)

    # pyplot.imshow(data, cmap='gray', vmin=0, vmax=255)
    # pyplot.show()

    # ------------ core loop -------------
    rows = data.shape[0]//ksize
    cols = data.shape[1]//ksize
    result = np.zeros((rows*ksize, cols*ksize), dtype=np.uint8)
    start_time = time.time()
    for row in range(rows):
        for col in range(cols):
            tile = data[row*ksize:(row+1)*ksize, col*ksize:(col+1)*ksize]
            best_match = -1
            for kanji in npz_data[:num_kanji]:
                # ------------ Different distance metrics -------------
                # score = jaccard_score(tile.flatten(), kanji.flatten(), average='weighted')
                # score = spatial.distance.braycurtis(tile.flatten()/255, kanji.flatten()/255)
                score = spatial.distance.jaccard(tile.flatten(), kanji.flatten())
                # score = abs(np.average(tile.flatten()) - np.average(kanji.flatten()))

                if score < best_match or best_match == -1:
                    print('New best Jaccard score between sample: %f' % score)
                    best_match = score
                    result[row*ksize:(row+1)*ksize, col *
                           ksize:(col+1)*ksize] = kanji #* (1-score)
                    # pyplot.imshow(out, cmap='gray', vmin=0, vmax=255)
                    # pyplot.show()
            print('Appending: %d x %d' % (row, col))

    pyplot.imshow(result, cmap='gray')
    pyplot.show()
    print('Time Taken: %f' % ((time.time() - start_time)/60))


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

# ------------ Samples -------------
create_mosaic('Swirl.jpg', 'kmnist-train-imgs.npz', use_canny=True)
# create_mosaic('Art.jpg', 'kmnist-train-imgs.npz', 60, use_canny=True)
# create_mosaic('Art.jpg', 'kkanji2-1.npz', 60, use_canny=True)

# TQDM for progress bar
