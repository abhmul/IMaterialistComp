import os
import sys
import random
import warnings
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.morphology import label

MODE2FUNC = {"rgb": lambda x: x, "ycbcr": rgb2ycbcr, "gray": lambda x: rgb2gray(x)[:, :, np.newaxis]}


def load_img(path_to_img, img_size=None, mode="rgb"):
    img = imread(path_to_img)
    # Then its a grayscale image
    if img.ndim == 2:
        print("Gray image?:", path_to_img)
        img = gray2rgb(img)
    # Cut out the alpha channel
    img = img[:, :, :3]
    img = MODE2FUNC[mode](img)

    orig_img_shape = img.shape[:2]
    # Resize to the input size
    if img_size is not None:
        img = resize(img, img_size, mode='constant', preserve_range=True).astype(np.uint8)
    return img, orig_img_shape


class IMaterialistData(object):

    def __init__(self, path_to_train="../input/train", path_to_validation="../input/validation",
                 path_to_test="../input/test"):
        self.path_to_train = path_to_train
        self.path_to_validation = path_to_validation
        self.path_to_test = path_to_test

    def load_train_data(self, ):