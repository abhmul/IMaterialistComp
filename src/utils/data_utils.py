import os
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

from pyjet.data import ImageDataset

MAX_LABEL = 228
MIN_LABEL = 1
NUM_LABELS = MAX_LABEL - MIN_LABEL + 1

MIN_IMG_ID = 1


def get_data_ids(path_to_data):
    return os.listdir(path_to_data)


def encode_label(label_num):
    return label_num - MIN_LABEL


class IMaterialistData(object):

    def __init__(self, path_to_train="../input/train", path_to_validation="../input/validation",
                 path_to_test="../input/test", img_size=None, mode="rgb"):
        self.path_to_train = path_to_train
        self.path_to_train_labels = path_to_train + "_labels.csv"
        self.path_to_validation = path_to_validation
        self.path_to_validation_labels = path_to_validation + "_labels.csv"
        self.path_to_test = path_to_test
        self.img_size = img_size
        self.mode = mode

    @staticmethod
    def load_images(path_to_imgs):
        img_ids = get_data_ids(path_to_imgs)
        img_ids.sort(key=lambda fname: int(os.path.splitext(fname)[0]))
        x = [None for _ in img_ids]
        img_ids_list = [None for _ in img_ids]

        logging.info("Opening images stored in %s" % path_to_imgs)
        for n, fname in enumerate(tqdm(img_ids)):
            img_id = int(os.path.splitext(fname)[0])
            x[n] = os.path.join(path_to_imgs, fname)
            img_ids_list[n] = img_id

        assert all(val is not None for val in x)
        assert all(img_id is not None for img_id in img_ids_list)
        img_ids_list = np.array(img_ids_list, dtype=int)
        x = np.asarray(x, dtype="O")
        return x, img_ids_list

    @staticmethod
    def load_labels(path_to_labels):
        logging.info("Creating labels stored in %s" % path_to_labels)
        csvfile = pd.read_csv(path_to_labels)
        # Create the labels array
        labels = np.zeros((len(csvfile), NUM_LABELS), dtype=np.int8)
        for i, label_ids in enumerate(csvfile.labelId):
            labels[csvfile.imageId[i] - MIN_IMG_ID, encode_label(np.array(list(map(int, label_ids.split()))))] = 1
        return labels, csvfile.imageId.values

    def load_labeled_data(self, path_to_imgs, path_to_labels):
        x, ids = self.load_images(path_to_imgs)
        y, ids2 = self.load_labels(path_to_labels)
        assert np.all(ids == ids2)

        return ImageDataset(x, y=y, ids=ids, img_size=self.img_size, mode=self.mode)

    def load_unlabeled_data(self, path_to_imgs):
        x, ids = self.load_images(path_to_imgs)
        return ImageDataset(x, ids=ids, img_size=self.img_size, mode=self.mode)

    def load_train_data(self):
        return self.load_labeled_data(self.path_to_train, self.path_to_train_labels)

    def load_validation_data(self):
        return self.load_labeled_data(self.path_to_validation, self.path_to_validation_labels)

    def load_test(self):
        return self.load_unlabeled_data(self.path_to_test)

    @staticmethod
    def save_submission(save_path, predictions, img_ids, thresholds=0.5):
        """Thresholds can be a numpy array of shape 1 x NUM_LABELS or float"""
        assert predictions.ndim == 2
        num_test_samples = len(predictions)
        rows, labels = np.where(predictions >= thresholds)
        submission_array = np.empty((num_test_samples, 2), dtype="O")
        submission_array[:, 0] = img_ids
        for i in range(num_test_samples):
            row_labels = labels[rows == i]
            submission_array[i, 1] = " ".join(str(label + MIN_LABEL) for label in row_labels)
        pd.DataFrame(submission_array, columns=["image_id", "label_id"]).to_csv(save_path, index=False)
