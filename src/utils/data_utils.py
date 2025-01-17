import os
import logging
import pickle as pkl

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

from pyjet.data import ImageDataset

MAX_LABEL = 228
MIN_LABEL = 1
NUM_LABELS = MAX_LABEL - MIN_LABEL + 1


MIN_IMG_ID = 1


def get_data_paths(path_to_data):
    return os.listdir(path_to_data)


def get_img_id(img_fname):
    return int(os.path.splitext(img_fname)[0])


# class LabelEncoder(object):
#
#     def __init__(self):
#         self.classes_ = np.arange(MIN_LABEL, MAX_LABEL + 1).astype(int)
#         self.encoder = {label: i for i, label in
#                         enumerate(self.classes_)}
#         self.decoder = [None] * len(self.encoder)
#         for label, i in self.encoder.items():
#             self.decoder[i] = label
#         assert(all(val is not None for val in self.decoder))
#
#     def encode(self, label):
#         return self.encoder[label]
#
#     def decode(self, i):
#         return self.decoder[i]
#
#     def transform(self, labels):
#         return np.vectorize(lambda label: self.encode(label))(labels)
#
#     def inverse_transform(self, nums):
#         return np.vectorize(lambda i: self.decode(i))(nums)


class IMaterialistData(object):
    def __init__(self,
                 path_to_train="../input/train",
                 path_to_validation="../input/validation",
                 path_to_test="../input/test",
                 img_size=None,
                 mode="rgb"):
        self.path_to_train = path_to_train
        self.path_to_train_labels = path_to_train + "_labels.csv"
        self.path_to_validation = path_to_validation
        self.path_to_validation_labels = path_to_validation + "_labels.csv"
        self.path_to_test = path_to_test
        self.img_size = img_size
        self.mode = mode
        self.label_encoder = MultiLabelBinarizer()

        # Fit the label encoder
        big_label_csv = pd.concat([
            pd.read_csv(self.path_to_train_labels),
            pd.read_csv(self.path_to_validation_labels)
        ])
        self.label_encoder.fit(
            [labels.split(" ") for labels in big_label_csv.labelId]
        )
        logging.info("Found %s labels in data."
                     % len(self.label_encoder.classes_))
        assert len(self.label_encoder.classes_) == NUM_LABELS

    @staticmethod
    def load_images(path_to_imgs):
        img_fnames = np.array(sorted(get_data_paths(path_to_imgs),
                              key=get_img_id))
        img_ids = np.array(list(map(get_img_id, img_fnames)), dtype=int)
        logging.info("Found {} images in {}".format(len(img_ids),
                                                    path_to_imgs))

        # Make the x array
        x = np.empty((len(img_ids),), dtype="O")
        logging.info("Opening images stored in %s" % path_to_imgs)
        for n, fname in enumerate(tqdm(img_fnames)):
            x[n] = os.path.join(path_to_imgs, fname)
            assert img_ids[n] == get_img_id(fname)

        return x, img_ids

    def load_labels(self, path_to_labels):
        logging.info("Creating labels stored in %s" % path_to_labels)
        csvfile = pd.read_csv(path_to_labels)

        labels = self.label_encoder.transform(
            [labels.split(" ") for labels in csvfile.labelId]
        )

        # Create the labels array
        return labels, csvfile.imageId.values

    def load_labeled_data(self, path_to_imgs, path_to_labels):
        x, ids = self.load_images(path_to_imgs)
        y, ids2 = self.load_labels(path_to_labels)
        assert np.all(ids == ids2), "Image ids and label ids don't match!"

        return ImageDataset(
            x, y=y, ids=ids, img_size=self.img_size, mode=self.mode)

    def load_unlabeled_data(self, path_to_imgs):
        x, ids = self.load_images(path_to_imgs)
        return ImageDataset(x, ids=ids, img_size=self.img_size, mode=self.mode)

    def load_train_data(self):
        return self.load_labeled_data(self.path_to_train,
                                      self.path_to_train_labels)

    def load_validation_data(self):
        return self.load_labeled_data(self.path_to_validation,
                                      self.path_to_validation_labels)

    def load_test(self):
        return self.load_unlabeled_data(self.path_to_test)

    def get_class_weights(self):
        train_data = self.load_train_data()
        val_data = self.load_validation_data()
        ratios = (np.sum(train_data.y, axis=0) + np.sum(
            val_data.y, axis=0)) / (len(train_data.y) + len(val_data.y))
        weights = np.zeros((ratios.shape[0], 2))
        weights[:, 0] = ratios
        weights[:, 1] = 1. - ratios
        return weights

    def save_submission(self, save_path, predictions, img_ids, thresholds=0.5):
        """Thresholds can be a numpy array of shape 1 x NUM_LABELS or float"""
        assert predictions.ndim == 2
        num_test_samples = len(predictions)
        khot_preds = predictions >= thresholds

        # Make the submission array
        class_preds = self.label_encoder.inverse_transform(khot_preds)
        assert(len(class_preds) == num_test_samples)

        submission_array = np.empty((num_test_samples, 2), dtype="O")
        submission_array[:, 0] = img_ids
        for i in range(num_test_samples):
            submission_array[i, 1] = " ".join(str(label) for label in
                                              class_preds[i])

        pd.DataFrame(
            submission_array, columns=["image_id", "label_id"]).to_csv(
                save_path, index=False)


CLASS_WEIGHTS = np.array([[1.00011227e+00, 8.90818261e+03],
                           [1.02198313e+00, 4.64894264e+01],
                           [1.00286830e+00, 3.49638567e+02],
                           [1.00520342e+00, 1.93181407e+02],
                           [1.00493619e+00, 2.03585254e+02],
                           [1.00469079e+00, 2.14183776e+02],
                           [1.02752666e+00, 3.73284142e+01],
                           [1.00222469e+00, 4.50501759e+02],
                           [1.01045130e+00, 9.66818611e+01],
                           [1.00287812e+00, 3.48449320e+02],
                           [1.00592099e+00, 1.69890713e+02],
                           [1.00121873e+00, 8.21524459e+02],
                           [1.00791026e+00, 1.27418035e+02],
                           [1.01295610e+00, 7.81836984e+01],
                           [1.01032076e+00, 9.78921166e+01],
                           [1.00002636e+00, 3.79422593e+04],
                           [1.33403783e+00, 3.99367291e+00],
                           [1.04010709e+00, 2.59332456e+01],
                           [1.14273078e+00, 8.00619744e+00],
                           [1.11704760e+00, 9.54353294e+00],
                           [1.00369562e+00, 2.71590933e+02],
                           [1.00120210e+00, 8.32878862e+02],
                           [1.00032614e+00, 3.06718862e+03],
                           [1.00033688e+00, 2.96939420e+03],
                           [1.00813244e+00, 1.23964303e+02],
                           [1.00922591e+00, 1.09390390e+02],
                           [1.00088615e+00, 1.12948291e+03],
                           [1.01586809e+00, 6.40195601e+01],
                           [1.00072091e+00, 1.38813144e+03],
                           [1.00503774e+00, 1.99501655e+02],
                           [1.00346949e+00, 2.89226708e+02],
                           [1.01262368e+00, 8.02161929e+01],
                           [1.00393857e+00, 2.54899477e+02],
                           [1.00306371e+00, 3.27402045e+02],
                           [1.00364252e+00, 2.75535503e+02],
                           [1.06530504e+00, 1.63127548e+01],
                           [1.01010359e+00, 9.99747243e+01],
                           [1.00923983e+00, 1.09227103e+02],
                           [1.00433029e+00, 2.31931401e+02],
                           [1.00665146e+00, 1.51343034e+02],
                           [1.00017867e+00, 5.59803825e+03],
                           [1.00483171e+00, 2.07966098e+02],
                           [1.00397103e+00, 2.52823544e+02],
                           [1.08025679e+00, 1.34600053e+01],
                           [1.00340855e+00, 2.94379598e+02],
                           [1.00006931e+00, 1.44287465e+04],
                           [1.01062774e+00, 9.50933816e+01],
                           [1.00845696e+00, 1.19245839e+02],
                           [1.07270186e+00, 1.47548069e+01],
                           [1.00089984e+00, 1.11231379e+03],
                           [1.01006475e+00, 1.00356681e+02],
                           [1.00871612e+00, 1.15729892e+02],
                           [1.16556797e+00, 7.03981556e+00],
                           [1.00267297e+00, 3.75115709e+02],
                           [1.00982679e+00, 1.02762664e+02],
                           [1.00812947e+00, 1.24009321e+02],
                           [1.00233353e+00, 4.29535010e+02],
                           [1.00303424e+00, 3.30571475e+02],
                           [1.04121497e+00, 2.52630268e+01],
                           [1.00364350e+00, 2.75461414e+02],
                           [1.02614954e+00, 3.92415920e+01],
                           [1.14223387e+00, 8.03067432e+00],
                           [1.02086593e+00, 4.89250203e+01],
                           [1.00553888e+00, 1.81541910e+02],
                           [1.01246856e+00, 8.12017280e+01],
                           [3.74445245e+00, 1.36437141e+00],
                           [1.00161617e+00, 6.19746521e+02],
                           [1.00033200e+00, 3.01306176e+03],
                           [1.00913942e+00, 1.10416146e+02],
                           [1.05015833e+00, 2.09368690e+01],
                           [1.00808681e+00, 1.24658189e+02],
                           [1.00894959e+00, 1.12736987e+02],
                           [1.03542269e+00, 2.92304905e+01],
                           [1.01150786e+00, 8.78971257e+01],
                           [1.00581334e+00, 1.73018240e+02],
                           [1.00023726e+00, 4.21580658e+03],
                           [1.00641411e+00, 1.56906264e+02],
                           [1.04717196e+00, 2.21990335e+01],
                           [1.06014565e+00, 1.76263076e+01],
                           [1.00246594e+00, 4.06524206e+02],
                           [1.00686516e+00, 1.46662992e+02],
                           [1.00891183e+00, 1.13210410e+02],
                           [1.00029391e+00, 3.40345847e+03],
                           [1.00005662e+00, 1.76627759e+04],
                           [1.00114144e+00, 8.77089897e+02],
                           [1.00006833e+00, 1.46348714e+04],
                           [1.02149501e+00, 4.75224289e+01],
                           [1.02222992e+00, 4.59844241e+01],
                           [1.00349112e+00, 2.87441358e+02],
                           [1.00109839e+00, 9.11424377e+02],
                           [1.03465721e+00, 2.98540289e+01],
                           [1.00880751e+00, 1.14539468e+02],
                           [1.00939894e+00, 1.07395010e+02],
                           [1.00036033e+00, 2.77626287e+03],
                           [1.01833304e+00, 5.55463319e+01],
                           [1.00310103e+00, 3.23473634e+02],
                           [1.03113721e+00, 3.31159205e+01],
                           [1.09611368e+00, 1.14043460e+01],
                           [1.01657979e+00, 6.13144003e+01],
                           [1.02094020e+00, 4.87550447e+01],
                           [1.01145293e+00, 8.83138793e+01],
                           [1.00580642e+00, 1.73223030e+02],
                           [1.00850362e+00, 1.18597013e+02],
                           [1.00018452e+00, 5.42032275e+03],
                           [1.48426043e+00, 3.06500458e+00],
                           [1.23533192e+00, 5.24931722e+00],
                           [1.00006638e+00, 1.50653088e+04],
                           [1.00047561e+00, 2.10357495e+03],
                           [1.00800547e+00, 1.25914577e+02],
                           [1.02934170e+00, 3.50811931e+01],
                           [1.00674840e+00, 1.49183195e+02],
                           [1.00111209e+00, 9.00211775e+02],
                           [1.08283848e+00, 1.30716847e+01],
                           [1.00985068e+00, 1.02515861e+02],
                           [1.02427814e+00, 4.21893172e+01],
                           [1.05994053e+00, 1.76832030e+01],
                           [1.01467878e+00, 6.91255735e+01],
                           [1.00387560e+00, 2.59024273e+02],
                           [1.00078543e+00, 1.27418035e+03],
                           [1.00425054e+00, 2.36264068e+02],
                           [1.00618676e+00, 1.62635498e+02],
                           [1.03147463e+00, 3.27716251e+01],
                           [1.00056941e+00, 1.75718868e+03],
                           [1.00040624e+00, 2.46259856e+03],
                           [1.00099764e+00, 1.00337023e+03],
                           [1.00437558e+00, 2.29540892e+02],
                           [1.00372020e+00, 2.69802739e+02],
                           [1.03091307e+00, 3.33487744e+01],
                           [1.00091939e+00, 1.08867269e+03],
                           [1.00373397e+00, 2.68811598e+02],
                           [1.05316502e+00, 1.98093590e+01],
                           [1.00787357e+00, 1.28007122e+02],
                           [1.06394087e+00, 1.66394497e+01],
                           [1.00092233e+00, 1.08521292e+03],
                           [1.01123727e+00, 8.99895467e+01],
                           [1.00929651e+00, 1.08567295e+02],
                           [1.11568264e+00, 9.64433921e+00],
                           [1.11290831e+00, 9.85674425e+00],
                           [1.00684042e+00, 1.47189799e+02],
                           [1.00759205e+00, 1.32716803e+02],
                           [1.00579161e+00, 1.73663502e+02],
                           [1.02698072e+00, 3.80634986e+01],
                           [1.01738442e+00, 5.85227649e+01],
                           [1.02160299e+00, 4.72898952e+01],
                           [1.00004198e+00, 2.38242093e+04],
                           [1.00240120e+00, 4.17457620e+02],
                           [1.00726313e+00, 1.38681603e+02],
                           [1.11132072e+00, 9.98305366e+00],
                           [1.00118644e+00, 8.43855848e+02],
                           [1.00791721e+00, 1.27307195e+02],
                           [1.03164186e+00, 3.26037045e+01],
                           [1.00403893e+00, 2.48590391e+02],
                           [1.34824494e+00, 3.87154205e+00],
                           [1.03322653e+00, 3.10964364e+01],
                           [1.02023254e+00, 5.04253298e+01],
                           [1.00007419e+00, 1.34794868e+04],
                           [1.00025386e+00, 3.94015769e+03],
                           [1.01234650e+00, 8.19946374e+01],
                           [1.01495121e+00, 6.78842356e+01],
                           [1.00187183e+00, 5.35235632e+02],
                           [1.00009567e+00, 1.04534796e+04],
                           [1.00005076e+00, 1.97007885e+04],
                           [1.00000781e+00, 1.28055125e+05],
                           [1.13020689e+00, 8.68008507e+00],
                           [1.01348021e+00, 7.51828123e+01],
                           [1.02153677e+00, 4.74322159e+01],
                           [1.01293307e+00, 7.83211774e+01],
                           [1.00442384e+00, 2.27048094e+02],
                           [1.00850957e+00, 1.18514692e+02],
                           [1.03673984e+00, 2.82184057e+01],
                           [1.31522553e+00, 4.17233192e+00],
                           [1.00023824e+00, 4.19852869e+03],
                           [1.00061242e+00, 1.63387719e+03],
                           [1.00028511e+00, 3.50835959e+03],
                           [1.05107093e+00, 2.05806095e+01],
                           [1.07033593e+00, 1.52174837e+01],
                           [1.00160148e+00, 6.25421856e+02],
                           [1.00728789e+00, 1.38213842e+02],
                           [1.00026363e+00, 3.79422593e+03],
                           [1.03765239e+00, 2.75587389e+01],
                           [1.01924185e+00, 5.29700620e+01],
                           [1.00207861e+00, 4.82089882e+02],
                           [1.01552067e+00, 6.54302229e+01],
                           [1.06776781e+00, 1.57562675e+01],
                           [1.00472429e+00, 2.12671995e+02],
                           [1.06268049e+00, 1.69539264e+01],
                           [1.00595655e+00, 1.68882460e+02],
                           [1.00315996e+00, 3.17459250e+02],
                           [1.02212079e+00, 4.62063506e+01],
                           [1.03657515e+00, 2.83409688e+01],
                           [1.00059091e+00, 1.69329091e+03],
                           [1.00108176e+00, 9.25420958e+02],
                           [1.02811961e+00, 3.65623684e+01],
                           [1.00063685e+00, 1.57122853e+03],
                           [1.00105046e+00, 9.52968372e+02],
                           [1.00318944e+00, 3.14535155e+02],
                           [1.00110035e+00, 9.09805506e+02],
                           [1.00061339e+00, 1.63127548e+03],
                           [1.00137434e+00, 7.28620910e+02],
                           [1.00696017e+00, 1.44674622e+02],
                           [1.00730077e+00, 1.37971852e+02],
                           [1.00152902e+00, 6.55013427e+02],
                           [1.01440950e+00, 7.03986394e+01],
                           [1.02510219e+00, 4.08371602e+01],
                           [1.03327863e+00, 3.10493120e+01],
                           [1.00285652e+00, 3.51076422e+02],
                           [1.00023824e+00, 4.19852869e+03],
                           [1.00084312e+00, 1.18706952e+03],
                           [1.00227077e+00, 4.41379147e+02],
                           [1.01454111e+00, 6.97705510e+01],
                           [1.00110915e+00, 9.02591189e+02],
                           [1.00700273e+00, 1.43801376e+02],
                           [1.00524978e+00, 1.91484299e+02],
                           [1.17435034e+00, 6.73557800e+00],
                           [1.00010446e+00, 9.57421495e+03],
                           [1.00728690e+00, 1.38232492e+02],
                           [1.00901021e+00, 1.11985243e+02],
                           [1.03296607e+00, 3.13342203e+01],
                           [1.00058212e+00, 1.71886074e+03],
                           [1.01468179e+00, 6.91115834e+01],
                           [1.00028316e+00, 3.53255517e+03],
                           [1.17790426e+00, 6.62100100e+00],
                           [1.00021089e+00, 4.74278241e+03],
                           [1.01456321e+00, 6.96661680e+01],
                           [1.00907978e+00, 1.11134845e+02],
                           [1.02800613e+00, 3.67064746e+01],
                           [1.00741568e+00, 1.35849489e+02],
                           [1.00176407e+00, 5.67871951e+02]])
