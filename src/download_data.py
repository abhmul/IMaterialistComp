from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import sys
import json
import urllib3
import multiprocessing

from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

__author__ = "Nicolas Lecoy"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_image(fnames_and_urls):
    """
    download image and save its with 90% quality as JPG format
    skip image downloading if image already exists at given path
    :param fnames_and_urls: tuple containing absolute path and url of image
    """
    fname, url = fnames_and_urls
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save(fname, format='PNG')


def parse_dataset(_dataset, _outdir, _max=None):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _outdir: output directory where data will be saved
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    _fnames_urls = []
    with open(_dataset, 'r') as f:
        data = json.load(f)
        for image in data["images"]:
            url = image["url"]
            fname = os.path.join(_outdir, "{}.jpg".format(image["imageId"]))
            _fnames_urls.append((fname, url))
    return _fnames_urls[:_max]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to the dataset to download.")
    parser.add_argument("outdir", help="Path to the the folder to download the dataset to.")
    parser.add_argument("-m", "--max", type=int, default=-1, help="Maximum number of images to download. " \
                                                                  "Defaults to all of them")
    args = parser.parse_args()

    # get args and create output directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # parse json dataset file
    fnames_urls = parse_dataset(args.dataset, args.outdir, _max=(None if args.max == -1 else args.max))

    # download data
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(fnames_urls)) as progress_bar:
        for _ in pool.imap_unordered(download_image, fnames_urls):
            progress_bar.update(1)

    sys.exit(1)
