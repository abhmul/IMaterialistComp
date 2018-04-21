from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import sys
import json
import urllib3
# from joblib import Parallel, delayed
import multiprocessing
import logging

from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

__author__ = "Nicolas Lecoy"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.Logger("download_data", level=logging.INFO)



def download_image(fnames_and_urls):
    """
    download image and save its with 90% quality as JPG format
    skip image downloading if image already exists at given path
    :param fnames_and_urls: tuple containing absolute path and url of image
    """
    fname, url = fnames_and_urls
    jpg_img = os.path.splitext(fname)[0] + ".jpg"
    if os.path.exists(jpg_img):
        os.rename(jpg_img, fname)
    elif not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save(fname, format='PNG')


def parse_dataset(_dataset, _outdir, _max=None, start=0):
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
            fname = os.path.join(_outdir, "{}.png".format(image["imageId"]))
            _fnames_urls.append((fname, url))
    return _fnames_urls[start:_max]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to the dataset to download.")
    parser.add_argument("outdir", help="Path to the the folder to download the dataset to.")
    parser.add_argument("-m", "--max", type=int, default=-1, help="Maximum number of images to download. " \
                                                                  "Defaults to all of them")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start number of images to download. " \
                                                                  "Defaults to all of them")
    parser.add_argument("-p", "--processes", type=int, default=12, help="Number of processes to use.")
    args = parser.parse_args()

    logger.info("Downloading data from %s" % args.dataset)
    logger.info("Downloading data into %s" % args.outdir)
    logger.info("Starting download from image %s" % args.start)
    if args.max == -1:
        logger.info("Downloading to end of data")
        max_download = None
    else:
        logger.info("Downloading to up to image %s" % args.max)
        max_download = args.max
    logger.info("Using %s threads to download" % args.processes)

    # get args and create output directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # parse json dataset file
    fnames_urls = parse_dataset(args.dataset, args.outdir, _max=max_download, start=args.start)

    # download data
    pool = multiprocessing.Pool(processes=args.processes)
    # parallel = Parallel(args.processes, backend='threading', verbose=0)
    # parallel(delayed(download_image)(fname_urls) for fname_urls in tqdm(fnames_urls))

    with tqdm(total=len(fnames_urls)) as progress_bar:

        for _ in pool.imap_unordered(download_image, fnames_urls):
            progress_bar.update(1)

    sys.exit(1)
