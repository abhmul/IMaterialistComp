import os
import logging


def safe_open_dir(dirpath):
    if not os.path.isdir(dirpath):
        logging.info("Directory %s does not exist, creating it" % dirpath)
        os.makedirs(dirpath)
    return dirpath


def get_model_path(model_id):
    return os.path.join(safe_open_dir("../models/"), str(model_id) + ".h5")


def get_plot_path(model_id):
    return os.path.join(safe_open_dir("../plots/"), str(model_id) + ".png")


def get_submission_path(model_id):
    return os.path.join(safe_open_dir("../submissions/"), str(model_id) + ".csv")