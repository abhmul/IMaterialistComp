import os
import glob
import logging

def safe_open_dir(dirpath):
    if not os.path.isdir(dirpath):
        logging.info("Directory %s does not exist, creating it" % dirpath)
        os.makedirs(dirpath)
    return dirpath


def get_model_path(run_id):
    return os.path.join(safe_open_dir("../models/"), str(run_id) + ".h5")


def get_plot_path(run_id):
    return os.path.join(safe_open_dir("../plots/"), str(run_id) + ".png")


def get_submission_path(run_id):
    return os.path.join(safe_open_dir("../submissions/"), str(run_id) + ".csv")


def get_cv_path(run_id):
    return os.path.join(safe_open_dir("../cv/"), str(run_id) + ".csv")

def safe_get_weights(model):  # keras.models.Model
    logging.info("Func safe_get_weights --- Entry Point --- Trace")
    dir = os.path.expanduser('~/.keras/models/*')
    file_list = glob.glob(dir)  # guarenteed
    if file_list:
        file = max(file_list, key=os.path.getctime)
        # if weights are not in ../models/ then the most recently downloaded weights from keras is assumed to be it
        if not os.path.isfile(get_model_path(model.run_id)):
            logging.info("Func safe_get_weights --- Moving most recently downloaded .h5 into ../models/ --- Trace")
            os.rename(file, get_model_path(model.run_id))
