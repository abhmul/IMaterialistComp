from .simple_cnn import *
from .transfer_cnn import *

MODEL_DICT = {simple_cnn.__name__: simple_cnn,
              transfer_model.__name__: transfer_model}


def load_model(model_name, **kwargs):
    return MODEL_DICT[model_name](**kwargs)