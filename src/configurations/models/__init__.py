from .simple_cnn import simple_cnn
from .transfer_cnn import transfer_model


MODEL_DICT = {simple_cnn.__name__: simple_cnn,
              transfer_model.__name__: transfer_model}


def load_model(model_name, **kwargs):
    return MODEL_DICT[model_name](**kwargs)
