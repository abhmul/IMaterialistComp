"""
Run Configuration is structured as follows
RUN_ID : {
    model_id: The id of the model to use.
    batch_size: The batch size to use during training.
    epochs: The number of epochs to use.
    validation: The style of validation to use. Options are:
        "use_provided": This will use the provided validation set
        -- The following are not implemented --
        "split": This will combine the provided and split the entire dataset
        "kfold": This will combine the provided and run kfold
    "img_size": A list of 2 elements: the image height and width; set to null
        for no resize

    -- The following are optional arguments --
    path_to_train: path to where training data is stored
    path_to_validation: path to where validation data is stored
    path_to_test: path to where test data is stored
    augmenters: A list of json objects. Each object is comprised of:
        [{augmenter_name, **arguments for augmenter}, ... ]
        Defaults to no augmenters
    test_batch_size: The batch size to use during testing. Defaults to
        batch_size
    threshold: The threshold to use during prediction. Options are:
        float: This float value will be used for all classes
        -- The following are not implemented --
        "cv": Will cross validate to find the best threshold for each class
}

Model Configurations are structured as follows
MODEL_ID : {
    model_name: The name of the function to construct the model
    -- All other fields of this json object will be passed to
    the model constructor as keyword arguments --
}
"""

import json
from functools import partial
from pyjet.augmenters import ImageDataAugmenter
from .models import load_model

with open("configurations/run_configurations.json", "r") as config_json:
    RUN_CONFIGS = json.load(config_json)
with open("configurations/model_configurations.json", "r") as config_json:
    MODEL_CONFIGS = json.load(config_json)

AUGMENTERS = {"image": ImageDataAugmenter}


def construct_model(model_id, run_id, img_size, **model_run_params):
    # Constructs the model and gives it relevant attributes
    model_config = MODEL_CONFIGS[model_id]
    model = load_model(**model_config, img_size=img_size, **model_run_params)
    model.model_id = model_id
    model.run_id = run_id
    model.img_size = img_size
    for attribute in model_run_params:
        assert not hasattr(model, attribute), "Keras model already has "
        "attribute %s" % attribute
        setattr(model, attribute, model_run_params[attribute])
    model.summary()
    return model


def build_augmenter(name, **kwargs):
    return AUGMENTERS[name](**kwargs)


def load_config(run_id):
    # Find the config dict in run configs
    run_config = RUN_CONFIGS[run_id]
    # Set up defaults
    run_config["run_id"] = run_id
    run_config["img_size"] = tuple(run_config["img_size"])
    run_config.setdefault("path_to_train", "../input/train")
    run_config.setdefault("path_to_validation", "../input/validation")
    run_config.setdefault("path_to_test", "../input/test")

    run_config.setdefault("test_batch_size", run_config["batch_size"])
    run_config.setdefault("threshold", 0.5)
    run_config.setdefault("augmenters", [])
    run_config.setdefault("num_test_augment", 1)

    model_id = run_config["model_id"]
    img_size = run_config["img_size"]
    run_config["model_func"] = partial(construct_model, model_id, run_id,
                                       img_size)
    run_config["augmenters"] = [build_augmenter(**augmenter_args) for
                                augmenter_args in run_config["augmenters"]]

    return run_config
