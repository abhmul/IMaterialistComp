import os
import glob
import numpy as np
import logging
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201, DenseNet121
from keras.applications.xception import Xception

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, \
    BatchNormalization, Dropout, Flatten, Input
from keras.optimizers import SGD

from .losses import f1_loss, f1_bce

imagenet_dict = {
    "inception-resnet-v2": InceptionResNetV2,
    "inception-v3": InceptionV3,
    "densenet-201": DenseNet201,
    "densenet-121": DenseNet121,
    "xception": Xception
}
finetune_dict = {"inception-resnet-v2": "conv2d_161"}
pretrained_dict = {"inception-resnet-v2": "inception_resnet_v2",
                   "densenet-121": "densenet121",
                   "xception": "xception"}


def safe_load_pretrained_weights(model: Model, base_model_name: str):
    # Fix the base model name
    base_model_name = pretrained_dict[base_model_name]
    # Get all the saved models in the keras models dir
    keras_models_dir = os.path.expanduser('~/.keras/models/*')
    pretrained_models_list = glob.glob(keras_models_dir)
    # Filter for models with the same base model name
    pretrained_models_list = [
        model_name for model_name in pretrained_models_list
        if base_model_name in model_name
    ]
    # Check if this file exists
    if pretrained_models_list:
        # Pick the latest one
        pretrained_model = max(pretrained_models_list, key=os.path.getctime)
        logging.info(
            "Loading pretrained weights from {}".format(pretrained_model))
        model.load_weights(pretrained_model, by_name=True)
        return
    # Otherwise we couldn't find the weights
    raise OSError(
        "Could not find pretrained weights for {}".format(base_model_name))


def create_transfer_model(num_outputs,
                          base_model="inception-resnet-v2",
                          mlp_units=1024,
                          **kwargs):
    input_shape = (kwargs["img_size"]) + (3,)
    # inputs = Input(input_shape)

    # create the base pre-trained model
    print(input_shape)
    base_model = imagenet_dict[base_model](
        weights='imagenet', include_top=False, input_shape=input_shape)

    # add a global spatial average pooling layer

    # x = base_model(inputs)
    x = Flatten()(base_model.output)
    # x = Dropout(0.5)(x)
    # let's add a fully-connected layer
    # x = Dense(mlp_units)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # and a logistic layer
    predictions = Dense(num_outputs, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model


def transfer_model_no_fine_tune(num_outputs,
                                base_model="inception-resnet-v2",
                                mlp_units=1024,
                                trainable=False,
                                **kwargs):
    logging.info(
        "Constructing a non-fine tune model based on {}".format(base_model))
    model, base_model_net = create_transfer_model(
        num_outputs, base_model=base_model, mlp_units=mlp_units, **kwargs)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional imagenet layers
    if not trainable:
        logging.info("Freezing training of base layers")
        for layer in base_model_net.layers:
            layer.trainable = False

    # compile the model (done *after* setting layers to non-trainable)
    model.compile(
        optimizer='adam',
        loss="binary_crossentropy",
        metrics=["accuracy", f1_loss])
    logging.info("Model is using losses %s" % model.loss_functions[0].__name__)

    logging.info("Grabbing weights from pretrained model")
    safe_load_pretrained_weights(model, base_model)

    return model


def transfer_model_fine_tune(num_outputs,
                             base_model="inception-resnet-v2",
                             mlp_units=1024,
                             **kwargs):

    logging.info(
        "Constructing a fine tune model based on {}".format(base_model))
    model, base_model_net = create_transfer_model(
        num_outputs, base_model=base_model, mlp_units=mlp_units, **kwargs)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional imagenet layers
    first_trainable = finetune_dict[base_model]
    first_trainable_index = [layer.name for layer in base_model_net.layers
                             ].index(first_trainable)
    logging.info("First trainable layer is %s / %s" %
                 (first_trainable_index, len(base_model_net.layers)))
    for layer in base_model_net.layers[:first_trainable_index]:
        # for layer in base_model_net.layers[:-34]:
        layer.trainable = False

    # compile the model (done *after* setting layers to non-trainable)
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss="binary_crossentropy",
        metrics=["accuracy", f1_loss])
    logging.info("Model is using losses %s" % model.loss_functions[0].__name__)

    return model


def transfer_model(num_outputs,
                   base_model="inception-resnet-v2",
                   mlp_units=1024,
                   fine_tune=False,
                   trainable=False,
                   **kwargs):
    if fine_tune:
        return transfer_model_fine_tune(
            num_outputs, base_model=base_model, mlp_units=mlp_units, **kwargs)
    else:
        return transfer_model_no_fine_tune(
            num_outputs,
            base_model=base_model,
            mlp_units=mlp_units,
            trainable=trainable,
            **kwargs)
