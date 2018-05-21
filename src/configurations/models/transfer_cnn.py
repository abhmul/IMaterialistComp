import numpy as np
import logging
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, \
    BatchNormalization, Dropout
from keras.optimizers import SGD

from .losses import f1_loss, f1_bce

imagenet_dict = {
    "inception-resnet-v2": InceptionResNetV2,
    "inception-v3": InceptionV3,
    "densenet-201": DenseNet201
}
finetune_dict = {"inception-resnet-v2": "conv2d_169"}


def create_transfer_model(num_outputs,
                          base_model="inception-resnet-v2",
                          mlp_units=1024,
                          **kwargs):
    # create the base pre-trained model
    base_model = imagenet_dict[base_model](
        weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    # let's add a fully-connected layer
    x = Dense(mlp_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # and a logistic layer
    predictions = Dense(num_outputs, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model


def transfer_model_no_fine_tune(num_outputs,
                                base_model="inception-resnet-v2",
                                mlp_units=1024,
                                **kwargs):

    model, base_model = create_transfer_model(
        num_outputs, base_model=base_model, mlp_units=mlp_units, **kwargs)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional imagenet layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (done *after* setting layers to non-trainable)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=["accuracy", f1_loss])
    print("Model is using losses %s" % model.loss_functions[0].__name__)

    return model


def transfer_model_fine_tune(num_outputs,
                             base_model="inception-resnet-v2",
                             mlp_units=1024,
                             **kwargs):

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
        optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),
        loss=f1_loss,
        metrics=["accuracy"])
    print("Model is using losses %s" % model.loss_functions[0].__name__)

    return model


def transfer_model(num_outputs,
                   base_model="inception-resnet-v2",
                   mlp_units=1024,
                   fine_tune=False,
                   **kwargs):
    if fine_tune:
        return transfer_model_fine_tune(
            num_outputs, base_model=base_model, mlp_units=mlp_units, **kwargs)
    else:
        return transfer_model_no_fine_tune(
            num_outputs, base_model=base_model, mlp_units=mlp_units, **kwargs)
