from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, BatchNormalization, Dropout

imagenet_dict = {"inception-resnet-v2": InceptionResNetV2,
                 "inception-v3": InceptionV3,
                 "densenet-201": DenseNet201}


def transfer_model(num_outputs, base_model="inception-resnet-v2", mlp_units=1024, **kwargs):
    # create the base pre-trained model
    base_model = imagenet_dict[base_model](weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    # let's add a fully-connected layer
    x = Dense(mlp_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # and a logistic layer
    predictions = Dense(num_outputs, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional imagenet layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    return model
