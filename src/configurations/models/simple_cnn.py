from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Input, Activation, GlobalAveragePooling2D, Dense, Dropout
from .losses import f1_loss


def simple_cnn(num_outputs, filters=64, num_layers=2, img_size=(256, 256), **kwargs):

    input_img = Input(img_size + (3,), name='input')

    # Resampler
    x = Conv2D(16, kernel_size=1)(input_img)

    for i in range(num_layers):
        x = Conv2D(filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if i != num_layers - 1:
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Flatten and send to dense
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_outputs, activation="sigmoid")(x)

    model = Model(inputs=input_img, outputs=x)

    # compile and return
    model.compile(optimizer='adam', loss=f1_loss, metrics=["accuracy"])
    print("Model is using losses %s" % model.loss_functions[0].__name__)

    return model
