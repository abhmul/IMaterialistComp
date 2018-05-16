import logging
import numpy as np

from pyjet.data import ImageDataset
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from callbacks import Plotter
import utils


def train_model(model: Model, train_dataset: ImageDataset, val_dataset: ImageDataset, augmenters=tuple(),
                epochs=100, batch_size=32, plot=False):
    traingen = train_dataset.flow(batch_size=batch_size, shuffle=True, seed=utils.get_random_seed())
    valgen = val_dataset.flow(batch_size=batch_size, shuffle=False)

    # Add the augmenters to the training generator
    for augmenter in augmenters:
        traingen = augmenter(traingen)

    # Create the callbacks
    callbacks = [ModelCheckpoint(utils.get_model_path(model.model_id), monitor="val_loss",
                                 save_best_only=True, save_weights_only=True),
                 ModelCheckpoint(utils.get_model_path(model.model_id + "_acc"), monitor="val_acc",
                                 save_best_only=True, save_weights_only=True),
                 Plotter(monitor="loss", scale="log", plot_during_train=plot,
                         save_to_file=utils.get_plot_path(model.model_id), block_on_end=False),
                 Plotter(monitor="acc", scale="linear", plot_during_train=plot,
                         save_to_file=utils.get_plot_path(model.model_id + "_acc"), block_on_end=False)]

    # Train the model
    history = model.fit_generator(traingen, steps_per_epoch=traingen.steps_per_epoch, epochs=epochs, verbose=1,
                                  callbacks=callbacks, validation_data=valgen, validation_steps=valgen.steps_per_epoch)

    # Log the output
    logs = history.history
    epochs = range(len(logs["val_loss"]))
    checkpoint = min(epochs, key=lambda i: logs["val_loss"][i])
    best_val_loss, best_val_acc = logs["val_loss"][checkpoint], logs["val_acc"][checkpoint]
    logging.info("LOSS CHECKPOINTED -- Loss: {} -- Accuracy: {}".format(best_val_loss, best_val_acc))

    checkpoint = max(epochs, key=lambda i: logs["val_acc"][i])
    best_val_loss, best_val_acc = logs["val_loss"][checkpoint], logs["val_acc"][checkpoint]
    logging.info("ACC CHECKPOINTED -- Loss: {} -- Accuracy: {}".format(best_val_loss, best_val_acc))


def test_model(model: Model, test_dataset: ImageDataset, augmenters=tuple(), batch_size=32):
    testgen = test_dataset.flow(batch_size=batch_size, shuffle=False)

    # Add any test augmentation
    for augmenter in augmenters:
        testgen = augmenter(testgen)

    predictions = model.predict_generator(testgen, steps=testgen.steps_per_epoch, verbose=1)

    return predictions

