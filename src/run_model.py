import logging
import argparse
import numpy as np

from pyjet.data import ImageDataset
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from callbacks import Plotter
from configurations import load_config
import utils

from keras import backend as K

if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("run_id", help="ID of run config to use")
parser.add_argument(
    "--train", action="store_true", help="Whether or not to train the model.")
parser.add_argument(
    "--test", action="store_true", help="Whether or not to test the model.")
parser.add_argument(
    "--plot",
    action="store_true",
    help="Whether or not to plot during training.")
parser.add_argument(
    "--load_model",
    action="store_true",
    help="Whether or not to continue training a saved model")
parser.add_argument(
    "--debug",
    action="store_true",
    help="Whether or not to run the script in debug mode")


def train_model(model: Model,
                train_dataset: ImageDataset,
                val_dataset: ImageDataset,
                augmenters=tuple(),
                epochs=100,
                batch_size=32,
                plot=False,
                load_model=False,
                **kwargs):
    logging.info("Training model with run id %s" % model.run_id)

    if load_model:
        logging.info("Reloading model from weights")
        model.load_weights(utils.get_model_path(model.run_id))

    traingen = train_dataset.flow(
        batch_size=batch_size, shuffle=True, seed=utils.get_random_seed())
    valgen = val_dataset.flow(batch_size=batch_size, shuffle=False)

    # Add the augmenters to the training generator
    for augmenter in augmenters:
        traingen = augmenter(traingen)

    # Create the callbacks
    callbacks = [
        ModelCheckpoint(
            utils.get_model_path(model.run_id),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True),
        ModelCheckpoint(
            utils.get_model_path(model.run_id + "_acc"),
            monitor="val_acc",
            save_best_only=True,
            save_weights_only=True),
        Plotter(
            monitor="loss",
            scale="log",
            plot_during_train=plot,
            save_to_file=utils.get_plot_path(model.run_id),
            block_on_end=False),
        Plotter(
            monitor="acc",
            scale="linear",
            plot_during_train=plot,
            save_to_file=utils.get_plot_path(model.run_id + "_acc"),
            block_on_end=False)
    ]

    # Train the model
    history = model.fit_generator(
        traingen,
        steps_per_epoch=5 if args.debug else traingen.steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=valgen,
        validation_steps=5 if args.debug else valgen.steps_per_epoch)

    # Log the output
    logs = history.history
    epochs = range(len(logs["val_loss"]))
    checkpoint = min(epochs, key=lambda i: logs["val_loss"][i])
    best_val_loss, best_val_acc = logs["val_loss"][checkpoint], logs[
        "val_acc"][checkpoint]
    logging.info("LOSS CHECKPOINTED -- Loss: {} -- Accuracy: {}".format(
        best_val_loss, best_val_acc))

    checkpoint = max(epochs, key=lambda i: logs["val_acc"][i])
    best_val_loss, best_val_acc = logs["val_loss"][checkpoint], logs[
        "val_acc"][checkpoint]
    logging.info("ACC CHECKPOINTED -- Loss: {} -- Accuracy: {}".format(
        best_val_loss, best_val_acc))


def test_model(model: Model,
               test_dataset: ImageDataset,
               augmenters=tuple(),
               batch_size=32,
               **kwargs):
    logging.info("Testing model with id %s" % model.run_id)

    testgen = test_dataset.flow(batch_size=batch_size, shuffle=False)

    # Add any test augmentation
    for augmenter in augmenters:
        augmenter.labels = False
        testgen = augmenter(testgen)

    predictions = model.predict_generator(
        testgen,
        steps=5 if args.debug else testgen.steps_per_epoch,
        verbose=1)

    if args.debug:
        debug_predictions = np.zeros(len(test_dataset), utils.NUM_LABELS)
        debug_predictions[:5] = predictions
        predictions = debug_predictions

    return predictions


def train(data, run_config):
    # Load the data
    if run_config["validation"] == "use_provided":
        train_data = data.load_train_data()
        val_data = data.load_validation_data()
    else:
        raise NotImplementedError(run_config["validation"])
    # Create the model
    model = run_config["model_func"](num_outputs=utils.NUM_LABELS)
    # Train the model
    train_model(model, train_data, val_data, **run_config)
    return model


def test(data: utils.IMaterialistData, run_config, model=None):
    # Load the data
    test_data = data.load_test()
    # Create the model
    if model is None:
        model = run_config["model_func"](num_outputs=utils.NUM_LABELS)
    model.load_weights(utils.get_model_path(model.run_id))
    # Test the model w/ augmentation
    predictions = np.zeros((len(test_data), utils.NUM_LABELS))
    for _ in range(run_config["num_test_augment"]):
        predictions += test_model(model, test_data, **run_config)
    predictions /= run_config["num_test_augment"]
    data.save_submission(
        utils.get_submission_path(model.run_id),
        predictions,
        test_data.ids,
        thresholds=run_config["threshold"])


if __name__ == "__main__":
    args = parser.parse_args()
    current_run_config = load_config(args.run_id)
    imaterialist_data = utils.IMaterialistData(
        path_to_train=current_run_config["path_to_train"],
        path_to_validation=current_run_config["path_to_validation"],
        path_to_test=current_run_config["path_to_test"],
        img_size=current_run_config["img_size"])

    trained_model = None
    if args.debug:
        logging.info("Running in debug mode...")
    if args.train:
        trained_model = train(imaterialist_data, current_run_config)
    if args.test:
        test(imaterialist_data, current_run_config, model=trained_model)
