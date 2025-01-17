import logging
import argparse
import numpy as np
# from sklearn.metrics import f1_score

from pyjet.data import ImageDataset
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from callbacks import Plotter
from configurations import load_config
import utils
from utils import f1_score

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
    "--cross_validate",
    action="store_true",
    help="Whether or not to train the model.")
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
                augmenters=(),
                epochs=100,
                batch_size=32,
                epoch_size=10000,
                plot=False,
                load_model=False,
                **kwargs):
    logging.info("Training model with run id %s" % model.run_id)
    logging.info("Using: \n\tbatch_size: {batch_size} \
        \n\tepochs: {epochs} \
        \n\tplot: {plot} \
        \n\tload_model: {load_model} \
        \n\tepoch_size: {epoch_size}".format(**locals()))

    if load_model:
        logging.info("Reloading model from weights")
        model.load_weights(utils.get_model_path(model.run_id), by_name=True)
    if model.fine_tune:
        old_run_id = model.run_id[:-len("-fine-tune")]
        logging.info(
            "Fine tuning model with weights from {}".format(old_run_id))
        model.load_weights(utils.get_model_path(old_run_id), by_name=True)

    steps = epoch_size // batch_size
    val_steps = epoch_size // 10 // batch_size
    traingen = train_dataset.flow(
        batch_size=batch_size,
        steps_per_epoch=steps,
        shuffle=True,
        replace=True,
        seed=utils.get_random_seed())
    valgen = val_dataset.flow(
        batch_size=batch_size,
        steps_per_epoch=val_steps,
        shuffle=True,
        replace=True,
        seed=utils.get_random_seed())

    # Add the augmenters to the training generator
    for augmenter in augmenters:
        traingen = augmenter(traingen)

    # Create the callbacks
    callbacks = [
        ModelCheckpoint(
            utils.get_model_path(model.run_id),
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=True,
            mode="min",
            verbose=1),
        ModelCheckpoint(
            utils.get_model_path(model.run_id + "_f1"),
            monitor="val_f1_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1),
        Plotter(
            monitor="loss",
            scale="linear",
            plot_during_train=plot,
            save_to_file=utils.get_plot_path(model.run_id),
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
    best_val_loss, best_val_f1 = logs["val_loss"][checkpoint], logs[
        "val_f1_loss"][checkpoint]
    logging.info("LOSS CHECKPOINTED -- Loss: {} -- F1: {}".format(
        best_val_loss, best_val_f1))

    checkpoint = min(epochs, key=lambda i: logs["val_f1_loss"][i])
    best_val_loss, best_val_f1 = logs["val_loss"][checkpoint], logs[
        "val_f1_loss"][checkpoint]
    logging.info("ACC CHECKPOINTED -- Loss: {} -- F1: {}".format(
        best_val_loss, best_val_f1))


def cross_validate_predictions(labels, predictions):
    assert labels.shape == predictions.shape
    print("Function cross_validate_predictions() --- Entry Point --- "
          "labels.shape = predictions.shape = {}".format(labels.shape))

    threshold_search = np.linspace(0., 1., num=1000)

    # Apply the metric to each label and calculate the best threshold
    best_thresholds = np.full((labels.shape[1], ), 0.5)
    best_scores = np.full((labels.shape[1], ), -float("inf"))
    # Try out all the different thresholds
    for t in threshold_search:
        class_preds = predictions >= t
        # Calculate the scores on each labels
        scores = f1_score(labels, class_preds)
        # Figure out for which labels the score improved
        is_best = (scores >= best_scores)
        if args.debug:
            logging.info("CALCULATING threshold = {}".format(t))
            logging.info("CALCULATING number of threshold "
                         "improvements = {}".format(np.sum(is_best)))
        # Set the new best thresholds
        best_thresholds[is_best] = t
        best_scores[is_best] = scores[is_best]

    # Any thresholds of 1 means it never guesses that never guesses that class
    # Replace with default value of 0.5
    no_labels = best_thresholds == 1.
    logging.info("Found {} labels with no validation "
                 "examples.".format(np.count_nonzero(no_labels)))
    best_thresholds[no_labels] = 0.5
    # Get the average best score for logging (disregard no labels)
    best_score = np.mean(best_scores[~no_labels])
    print("Validation f1: {score}".format(score=best_score))
    print("\tThresholds: {}".format(best_thresholds))
    print("\Scores: {}".format(best_scores))

    return best_thresholds


def cross_validate_model(model: Model,
                         val_dataset: ImageDataset,
                         test_batch_size=32,
                         **kwargs):
    logging.info("Validationg model with id %s" % model.run_id)
    logging.info("Using: batch_size: {test_batch_size}".format(**locals()))

    valgen = val_dataset.flow(batch_size=test_batch_size, shuffle=False)

    predictions = model.predict_generator(
        valgen, steps=5 if args.debug else valgen.steps_per_epoch, verbose=1)

    if args.debug:
        debug_predictions = np.zeros((len(val_dataset), utils.NUM_LABELS))
        debug_predictions[:len(predictions)] = predictions
        predictions = debug_predictions

    best_thresholds = cross_validate_predictions(val_dataset.y, predictions)
    print("Best Thresholds:")
    print(best_thresholds)
    # Save the results
    np.save(utils.get_cv_path(model.run_id), best_thresholds)

    return best_thresholds


def test_model(model: Model,
               test_dataset: ImageDataset,
               augmenters=(),
               test_batch_size=32,
               **kwargs):
    logging.info("Testing model with id %s" % model.run_id)
    logging.info("Using: batch_size: {test_batch_size}".format(**locals()))

    testgen = test_dataset.flow(batch_size=test_batch_size, shuffle=False)

    # Add any test augmentation
    for augmenter in augmenters:
        augmenter.labels = False
        testgen = augmenter(testgen)

    predictions = model.predict_generator(
        testgen, steps=5 if args.debug else testgen.steps_per_epoch, verbose=1)

    if args.debug:
        debug_predictions = np.zeros((len(test_dataset), utils.NUM_LABELS))
        debug_predictions[:len(predictions)] = predictions
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
    model = run_config["model_func"](num_outputs=228)

    # Train the model
    train_model(model, train_data, val_data, **run_config)
    return model


def cross_validate(data: utils.IMaterialistData, run_config, model=None):

    validation_data = data.load_validation_data()
    # Create the model
    if model is None:
        model = run_config["model_func"](num_outputs=utils.NUM_LABELS)
    model.load_weights(utils.get_model_path(model.run_id), by_name=True)

    return cross_validate_model(model, validation_data, **run_config)


def test(data: utils.IMaterialistData, run_config, model=None):
    # Load the data
    test_data = data.load_test()
    # Create the model
    if model is None:
        model = run_config["model_func"](num_outputs=utils.NUM_LABELS)
    model.load_weights(utils.get_model_path(model.run_id), by_name=True)
    # Test the model w/ augmentation
    predictions = np.zeros((len(test_data), utils.NUM_LABELS))
    if run_config["num_test_augment"] == 0:
        logging.info("Running testing without augmentation")
        run_config["augmenters"] = tuple()
        run_config["num_test_augment"] = 1
    for _ in range(run_config["num_test_augment"]):
        predictions += test_model(model, test_data, **run_config)
    predictions /= run_config["num_test_augment"]
    # Load the thresholds if we need to
    if run_config["threshold"] == "cv":
        thresholds = np.load(utils.get_cv_path(model.run_id)).reshape(
            (1, predictions.shape[1]))
    else:
        thresholds = run_config["threshold"]

    # Cache the predictions for later
    np.save(utils.get_prediction_path(model.run_id), predictions)

    data.save_submission(
        utils.get_submission_path(model.run_id),
        predictions,
        test_data.ids,
        thresholds=thresholds)


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
    if current_run_config["threshold"] == "cv" and args.cross_validate:
        cross_validate(imaterialist_data, current_run_config, trained_model)
        # TODO add the model loading function
        # cross_validate_model(model, val_dataset, test_batch_size)
    if args.test:
        test(imaterialist_data, current_run_config, model=trained_model)
