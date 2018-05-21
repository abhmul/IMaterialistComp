import logging
from keras.callbacks import Callback
import numpy as np
from pyjet.data import Dataset

import utils


class Validator(Callback):
    def __init__(self, validation_data: Dataset, batch_size, debug, *arg_metrics,
                 **kwarg_metrics):
        super().__init__()
        self.validation_data = validation_data
        assert self.validation_data.output_labels
        self.batch_size = batch_size
        self.metrics = kwarg_metrics
        for i, metric in enumerate(arg_metrics):
            self.metrics[i] = metric
        self.debug = debug

        self.threshold_search = np.linspace(0., 1., num=50)[1:-1]
        self.best_thresholds = {}

    def save_data(self):
        logging.info("Saving best thresholds")
        np.savez(utils.get_cv_path(self.model.run_id), **self.best_thresholds)

    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        val_labels = self.validation_data.y

        self.validation_data.output_labels = False
        valgen = self.validation_data.flow(self.batch_size, shuffle=False)
        val_preds = self.model.predict_generator(
            valgen,
            steps = valgen.steps_per_epoch if not self.debug else 5,
            verbose=1)

        if self.debug:
            debug_val_preds = np.zeros((val_labels.shape))
            debug_val_preds[:len(val_preds)] = val_preds
            val_preds = debug_val_preds

        for metric_name, metric in self.metrics.items():
            # Apply the metric to each label and calculate the best threshold
            thresholds = np.zeros((val_labels.shape[1]))
            best_scores = thresholds - float("inf")
            # Try out all the different thresholds
            for t in self.threshold_search:
                class_preds = val_preds > t
                if np.sum(class_preds) == 0:
                    continue
                scores = np.array([
                    metric(val_labels[:, i], class_preds[:, i])
                    for i in range(val_labels.shape[1])
                ])
                is_best = (scores >= best_scores)
                thresholds[is_best] = t
                best_scores[is_best] = scores[is_best]
            # Store the best thresholds
            self.best_thresholds[metric_name] = thresholds

            # Get the average best score for logging
            best_score = np.mean(best_scores)
            logs["val_" + metric_name] = best_score
            print("Validation {metric_name}: {score}".format(
                metric_name=metric_name, score=best_score))
            print("\tThresholds: {}".format(self.best_thresholds[metric_name]))

        self.save_data()
        return
