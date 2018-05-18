from keras.callbacks import Callback
import numpy as np
from pyjet.data import Dataset


class Validator(Callback):
    def __init__(self, validation_data: Dataset, batch_size, *arg_metrics,
                 **kwarg_metrics):
        super().__init__()
        self.validation_data = validation_data
        assert self.validation_data.output_labels
        self.batch_size = batch_size
        self.metrics = kwarg_metrics
        for i, metric in enumerate(arg_metrics):
            self.metrics[i] = metric

    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        valgen = self.validation_data.flow(self.batch_size, shuffle=False)
        val_labels = np.concatenate(
            [next(valgen)[1] for i in range(valgen.steps_per_epoch)], axis=0)

        self.validation_data.output_labels = False
        valgen = self.validation_data.flow(self.batch_size, shuffle=False)
        val_preds = self.model.predict_generator(valgen,
                                                 valgen.steps_per_epoch)

        for metric_name, metric in self.metrics.items():
            score = metric(val_labels, val_preds)
            logs["val_" + metric_name] = score
            print("Validation {metric_name}: {score}".format(
                metric_name=metric_name, score=score))
        return
