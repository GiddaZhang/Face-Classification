import tensorflow as tf
from PIL.Image import NONE


class MyCallback(tf.keras.callbacks.Callback):

    def __init__(self, patience=[0, 0]):
        super(MyCallback, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=NONE):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = [0, 0]
        self.best_acc = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=NONE):

        val_acc = logs.get("val_accuracy")
        train_acc = logs.get("accuracy")

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.wait[0] = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.model_best = self.model

        elif val_acc < self.best_acc - 0.02:
            self.wait[0] += 1

            if self.wait[0] >= self.patience[0]:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

        if val_acc < self.best_acc - 0.5:
            self.model.set_weights(self.best_weights)

        if train_acc - val_acc > 0.025 and train_acc > 0.96:

            self.wait[1] += 1

            if self.wait[1] >= self.patience[1]:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
