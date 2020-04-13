from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class Monitor(BaseLogger):
    """
    CONSTRUCTOR
    figure_path : path to output plot
    jsonPath : (optional) path to json file to serialize values
    startAt : Starting epoch when training resumed after stopped by ctrl + c
    """

    def __init__(self, figure_path, json_path=None, start_epoch=0):
        super(Monitor, self).__init__()
        self.figure_path = figure_path
        self.json_path = json_path
        self.start_epoch = start_epoch

    """
    This method called called once when training begines
    If previous JSON data exists they are loaded into a local dictionary and 
      if we have specified a starting epoch, data is saved  in the dictionary only until that epoch

    Our H will look something like this eventually. Values for each epoch

    H = {
      "train_loss" : [value1, value2, value3],
      "train_acc" : [value1, value2, value3],
      "val_loss" : [value1, value2, value3],
      "val_acc" : [value1, value2, value3]
    }
    """

    def on_train_begin(self, logs={}):
        # Initialize dictionary to store history of "losses"
        self.H = {}

        # If the JSON file exists, load its' content and update H[]
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                # If starting epoch was provided remove all values past the starting epoch
                if self.start_epoch > 0:
                    for k in self.H.keys():
                        values = self.H[k]
                        values = values[:self.start_epoch]
                        self.H[k] = values

    """
    Called each time an epoch ends
    epoch : current epoch number
    logs : contains the training and validation loss + acc for the current epoch.

    """

    def on_epoch_end(self, epoch, logs={}):
        # Loop over the logs and update the loss, acc, etc. for the entire training proces
        for (k, v) in logs.items():
            # If H has a key with current metric retrive it, if not create one with value = []
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        # Check to see if the training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        # Ensure at least two epochs have passed before plotting
        if len(self.H["loss"]) > 1:
            # Plot the training loss and acc
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="training loss")
            plt.plot(N, self.H["val_loss"], label="validation loss")
            plt.title("Training/Validation Loss [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()

            # save the figure
            plt.savefig(self.figure_path + "_loss.png")
            plt.close()

            # Plot the training loss and acc
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["accuracy"], label="training accuracy")
            plt.plot(N, self.H["val_accuracy"], label="validation accuracy")
            plt.title("Training/Validation acc [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("acc")
            plt.legend()

            # save the figure
            plt.savefig(self.figure_path + "_acc.png")
            plt.close()