import os
import random
from config import *
import numpy as np
import matplotlib.pyplot as plt


train_folder = os.path.join("datasets", "train_set")
test_folder = os.path.join("datasets_2", "test")
valid_folder = os.path.join("datasets_2", "valid")
one_hot_labels = np.eye(n_classes, dtype="uint8")


class data_tools:
    def __init__(self, data_folder, split_name):
        self.name = split_name
        self.data_folder = data_folder
        self._data = os.listdir(self.data_folder)
        if split_name == "train":
            self.it = int(batch_size / 8)
        else:
            self.it = int(32 / 8)

    def batch_dispatch(self):
        counter = 0
        random.shuffle(self._data)
        while counter <= len(self._data):
            image_seqs = np.empty((0, time, height, width, color_channels))
            labels = np.empty((0, 2))
            for i in range(self.it):
                npz_path = os.path.join(self.data_folder, self._data[counter])
                np_data = np.load(npz_path, "r")
                image_seqs = np.vstack((image_seqs, np_data["images"] / 255))
                labels = np.vstack((labels, np_data["labels"]))
                counter += 1
                if counter >= len(self._data):
                    counter = 0
                    random.shuffle(self._data)
            yield image_seqs, labels
