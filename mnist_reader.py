"""
The dataset used is a subset of the classical MNIST database of handwritten digits (28*28). It was
retrieved from : https://www.kaggle.com/datasets/hojjatk/mnist-dataset, and I reused the provided
reader (I retyped it). It has a training set of precisely 60,000 examples, and a test set of
precisely 10,000 examples.
"""

import struct
from array import array
import random
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []

        # The files are written in binary, thus the "b" mode.
        with open(labels_filepath, mode="rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            images[i] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


#
# Set file paths based on added MNIST Datasets
#
INPUT_PATH = "./data"
training_images_filepath = (
    INPUT_PATH + "/train-images-idx3-ubyte/train-images-idx3-ubyte"
)
training_labels_filepath = (
    INPUT_PATH + "/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)
test_images_filepath = INPUT_PATH + "/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
test_labels_filepath = INPUT_PATH + "/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    plt.close()
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0].reshape(28, 28)
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != "":
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()


def load_dataset(training_size: int = 1000, old_format: bool = False) -> Tuple:
    """
    Load the dataset into 4 ready to use objects. This function expects in the repository the
    presence of a "data" file containing the dataset in the form of four folders named "t10k-
    images.idx3-ubyte", "t10k-labels.idx1-ubyte", "train-images.idx3-ubyte" and "train-labels.
    idx1-ubyte", each oh them containing a file with the same name.

    Args:
        training_size (int): the number of examples in the training size. Default to 1000, so that
            one learning step takes 1s.
        old_format (bool): Wether the training set should be in the old format (the one I first
            coded) List[List[NDArray, NDArray]] or the standard one :
            Tuple[Tuple[List[NDArray], List[int]], Tuple[List[NDArray], List[int]]].
    """
    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    if not old_format:
        return (x_train[: training_size - 1], y_train[: training_size - 1]), (
            x_test,
            y_test,
        )
    else:
        training_set = []
        test_set = []
        for i in range(training_size):
            training_set.append([[x_train[i], y_train[i]]])
        for i in range(len(x_test)):
            test_set.append([x_test[i], y_test[i]])

        return training_set, test_set


def main():
    pass


if __name__ == "__main__":
    main()
