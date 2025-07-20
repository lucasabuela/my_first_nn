# A small work-around to execute the tests properly while using the src/tests layout in the
# repository. Not the most professional. As designed, pytest has to be executed from the
# source directory, not the test directory. I also modified it so as to be able to launch debbug
# sessions of this script. It wouldn't work at first because I used to use relative paths which
# failed because their start location would be the folder tests (and not the root as when using
# pytest).
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


# Imports
import random
from src.dataset_reader import (
    MnistDataloader,
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
    show_images,
    load_dataset,
    download_dataset_from_internet,
)


def test_download_dataset_from_internet():
    download_dataset_from_internet()


def test_load_dataset():
    # First we test it asking for the standard format of the datasets.
    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    #
    # Show some random training and test images
    #
    images_2_show = []
    titles_2_show = []
    for _ in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append("training image [" + str(r) + "] =" + str(y_train[r]))
    for _ in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])
        titles_2_show.append("test image[" + str(r) + "] =" + str(y_test[r]))

    show_images(images_2_show, titles_2_show)

    # Then we test it asking for the old format.
    training_set, test_set = load_dataset(training_size=1000, old_format=True)
