"""
The convergence of the parameters seem to depend heavily on the interplay between the nudge
strenght, the general form of the nudge (should we use instead a condition based on the dot product
of the two last cost_gradients ?) and the number and size of layers. To better understand their
relation, this script gathers functions which plot the cost during the learning process for
different values of these parameters, hoping to uncover experimentally their relationship.
"""

# A small work-around to execute the tests properly while using the src/tests layout in the
# repository. Not the most professional. As designed, pytest has to be executed from the
# source directory, not the test directory.
import sys

sys.path.append("./src")


# Imports
from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray
import script
from script import learning
import mnist_reader
from mnist_reader import load_dataset


def modify_dataset_structure(
    loaded_dataset=Tuple[
        Tuple[List[NDArray], List[int]], Tuple[List[NDArray], List[int]]
    ]
):
    """
    A small utility function to give the dataset the desired structure, that is
    [List[NDArray,NDArray]]. The load_dataset function outputs a Tuple[Tuple[List[]]] where the
    elements of the tuple are lists are NDArrays for images, and int for the labels.

    Args:
        loaded_dataset ([List[NDArray,NDArray]])

    Returns:
        (training_set, test_set) (tuple)

    """
    (x_train, y_train), (x_test, y_test) = loaded_dataset
    nb_training_examples = len(y_train)
    nb_test_examples = len(y_test)
    training_set = []
    test_set = []
    for training_example, s in enumerate(x_train):
        training_set.append([training_example, y_train[s]])
    for test_example, s in enumerate(x_test):
        test_example.append([test_example, y_test[s]])
    return training_set, test_set


def test_1():
    (x_train, y_train), (x_test, y_test) = load_dataset(training_size=2)
    training_set, test_set = modify_dataset_structure(
        ((x_train, y_train), (x_test, y_test))
    )
    print(training_set)
    layout = [5, 5, 5]
    N = len(layout)
    multilayer_perceptron = script.MultilayerPerceptron(layout=layout, dtype=np.float64)
    labeled_example_1 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    labeled_example_2 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    training_set = [labeled_example_1, labeled_example_2]
    eta = 1
    previous_costs = learning(
        multilayer_perceptron=multilayer_perceptron,
        training_set=training_set,
        eta=eta,
        max_stagnation_steps=10,
    )
