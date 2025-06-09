"""
The convergence of the parameters seem to depend heavily on the interplay between the nudge
strenght, the general form of the nudge (should we use instead a condition based on the dot product
of the two last cost_gradients ?) and the number and size of layers. To better understand their
relation, this script gathers functions which plot the cost during the learning process for
different values of these parameters, hoping to uncover experimentally their relationship.
"""

import numpy as np
import script
from script import learning
import mnist_reader
from mnist_reader import load


def test_1():
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
