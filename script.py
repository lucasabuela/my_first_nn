# Imports
from typing import Callable
import logging
from copy import deepcopy
import numpy as np
import pandas as pd

# import torch


# Definititon of objects
def relu(x: float) -> float:
    """
    The Rectified Linear function.
    """
    return np.maximum(0, x)


def sigmoid(x: float) -> float:
    """
    The sigmoid, used as an activation function.
    """
    return 1 / (1 + np.exp(-x))


def relu_derivative(x: float) -> float:
    """
    The derivative of ReLu.
    """
    return np.where(x <= 0, 0, 1)


def sigmoid_derivative(x: float) -> float:
    """
    The derivative of the sigmoid.
    """
    return -(-np.exp(-x)) / ((1 + np.exp(-x)) ** 2)


def get_derivative_activation_fct(
    f: Callable[[float], float],
) -> Callable[[float], float]:
    """
    Returns the derivative of the function passed as an argument. /!\, Only works if the function
    is relu or the sigmoid.
    """
    if f.__code__.co_code == relu.__code__.co_code:
        return relu_derivative
    if f.__code__.co_code == sigmoid.__code__.co_code:
        return sigmoid_derivative


def variables_instantiation(uncomplete_multilayer_perceptron) -> list[list[np.array]]:
    """
    A utility to declutter the init function of the MultilayerPerceptron class. Instantiates the
    "variables" attribute.

    It has the following structure: it is a list of lists. Furthermore :
        - the first dimension (the first list) represents the layers of the nn;
        - Each layer is represented by a list of three elements : [B, W, A];
        - B is a np.array of size the size of the layer, with the biaises;
        - A is a np.array of size the size of the layer, with the values of the nodes;
        - W is a 2D np.array of size the size of the layer * the size of the previous layer.
    This structure is designed to allow for vectorized operations (both shorter and more easy to read).

    Args:
        uncomplete_multilayer_perceptron : the multilayer perceptron to be completed with the
            "variables" attribute. No type to avoid issues.

    Returns:
        variables (list[list[np.array]]): the variables attribute of the multilayer perceptron.
    """
    variables = []
    for layer in uncomplete_multilayer_perceptron.layers:
        B = layer.biaises
        W = layer.weights
        A = layer.values
        variables.append([B, W, A])

    return variables


class Node:
    """
    Nodes of a neural network.
    """

    def __init__(
        self,
        weights: np.array,
        y_coord: int,
        layer: "Layer",
        value: float,
        biais: float,
        activation_fct: Callable[[float], float] = relu,
    ):
        self.x_coord = layer.rank
        self.value = value
        if self.x_coord != 0:
            self.biais = biais
            self.weights = weights
            self.activation_fct = activation_fct
        else:
            self.biais = None
            self.weights = None
            self.activation_fct = None
        self.y_coord = y_coord
        self.layer = layer


class Layer:
    """
    Layer of a neural network.
    """

    def __init__(
        self,
        multilayer_perceptron: "MultilayerPerceptron",
        rank: int,
        size: int,
        dtype: type,
        activation_fct: Callable[[float], float] = relu,
        activation_fct_derivative: Callable[[float], float] = relu_derivative,
    ):
        self.multilayer_perceptron = multilayer_perceptron
        self.rank = rank
        self.size = size
        self.activation_fct = activation_fct
        self.activation_fct_derivative = activation_fct_derivative
        self.nodes = []

        if rank == 0:
            previous_layer_size = 0
        else:
            previous_layer_size = self.multilayer_perceptron.layers[rank - 1].size

        for j in range(size):
            self.nodes.append(
                Node(
                    biais=1 * (np.random.rand() - 0.5),
                    weights=1 * np.random.rand(previous_layer_size),
                    value=1 * np.random.rand(),
                    y_coord=j,
                    layer=self,
                    activation_fct=self.activation_fct,
                )
            )

        # Finally, we define the arrays attributes. It's no surprising that we have to use a loop
        # to do so, as they are define precisely to vectorize the computation.
        biaises = []
        weights = []
        values = []
        for node in self.nodes:
            biaises.append(node.biais)
            weights.append(node.weights)
            values.append(node.value)

        # We use a lower precision type than usual (float64 instead of float16) because the
        # precision is unecessary here, and we need to save space. Note that we also require
        # the arrays created to be 2D (and not 1D which is the default behavior). It will be
        # required to perform matrices multiplication later on. We also use row vectors rather
        # than column ones as in my notes because operations are slightly faster on them (as numpy
        # and C underneath have the convention row-major). We could change the convention for these
        # arrays but some numpys functions are optimized for row-major order, and might make row-
        # major order copies of the arrays anyway. Thus, remember throughout the code that every
        # matrix is the transposed of its counterpart in my notes.
        self.biaises = np.array(biaises, dtype=dtype, ndmin=2)
        self.weights = np.array(weights, dtype=dtype, ndmin=2)
        self.values = np.array(values, dtype=dtype, ndmin=2)


class MultilayerPerceptron:
    """
    Multilayer-perceptron.
    """

    def __init__(
        self,
        layout: list,
        activation_fct: Callable[[float], float] = relu,
        dtype: type = np.float16,
    ):
        """
        Args:
            layout (list): gives the number of layers, as well as the number of nodes in each layer.
                In the list, the integer at position i (starts at 0) is the number of nodes in layer
                i.
            activation_fct (Callable[[float], float]): the activation function to be used. No matter
                its value, the nodes on the last layer will use the sigmoid.
            dtype (type): the the type with wich the parameters should be saved. Affects memory and
                performance. Default to np.float16.
        """
        self.cost = -1
        self.activation_fct = activation_fct
        self.activation_fct_derivative = get_derivative_activation_fct(activation_fct)
        self.layers = []
        for i, size in enumerate(layout):
            # We isolate the last layer because it needs to have values in the range [0,1], and thus
            # a constrained activation function.
            if i != len(layout) - 1:
                layer = Layer(
                    multilayer_perceptron=self,
                    rank=i,
                    size=size,
                    activation_fct=self.activation_fct,
                    activation_fct_derivative=self.activation_fct_derivative,
                    dtype=dtype,
                )
                self.layers.append(layer)
            else:
                layer = Layer(
                    multilayer_perceptron=self,
                    rank=i,
                    size=size,
                    activation_fct=sigmoid,
                    activation_fct_derivative=sigmoid_derivative,
                    dtype=dtype,
                )
                self.layers.append(layer)

        # We gather all the nn's variables (parameters + values) in one attribute. To undertand its
        # structure, see the description of the corresponding function. Using a deepcopy is crucial
        # to achieve performance (otherwise, the elements of the arrays are not numbers, but
        # pointers to object attributes).
        self.variables = variables_instantiation(uncomplete_multilayer_perceptron=self)


# Definition of functions
def pre_regularization_value(
    biais: float, weights: np.array, values: np.array
) -> float:
    """
    As its name suggests. It corresponds to the following equation : Z^i=W^i A^(i-1)-B^i.
    """
    z = values @ weights.T - biais
    return z


def feed(multilayer_perceptron: "MultilayerPerceptron", example: np.array):
    """
    This function plugs the values of an example (image in this case) on the first layer, and
    computes the new values of all the other nodes up until the last layer. Works in place.

    Args:
        multilayer_perceptron (MultilayerPerceptron): the mutlilayer perceptron to be updated.
        example (np.array): the example to be fed.
    """
    N = len(multilayer_perceptron.layers)

    # Plugs the value of the example on the first layer.
    multilayer_perceptron.variables[0][2] = example

    for i in range(1, N):
        z = pre_regularization_value(
            biais=multilayer_perceptron.variables[i][0],
            weights=multilayer_perceptron.variables[i][1],
            values=multilayer_perceptron.variables[i - 1][2],
        )
        value = multilayer_perceptron.layers[i].activation_fct(z)
        multilayer_perceptron.variables[i][2] = value


def expected_values_last_layer(label: int) -> np.array:
    """
    A small utility which takes as an argument a label (ex : 2) and returns the array of the
    expected values on the last layer of the neural network (ex : [0,0,1,0,0,0,0,0,0,0])
    """
    _expected_values_last_layers = np.zeros(10)
    _expected_values_last_layers[label] = 1
    return _expected_values_last_layers


def cost_gradient_one_example(
    multilayer_perceptron: MultilayerPerceptron, labeled_example: list
):
    """
    Compute the gradient of the cost of the multilayer perceptron with respect to one element
    of the training set.

    Args:
        multilayer_perceptron (MultilayerPerceptron): the neural network whose gradient of the
            cost with respect to the labeled example we want to calculate.
        labeled_example (list): of the form [label, example]. "label" and "example" are 2D row
            arrays of size respectively the size of the last layer and the size of the first layer.
            Note that label is not a str, or an int, it has to be directly the output on the last
            layer that the neural network should have. A small function tailored to each problem
            should be used to turn a label in natural language into the corresponding array of
            output values. Here, exemple is an array of 784 floats between 0 and 1, and abel is an
            array of 10 floats between 0 and 1.

    Returns:
        _cost_gradient_one_example (np.array): ibid.
    """
    # First, we feed the neural network with the example.
    feed(multilayer_perceptron=multilayer_perceptron, example=labeled_example[1])

    # We'll have to compute the partial derivatives w.r.t all the variables, i.e. w.r.t the
    # weights, the biaises and also the values. Thus we define :
    partial_derivatives = deepcopy(multilayer_perceptron.variables)

    # N is the number of layers
    N = len(multilayer_perceptron.layers)

    # Let's start by tackling the special case of the partial derivatives with regard to the values
    # of the last layer. The formula is (𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝐴^(𝑁−1) )=−2 ∗(𝑦−𝐴^(𝑁−1)):
    partial_derivatives[N - 1][2] = -2 * (
        labeled_example[0] - multilayer_perceptron.variables[N - 1][2]
    )

    # Then, the rest of the partial derivatives can be computed layer by layer, by "going back into
    # the tree". It is not possible to vectorize along the principal axis of the nn (on which the
    # layers are attached) as the computation one layers requires the results of the following.
    for i in range(N - 1, 0, -1):
        # There are three (independent) series of computation to be done there :
        # - the gradients w.r.t the biaises b_{j}^{i};
        # - the gradients w.r.t the weights w_{j,k}^{i};
        # - the gradients w.r.t. the values of the previous layer a_{k}^{i-1}.
        # They each can be vectorized, which is what I'll try to implement in a second time.

        # Computation of the gradients w.r.t. the b_{j}^{i}. Note that the product employed is the
        # element-wise product. The formula employed is
        # (𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝐵^𝑖 )=−(𝜕𝐶_(/image))/(𝜕𝐴^𝑖 )⊙𝑓_𝑖^′ (𝑍^𝑖):
        partial_derivatives[i][0] = -(
            partial_derivatives[i][2]
            * multilayer_perceptron.layers[i].activation_fct_derivative(
                pre_regularization_value(
                    biais=multilayer_perceptron.variables[i][0],
                    weights=multilayer_perceptron.variables[i][1],
                    values=multilayer_perceptron.variables[i - 1][2],
                )
            )
        )

        # Computation of the gradients w.r.t. the w_{j,k}^{i}. We use a relation between these and
        # the gradients w.r.t. the biases to accelerate the compute. The formula is:
        # ((𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝑊^𝑖 )=−(𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝐵^𝑖 )^T*(𝐴^(𝑖−1) ).
        partial_derivatives[i][1] = (
            -partial_derivatives[i][0].T @ multilayer_perceptron.variables[i - 1][2]
        )

        # Computation of the gradients w.r.t. the a_{k}^{i-1} with the formula :
        # (𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝐴^(𝑖−1) )=-(𝜕𝐶_(/image))/(𝜕𝐵^𝑖 )∗𝑊^𝑖. Same note as above.
        partial_derivatives[i - 1][2] = -(
            partial_derivatives[i][0] @ multilayer_perceptron.variables[i][1]
        )

    _cost_gradient_one_example = partial_derivatives

    return _cost_gradient_one_example


def cost_gradient(
    multilayer_perceptron: MultilayerPerceptron, training_set: pd.DataFrame
) -> np.array:
    """
    Compute the gradient of the cost of the multilayer perceptron with respect to the training set.

    Args:
        multilayer perceptron (MultilayerPerceptron): the neural network to be trained.
        training_set (pd.DataFrame): the training set.

    Returns:
        _cost_gradient (array): ibid. Same size as the multilayer perceptron parameters.
    """
    _cost_gradients_one_example = cost_gradient_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=training_set
    )
    _cost_gradient = np.mean(_cost_gradients_one_example)
    return _cost_gradient


def learning_one_step(
    multilayer_perceptron: MultilayerPerceptron,
    training_set: pd.DataFrame,
    eta: float = 1,
):
    """Compute the gradient of the cost of the multilayer perceptron with respect to the
    training set, then modify the parameters of the neural network in the opposite direction
    to the gradient. Work in-place.

    Args:
        multilayer_perceptron (MultilayerPerceptron): the neural network to be trained.
        training_set (pd.DataFrame): the training set.
        eta (float): "learning boldness". Hyperparameter. Parameters are nudged by -eta * grad C.
            Default to 1.
    """
    _cost_gradient = cost_gradient(
        multilayer_perceptron=multilayer_perceptron, training_set=training_set
    )
    multilayer_perceptron.parameters += -eta * _cost_gradient


def learning(
    multilayer_perceptron: MultilayerPerceptron,
    training_set: pd.DataFrame,
    stagnation_epsilon: float = 0.1,
    stagnation_steps: int = 3,
    steps_number=None,
    eta: float = 1,
) -> MultilayerPerceptron:
    """
    Train the multilayer perceptron provided on the training set provided. Uses gradient
    descent and retropropagation. Works in-place.

    Args:
        multilayer_perceptron (MultilayerPerceptron): the neural network to be trained.
        training_set (pd.DataFrame): the training set to learn from.
        stagnation_epsilon (float): hyperparameter. Maximal variation of the gradient over
            stagnation_steps below wich a local minimum is considered to have been found.
            Default to ...
        stagnation_steps (int): the number of learning steps upon which the stagnation of
            learning is compared to the stagnation threshold epsilon. Default to 3.
        steps_number (int): number of learning steps to undergo. Optional. If an epsilon
            value is also provided, override the epsilon rule.
        eta (float): "learning boldness". Hyperparameter. At each step of the gradient
            descent, parameters are nudged by -eta * grad C. Default to 1.
    Returns:
        trained_multilayer_perceptron (MultilayerPerceptron): the trained MultilayerPerceptron
    """
    if steps_number is not None:
        for _ in range(steps_number):
            learning_one_step(
                multilayer_perceptron=multilayer_perceptron,
                training_set=training_set,
                eta=eta,
            )
    # La suite à écrire plus tard.
    return multilayer_perceptron


def cost_one_example(
    multilayer_perceptron: MultilayerPerceptron, labeled_example: list
):
    """Make the multilayer perceptron guess for the provided example and computes the square of the
    L^2 distance of its output to the label provided. Note that this function works in place, it
    modifies the values of the neural network (not its parameters).

    Args:
        multilayer_perceptron (MultilayerPerceptron)
        labeled_example (list): of the form [label, example]. "label" and "example" are 2D row
            arrays of size respectively the size of the last layer and the size of the first layer.
            Note that label is not a str, or an int, it has to be directly the output on the last
            layer that the neural network should have. A small function tailored to each problem
            should be used to turn a label in natural language into the corresponding array of
            output values.

    Returns:
        _cost_one_example (float): with, I suppose, the least precision between the one of the
            parameters of multilayer_perceptron.variables and the labeled_example ? Semble être np.float64 par défaut.
    """
    feed(multilayer_perceptron=multilayer_perceptron, example=labeled_example[1])
    N = len(multilayer_perceptron.variables)
    return (
        np.linalg.norm(labeled_example[0] - multilayer_perceptron.variables[N - 1][2])
        ** 2
    )


def main():
    # Initialization of the logging class.
    logging.basiConfig(level="DEBUG")

    return 1


if __name__ == "__main__":
    main()
