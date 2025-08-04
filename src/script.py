# Imports
from typing import Callable
import logging
import pickle
from copy import deepcopy
import random
import numpy as np
import tqdm

# import torch


# Definititon of objects
def relu(x: float | np.ndarray) -> float | np.ndarray:
    """
    The Rectified Linear function.
    """
    return np.maximum(0, x)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """
    The sigmoid, used as an activation function.
    """
    # Using 1 / (1 + np.exp(-x)) seems natural, but it raises an overflow error if x is too
    # "negative" ( precisely if x < -709. Note that no error is raised if x is too large; and 0 is
    # returned, which is an acceptable behavior). A first idea is to use :
    #
    # return np.where(x >= -709, 1 / (1 + np.exp(-x)), 1)
    #
    # It works. The only thing is that it raises warnings of overflow because the left expression
    # is evaluated, even if it is not chosen after because x is too large. A method that avoids
    # this is :
    #
    # return np.piecewise(x, [x >= -709, x < -709], [lambda y: 1 / (1 + np.exp(-y)), 1])
    #
    # It's almost perfect. It can still be slightly improved with :
    return np.piecewise(
        x,
        [x >= -709, x < -709],
        [lambda y: 1 / (1 + np.exp(-y)), lambda y: np.exp(y) / (np.exp(y) + 1)],
    )
    # It is better because when x is in [-744, -710], it returns the real (small) value rather than
    # zero. It is very subtle, it stems from the fact that np.exp(x) "breaks" at 709 in the non-
    # negative but -745 in the negative.
    #
    # One could wonder wether these approximations aren't detrimental for the proper working if the
    # algorithm. I don't know yet how to show it, even intuitively. We'll see if it works.


def relu_derivative(x: float) -> float:
    """
    The derivative of ReLu.
    """
    return np.where(x <= 0, 0, 1)


def sigmoid_derivative(x: float) -> float:
    """
    The derivative of the sigmoid.
    """
    # The same reasoning as for sigmoid applies. It is slightly more complicated here because of
    # the intermediate terms of the expressions which becomes huge faster than in the regular
    # sigmoid function. The frontier here is 354, which is rougly 709 / 2, which makes sense.
    return np.piecewise(
        x,
        [x >= -354, x < -354],
        [
            lambda y: -(-np.exp(-y)) / ((1 + np.exp(-y)) ** 2),
            lambda y: -(-np.exp(y)) / ((np.exp(y) + 1) ** 2),
        ],
    )


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


# à transormer peut-être en méthode de la classe multilayerperceptron nn ?
def variables_instantiation(uncomplete_multilayer_perceptron) -> list[list[np.ndarray]]:
    """
    A utility to declutter the init function of the MultilayerPerceptron class. Instantiates the
    "variables" attribute.

    It has the following structure: it is a list of lists. Furthermore :
        - the first dimension (the first list) represents the layers of the nn;
        - Each layer is represented by a list of three elements : [B, W, A];
        - B is a np.ndarray of size the size of the layer, with the biaises;
        - A is a np.ndarray of size the size of the layer, with the values of the nodes;
        - W is a 2D np.ndarray of size the size of the layer * the size of the previous layer.
    This structure is designed to allow for vectorized operations (both shorter and more easy to
    read).

    Args:
        uncomplete_multilayer_perceptron : the multilayer perceptron to be completed with the
            "variables" attribute. No type to avoid issues.

    Returns:
        variables (list[list[np.ndarray]]): the variables attribute of the multilayer perceptron.
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
        weights: np.ndarray,
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
                    weights=1 * (np.random.rand(previous_layer_size) - 0.5),
                    value=1 * np.random.rand(),
                    y_coord=j,
                    layer=self,
                    activation_fct=self.activation_fct,
                )
            )

        # Finally, we define the arrays attributes. It's no surprising that we have to use a loop
        # to do so, as they are defined precisely to vectorize the computation.
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
        # than column ones as in my initial notes (addendum : I've updated them now) because
        # operations are slightly faster on them (as numpy and C underneath have the convention
        # row-major). We could change the convention for these arrays but some numpys functions
        # are optimized for row-major order, and might make row-major order copies of the arrays
        # anyway.
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
                performance. Default to ``np.float16``.
        """
        self.cost = -1
        self.layout = layout
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
    biais: np.ndarray, weights: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """
    As its name suggests. It corresponds to the following equation : Z^i=W^i A^(i-1)-B^i.

    It was initally written to work in the case with one example but it so happens that the same
    function can be re-used without modifiaction in the case with multiple images. Nevertheless,
    no matter the case, all the matrices always have the same dimension : they are 2D-matrices.
    """
    z = values @ weights.T - biais
    return z


def feed(
    multilayer_perceptron: "MultilayerPerceptron",
    example: np.ndarray = None,
    start_layer_rank: int = None,
):
    """
    This function plugs the values of an example (image in this case) on the first layer, and
    computes the new values of all the other nodes up until the last layer. Works in place.

    It was initally written to work in the case with one example but it so happens that the same
    function can be re-used without modifiaction in the case with multiple images. Nevertheless,
    no matter the case, all the matrices always have the same dimension : they are 2D-matrices.

    Args:
        multilayer_perceptron (MultilayerPerceptron): the mutlilayer perceptron to be updated.
        example (np.ndarray): Optional only if layer_rank != None (see below why). The example(s) to
            be fed. Always a 2D-matrix, whose rows correspond to examples.
        start_layer_rank (int): Optional. If a value is provided, the example isn't plugged and the
            values of the multilayer perceptron are simply recomputed starting from the layer
            ranked start_layer_rank, included. This is useful in the tests of the
            cost_gradient_one_example function, to verify that the partial derivatives w.r.t. the
            values are correct.
    """
    N = len(multilayer_perceptron.layers)

    if start_layer_rank is None:
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

    else:
        for i in range(start_layer_rank, N):
            z = pre_regularization_value(
                biais=multilayer_perceptron.variables[i][0],
                weights=multilayer_perceptron.variables[i][1],
                values=multilayer_perceptron.variables[i - 1][2],
            )
            value = multilayer_perceptron.layers[i].activation_fct(z)
            multilayer_perceptron.variables[i][2] = value


def expected_values_last_layer(label: int) -> np.ndarray:
    """
    A small utility which takes as an argument a label (ex : 2) and returns the array of the
    expected values on the last layer of the neural network (ex : np.array([[0,0,1,0,0,0,0,0,0,0]])
    ). Used in dataset_reader.
    """
    _expected_values_last_layers = np.zeros(shape=(1, 10))
    _expected_values_last_layers[0][label] = 1
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
        _cost_gradient_one_example (np.ndarray): ibid.
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
    multilayer_perceptron: MultilayerPerceptron, training_set: list
) -> list:
    """
    Compute the gradient of the cost of the multilayer perceptron with respect to the training set.

    Args:
        multilayer perceptron (MultilayerPerceptron): the neural network to be trained.
        training_set (list): the training set.

    Returns:
        _cost_gradient (list): ibid. Same size as the multilayer perceptron parameters.
    """
    _cost_gradients_one_example = []
    for labeled_example in training_set:
        _cost_gradient_one_example = cost_gradient_one_example(
            multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example
        )
        _cost_gradients_one_example.append(_cost_gradient_one_example)

    N = len(multilayer_perceptron.layers)
    nb_tests = len(training_set)
    _cost_gradient = []
    for i in range(N):
        # bais
        dcdb = np.average(
            np.array([_cost_gradients_one_example[s][i][0] for s in range(nb_tests)]),
            axis=0,
        )

        # weights
        dcdw = np.average(
            np.array([_cost_gradients_one_example[s][i][1] for s in range(nb_tests)]),
            axis=0,
        )

        # values
        dcdv = np.average(
            np.array([_cost_gradients_one_example[s][i][2] for s in range(nb_tests)]),
            axis=0,
        )

        partial_derivative_wrt_layer = [dcdb, dcdw, dcdv]
        _cost_gradient.append(partial_derivative_wrt_layer)

    return _cost_gradient


def cost_one_example(
    multilayer_perceptron: MultilayerPerceptron,
    labeled_example: list,
    start_layer_rank: int = None,
):
    """Make the multilayer perceptron guess for the provided example and computes the square of the
    L^2 distance of its output to the label provided. Note that this function works in place, it
    modifies the values of the neural network (not its parameters). Alternatively, if a layer rank
    is provided, the function doesn't use the example provided and simply recomputes the values of
    the neural network starting from the starting layer rank provided, and computes the cost (with
    respect to the label provided, thus the label is still required).

    Args:
        multilayer_perceptron (MultilayerPerceptron)
        labeled_example (list): If a start_layer_rank is provided, of the form [label]. Else, of
            the form [label, example]. "label" and "example" are 2D row arrays of size
            respectively the size of the last layer and the size of the first layer. Note that
            label is not a str, or an int, it has to be directly the output on the last layer that
            the neural network should have. A small function tailored to each problem should be
            used to turn a label in natural language into the corresponding array of output values.
        start_layer_rank (int). Optional. Cf function description.

    Returns:
        _cost_one_example (float): with the highest precision between the one of the
            parameters of multilayer_perceptron.variables and the labeled_example. ?
    """
    if start_layer_rank is None:
        feed(multilayer_perceptron=multilayer_perceptron, example=labeled_example[1])
    else:
        feed(
            multilayer_perceptron=multilayer_perceptron,
            start_layer_rank=start_layer_rank,
        )

    N = len(multilayer_perceptron.variables)
    return (
        np.linalg.norm(labeled_example[0] - multilayer_perceptron.variables[N - 1][2])
        ** 2
    )


def cost(
    multilayer_perceptron: MultilayerPerceptron,
    training_set: list[list[np.ndarray, np.ndarray]],
):
    """
    Returns the average cost of the multilayer perceptron provided over the training set provided.

    Args:
        multilayer_perceptron (MultilayerPerceptron)
        training_set (list): list of the form [labeled_examples] where labeled_example = [label,
            example], label and example being np.ndarrays.

    Returns:
        _cost (float)
    """
    _costs = [
        cost_one_example(
            multilayer_perceptron=multilayer_perceptron,
            labeled_example=labeled_example,
        )
        for labeled_example in training_set
    ]
    _cost = np.average(_costs)
    return _cost


def learning_one_step(
    multilayer_perceptron: MultilayerPerceptron,
    training_set: list[list[np.ndarray, np.ndarray]],
    eta: float = 1,
    inertia: bool = True,
    returns_cost_gradient: bool = False,
    **kwargs,
):
    """Compute the gradient of the cost of the multilayer perceptron with respect to the
    training set, then modify the parameters of the neural network in the opposite direction
    to the gradient. Work in-place.

    Args:
        multilayer_perceptron (MultilayerPerceptron): the neural network to be trained.
        training_set (list[list[np.ndarray, np.ndarray]]): the training set.
        eta (float): "learning boldness/nudge strenght factor". Hyperparameter. Parameters are
            nudged by -eta * np.average(layout) * grad C. Default to 1e-02.
        inertia (bool): Wether at each step in gradient descent the parameters should also be
            updated along the gradient of the step before (as if the parameters were a ball rolling
            down the cost landscape). Default to True. In this case, the inertia term in the update
            of the parameters is inertia_strength * (the previous update).
        returns_cost_gradient (bool): Default to false. Used in learning to computes the inner
            products of consecutives gradients.

    Kwarg:
        inertia_strength (float): If inertia=True.
        previous_cost_gradient (list): to be used if inertia=True.
    """
    _cost_gradient = cost_gradient(
        multilayer_perceptron=multilayer_perceptron, training_set=training_set
    )
    N = len(multilayer_perceptron.layers)
    layout = multilayer_perceptron.layout
    # Because of the inhomogenous shape of multilayer_perceptron.variables, we have to use for
    # loops up until we manipulate arrays.
    for i in range(N):
        for variable in range(2):
            multilayer_perceptron.variables[i][variable] += (
                -eta * np.average(layout) * _cost_gradient[i][variable]
            )
            if inertia:
                multilayer_perceptron.variables[i][variable] += (
                    -eta
                    * np.average(layout)
                    * kwargs["previous_cost_gradient"][i][variable]
                )
    multilayer_perceptron.cost = cost(multilayer_perceptron, training_set)
    if returns_cost_gradient:
        return _cost_gradient


def flatten_cost_gradient(_cost_gradient: list) -> np.ndarray:
    """
    cost_gradient have a nested structure (designed for efficency of computation). It is a list of
    lists, one for each layer. Each list is made of three np.ndarrays. The first one represents the
    biaises, the second one the weights and the last ones the values. This function flattens
    cost_gradient, remove the unnecessaries partial derivatives w.r.t to the values of all layers
    and the biaises and weights of the first layers (all of which are irrelevant because they change
    depending on the last image fed to the model in one case, and because they are NaN in the other).
    """
    flattened_grad = np.array([])
    for i in range(1, len(_cost_gradient)):
        # biaises
        flattened_grad = np.concatenate((flattened_grad, _cost_gradient[i][0][0]))

        # weights
        dcdw = np.reshape(a=_cost_gradient[i][1], newshape=-1)
        flattened_grad = np.concatenate((flattened_grad, dcdw))

    return flattened_grad


def consecutive_gradients_cosine(cost_gradient_1: list, cost_gradient_2: list) -> float:
    """
    A small utility to declutter learning. Computes the cosine of the angle of the two provided
    costs gradients. Before that, flatten them to a 1D-array, delete the partial derivative w.r.t.
    the node values and the biaises and weights of the first layer (all of which are irrelevant).
    """
    flattened_grad_1 = flatten_cost_gradient(_cost_gradient=cost_gradient_1)
    flattened_grad_2 = flatten_cost_gradient(_cost_gradient=cost_gradient_2)

    inner_product = float(
        np.inner(flattened_grad_1, flattened_grad_2)
    )  # The result of np.inner is a np.float.
    cosine_of_angle = inner_product / (
        np.linalg.norm(flattened_grad_1) * np.linalg.norm(flattened_grad_2)
    )
    return cosine_of_angle


def learning(
    multilayer_perceptron: MultilayerPerceptron,
    training_set: list[list[np.ndarray, np.ndarray]],
    eta: float,
    stop_condition: str = "fixed_steps_number",
    stochastic: bool = True,
    inertia: bool = True,
    metrics_to_track: list[str] = [""],
    **kwargs,
):
    """
    Train the multilayer perceptron provided on the training set provided. Uses gradient descent
    and retropropagation. Works in-place. The stop condition used is explained below.

    Args:
        multilayer_perceptron (MultilayerPerceptron): the neural network to be trained.
        training_set (list): the training set to learn from.
        stop_condition (str): Either "fixed_steps_number" or "stagnation". If = fixed_steps_numbers
            , a fixed number of learning steps steps_number are undergone. If = stagnation,
            learning steps are undergone until the training cost doesn't variate more than
            stagnation_espilon over max_stagnation_steps.
        eta (float): "learning boldness/nudge strength". Hyperparameter. At each step of the
            gradient descent, parameters are nudged by -eta * ...() * grad C.
        stochastic (bool): Wether the gradient descent ought to be stochastic or not. True by
            default. The number of examples selected at each step is step_training_size.
        inertia (bool): Wether at each step in gradient descent the parameters should also be
            updated along the gradient of the step before (as if the parameters were a ball rolling
            down the cost landscape). Default to True. In this case, the inertia term in the update
            of the parameters is inertia_strength * (the previous update).
        metrics_to_track (list): The list of metrics to track during training. Useful for
            exploration. Includes "training_costs" (cost on the set used for training at each step)
            , "accuracies" (on the test set test_set provided as a kwargs), "gradients_norms" and
            "consecutive_gradients_cosines". The two first add significant time (~50%) while the
            last two are negligeable. Defaults to an empty list.

    **Kwargs:
        steps_number (int): If stop_condition=fixed_steps_number. Number of learning steps to
            undergo.
        stagnation_epsilon (float): If stop_condition=stagnation. Maximal variation of the cost
            over stagnation_steps below wich a local minimum is considered to have been found.
            No default value because depends too much on the problem.
        max_stagnation_steps (int): If stop_condition=stagnation. Number of learning steps upon
            which the stagnation of learning is compared to the stagnation threshold epsilon.
            Default to 3.
        step_training_size (int): If stochastic=True. Size of the subset of the training set used
            for learning at each step. Defaults to 1000.
        inertia_strength (float): If inertia=True. Default to 0.1.
        test_set (list): If computes_accuracies_during_training, returns the list of accuracies of
            the model on this set after each learning step.

    Returns:
        tracked_metrics (dict): of the form {"training_costs" : list, "accuracies": list,
            "gradients_norms", "consecutive_gradients_angles"}. Only the requested metrics are
            included. If is empty, the function doesn't return it.
    """
    # Defaults values of kwargs (maybe a cleaner way of organising the code ?)
    default_steps_number = 100
    default_max_stagnation_steps = 3
    default_step_training_size = 1000
    default_inertia_strength = 0.1

    tracked_metrics = {}

    if inertia:
        previous_cost_gradient = [
            [
                np.zeros(shape=multilayer_perceptron.variables[i][variable].shape)
                for variable in range(3)
            ]
            for i in range(len(multilayer_perceptron.layout))
        ]
    # Cette section est améliorable, clarifiable je pense. For ... in metrics_to_computes: ?
    if "training_costs" in metrics_to_track:
        costs_during_training = [
            cost(multilayer_perceptron, training_set)
        ]  # Starts before the first learning step (at =(after) the 0-th learning step)
    if "accuracies" in metrics_to_track:
        accuracies_during_training = [
            accuracy(multilayer_perceptron, kwargs["test_set"])
        ]  # Starts before the first learning step (at the 0-th learning step)
    if "gradients_norms" in metrics_to_track:
        gradients_norms_during_training = []  # Starts at the first learning step
    if "consecutive_gradients_cosines" in metrics_to_track:
        consecutive_gradients_cosines_during_training = (
            []
        )  # Starts at the first learning step
        previous_cost_gradient = cost_gradient_one_example(
            multilayer_perceptron=multilayer_perceptron, labeled_example=training_set[0]
        )

    if stop_condition == "fixed_steps_number":
        for _ in tqdm.tqdm(
            range(kwargs.get("steps_number", default_steps_number))
        ):  # tqdm here adds a progress bar, nothing more.
            if stochastic:
                training_set_at_this_step = random.sample(
                    population=training_set,
                    k=kwargs.get("step_training_size", default_step_training_size),
                )
            else:
                training_set_at_this_step = training_set

            if inertia:
                _cost_gradient = learning_one_step(
                    multilayer_perceptron=multilayer_perceptron,
                    training_set=training_set_at_this_step,
                    eta=eta,
                    inertia=True,
                    returns_cost_gradient=True,
                    previous_cost_gradient=previous_cost_gradient,
                    inertia_strenght=kwargs.get(
                        "inertia_strength", default_inertia_strength
                    ),
                )
                previous_cost_gradient = _cost_gradient
            else:
                _cost_gradient = learning_one_step(
                    multilayer_perceptron=multilayer_perceptron,
                    training_set=training_set_at_this_step,
                    eta=eta,
                    inertia=False,
                    returns_cost_gradient=(
                        "consecutive_gradients_cosines" in metrics_to_track
                    ),
                )  # Note that if computes_consecutive_gradients.. = False, cost_gradient = None. I
            # chose this over introducing yet again another if condition.

            if "training_costs" in metrics_to_track:
                costs_during_training.append(multilayer_perceptron.cost)
            if "accuracies" in metrics_to_track:
                accuracies_during_training.append(
                    accuracy(multilayer_perceptron, kwargs["test_set"])
                )
            if "gradients_norms" in metrics_to_track:
                gradients_norms_during_training.append(
                    np.linalg.norm(flatten_cost_gradient(_cost_gradient))
                )
            if "consecutive_gradients_cosines" in metrics_to_track:
                _consecutive_gradients_cosine = consecutive_gradients_cosine(
                    cost_gradient_1=previous_cost_gradient,
                    cost_gradient_2=_cost_gradient,
                )
                consecutive_gradients_cosines_during_training.append(
                    _consecutive_gradients_cosine
                )
                previous_cost_gradient = _cost_gradient

    elif stop_condition == "stagnation":
        # We define the counter of current steps where the cost is in [cost_at_start +/-
        # stagnation_epsilon] where cost_at_start is the cost at the time the counter was started. The
        # counter restarts when the cost escapes this interval.
        costs_during_training = [cost(multilayer_perceptron, training_set)]
        cost_at_start = costs_during_training[0]
        counter = 0
        while counter < kwargs.get(
            "max_stagnation_steps", default_max_stagnation_steps
        ):
            learning_one_step(
                multilayer_perceptron=multilayer_perceptron,
                training_set=training_set,
                eta=eta,
                inertia=False,
            )

            costs_during_training.append(multilayer_perceptron.cost)

            if (
                np.abs(cost_at_start - multilayer_perceptron.cost)
                < kwargs["stagnation_epsilon"]
            ):
                counter += 1
            else:
                cost_at_start = multilayer_perceptron.cost
                counter = 0

    if "training_costs" in metrics_to_track:
        tracked_metrics["training_costs"] = costs_during_training
    if "accuracies" in metrics_to_track:
        tracked_metrics["accuracies"] = accuracies_during_training
    if "gradients_norms" in metrics_to_track:
        tracked_metrics["gradients_norms"] = gradients_norms_during_training
    if "consecutive_gradients_cosines" in metrics_to_track:
        tracked_metrics["consecutive_gradients_cosines"] = (
            consecutive_gradients_cosines_during_training
        )

    if tracked_metrics != {}:
        return tracked_metrics
    return None


def prediction_result(
    multilayer_perceptron: MultilayerPerceptron,
    labeled_example: list[np.ndarray, np.ndarray],
) -> bool | np.ndarray:
    """
    A small utility to declutter accuracy. Have the model guess for the example. Returns the 0-1
    distance between its guess and the correct label.

    This is surprinsingly also almost without modifications to the initial code a vectorized
    equivalent appliabke in parallel to many different labeled examples. Formally, it  can also
    accepts as argument a batch of labeled examples stiched in an orthogonal dimension (which make
    a new matrix) and returns a 1D matrix of prediction results instead of one.

    Args:
        multilayer_perceptron (MultilayerPerceptron)
        labeled_example (list): either of the form [label, example] where label and example are 2D
            np.ndarray, or of the form [labels_batch, examples_batch] where both are also 2D
            np.ndarray.

    Returns:
        _prediction_result (bool | np.ndarray)
    """
    feed(multilayer_perceptron=multilayer_perceptron, example=labeled_example[1])
    # Note that precising axis=1 is optional in the case with one example, but it is added so that
    # the function can be reused in the multiple-examples case. In the latter case, guess and truth
    # are 1D-arrays, not floats.
    guess = np.argmax(multilayer_perceptron.variables[-1][2], axis=1)
    truth = np.argmax(labeled_example[0], axis=1)
    return guess == truth


def accuracy(
    multilayer_perceptron: MultilayerPerceptron,
    test_set: list[list[np.ndarray, np.ndarray]],
) -> float:
    """
    This function evaluates the trained (or untrained) model on a test set. It is similar to cost
    with a different distance, the 0-1 distance rather than the L2-norm. The test set has to be in
    the form of batches of examples, either containing one example (as previously in my code, what
    is refered to elsewhere as the old format : [labeled_examples] where labeled_example = [label,
    example], label and example being 2D np.arrays with one row), or multiple examples (in the same
    format, but now understood as [labeled_examples_batches] where labeled_examples_batch = [
    labels_batch, examples_batch], labels_batch and examples_batch being 2D np.ndarray with multiple
    rows).

    The hope is to accelerate computation. More precisely, the test_set is not entirely vectorized
    because it would probably lead to matrices too big to be handled by my machine. Rather, I've
    added a degree of freedom, which is the batch_size. The test set is divided in batches that are
    vectorized. The batch size can be configured (once for all) depending on the optimal size of
    matrices for one's machine.

    Args:
        multilayer_perceptron (MultilayerPerceptron)
        test_set (list): list of the form [labeled_examples_batches] where labeled_example_batch =
        [labels_batch, examples_batch], labels_batch and examples_batch being 2D np.ndarrays with
        one or several rows.

    Returns:
        _accuracy (np.float): the proportion of sucessful guesses.
    """
    # np.average is used two times because trying to average over an unique array breaks when the
    # last batch is not full (because this array can't be easily created). I beleive this is the
    # fastest way but I might be wrong.
    batches_accuracies = [
        np.average(
            prediction_result(
                multilayer_perceptron=multilayer_perceptron,
                labeled_example=labeled_examples_batch,
            )
        )
        for labeled_examples_batch in test_set
    ]
    _accuracy = np.average(batches_accuracies)
    return _accuracy


def vectorize_learning_set(
    learning_set: list[list[np.ndarray, np.ndarray]], max_batch_size: int
) -> list[list[np.ndarray, np.ndarray]]:
    """
    This function divides sets (for training or test) into batches, vectorize the batches, and
    return the set in this form.

    Args:
        learning_set (list[list]) : list of the form [labeled_examples] where labeled_example =
            [label, example], label and example being np.ndarrays.
        max_batch_size (int) : Maximum size of the batches. Deduced from the maximum amount of
            usable memory. All of the batches have this size except eventually the last one.

    Returns:
        vectorized_set (list[list[np.ndarray, np.ndarray]]) : list of the form
        [batches_of_labeled_examples] where batch_of_labeled_examples = [batch_of_labels,
        batch_of_examples]. batch_of_labels and batch_of_examples are two-dimensional np.ndarrays
        (labels and examples respectively stacked over a new dimension).
    """
    if max_batch_size > len(learning_set):
        raise ValueError(
            "The maximum batch size has to be smaller than the size of the learning set"
        )
    vectorized_set = []
    nb_full_batches = len(learning_set) // max_batch_size
    label_length = len(learning_set[0][0][0])
    example_length = len(learning_set[0][1][0])
    for i in range(nb_full_batches):

        labels_to_be_batched = np.array(
            [
                learning_set[j][0][0]
                for j in range(i * max_batch_size, (i + 1) * max_batch_size)
            ]
        )
        labels_batch = np.resize(
            a=labels_to_be_batched, new_shape=(max_batch_size, label_length)
        )
        examples_to_be_batched = np.array(
            [
                learning_set[j][1][0]
                for j in range(i * max_batch_size, (i + 1) * max_batch_size)
            ]
        )
        examples_batch = np.resize(
            a=examples_to_be_batched, new_shape=(max_batch_size, example_length)
        )

        vectorized_set.append([labels_batch, examples_batch])

    if len(learning_set) % max_batch_size != 0:
        last_labels_to_be_batched = np.array(
            [
                learning_set[j][0][0]
                for j in range(nb_full_batches * max_batch_size, len(learning_set))
            ]
        )
        last_labels_batch = np.resize(
            a=last_labels_to_be_batched,
            new_shape=(len(learning_set) % max_batch_size, label_length),
        )
        last_examples_to_be_batched = np.array(
            [
                learning_set[j][1][0]
                for j in range(nb_full_batches * max_batch_size, len(learning_set))
            ]
        )
        last_examples_batch = np.resize(
            a=last_examples_to_be_batched,
            new_shape=(len(learning_set) % max_batch_size, example_length),
        )
        vectorized_set.append([last_labels_batch, last_examples_batch])

    return vectorized_set


def save_object(obj, name: str):
    """
    This function saves a multilayer perceptron to the "saved_objects" directory with the name
    "name". /!\\ Beware, this function is meant to be used only inside playground.ipynb, as it
    uses a relative path.
    """
    pickle.dump(
        obj=obj,
        file=open(file=f"saved_objects/{name}.pkl", mode="wb"),
    )


def load_object(name: str):
    """
    This function loads a multilayer perceptron from the "saved_objects" directory with the name
    "name". /!\ Beware, this function is meant to be used only inside playground.ipynb, as it
    uses a relative path.
    """
    return pickle.load(file=open(file=f"saved_objects/{name}.pkl", mode="rb"))


def main():
    # Initialization of the logging class.
    logging.basicConfig(level="DEBUG")

    return 1


if __name__ == "__main__":
    main()
