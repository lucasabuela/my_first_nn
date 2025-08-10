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
        self.layer_count = len(layout)
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
    mlp: "MultilayerPerceptron",
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
        mlp (MultilayerPerceptron): the mutlilayer perceptron to be updated.
        example (np.ndarray): Optional only if layer_rank != None (see below why). The example(s)
            to be fed. Always a 2D-matrix, whose rows correspond to examples.
        start_layer_rank (int): Optional. If a value is provided, the example isn't plugged and the
            values of the multilayer perceptron are simply recomputed starting from the layer
            ranked start_layer_rank, included. This is useful in the tests of the cost_gradients_
            on_examples_of_one_batch function, to verify that the partial derivatives w.r.t. the
            values are correct.
    """
    N = mlp.layer_count

    if start_layer_rank is None:
        # Plugs the value of the example on the first layer.
        mlp.variables[0][2] = example

        for i in range(1, N):
            z = pre_regularization_value(
                biais=mlp.variables[i][0],
                weights=mlp.variables[i][1],
                values=mlp.variables[i - 1][2],
            )
            value = mlp.layers[i].activation_fct(z)
            mlp.variables[i][2] = value

    else:
        for i in range(start_layer_rank, N):
            z = pre_regularization_value(
                biais=mlp.variables[i][0],
                weights=mlp.variables[i][1],
                values=mlp.variables[i - 1][2],
            )
            value = mlp.layers[i].activation_fct(z)
            mlp.variables[i][2] = value


def expected_values_last_layer(label: int) -> np.ndarray:
    """
    A small utility which takes as an argument a label (ex : 2) and returns the array of the
    expected values on the last layer of the neural network (ex : np.array([[0,0,1,0,0,0,0,0,0,0]])
    ). Used in dataset_reader.
    """
    _expected_values_last_layers = np.zeros(shape=(1, 10))
    _expected_values_last_layers[0][label] = 1
    return _expected_values_last_layers


def cost_gradients_on_examples_of_one_batch(
    mlp: MultilayerPerceptron, labeled_examples_batch: list
):
    """
    Compute the gradients of the cost of the multilayer perceptron on the labeled examples of a
    batch of the training set.

    Args:
        mlp (MultilayerPerceptron): the neural network whose gradient of the cost with respect to
            the labeled example we want to calculate.
        labeled_examples_batch (list): of the form [labels_batch, examples_batch]. "labels_batch"
            and "examples_batch" are 2D row arrays of size respectively (nbs of examples in the
            batch * size of the last layer) and (nbs of examples in the batch *size of the first
            layer). Note that label is not a str, or an int, it has to be directly the output on
            the last layer that the neural network should have.

    Returns:
        _cost_gradients_on_examples_of_one_batch (list): each element of this list is is a list of
            the form [dcd𝐁^𝐢, dcd𝐖^𝐢, dcd𝐀^𝐢], where dcd𝐁^𝐢 and dcd𝐀^𝐢 are 2D ndarray of shape (
            batch_size, layer[i].size), and dcd𝐖^𝐢 a 3D ndarray of shape (batch_size, layer[i].size
            , layer[i-1].size).
    """
    # First, we feed the neural network with the example.
    feed(mlp=mlp, example=labeled_examples_batch[1])

    # We'll have to compute the partial derivatives w.r.t all the variables, i.e. w.r.t the
    # weights, the biaises and also the values. Thus we define :
    partial_derivatives = deepcopy(mlp.variables)

    N = mlp.layer_count

    # Let's start by tackling the special case of the partial derivatives with regard to the values
    # of the last layer. The formula is (𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐀^(𝐍−𝟏) )=−2 ∗(𝐲−𝐀^(𝐍−𝟏)):
    partial_derivatives[N - 1][2] = -2 * (
        labeled_examples_batch[0] - mlp.variables[N - 1][2]
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
        # (𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐁^𝐢 )=−(𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐀^𝐢 )⊙𝑓_𝑖^′ (𝐙^𝐢 ):
        partial_derivatives[i][0] = -(
            partial_derivatives[i][2]
            * mlp.layers[i].activation_fct_derivative(
                pre_regularization_value(
                    biais=mlp.variables[i][0],
                    weights=mlp.variables[i][1],
                    values=mlp.variables[i - 1][2],
                )
            )
        )

        # Computation of the gradients w.r.t. the w_{j,k}^{i}. We use a relation between these and
        # the gradients w.r.t. the biases to accelerate the compute. The formula is:
        # (𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐖^𝐢 )=−((𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐁^𝑖 ))^(𝑅_1 )∗〖𝐀^(𝐢−𝟏)〗^(𝑅_2 ), where R1 and R2
        # represent certain rotations in space and * the usual dot product in N-d (sum over the
        # second-to-last dimension of the first entry and the last dimension of the second entry).
        # To see a diagram of this operation, see the docs. Intuitively, it's the operation for the
        # case with one example per batch, stacked over a new dimension.
        batch_size = len(labeled_examples_batch[0])
        partial_derivatives[i][1] = -np.reshape(
            a=partial_derivatives[i][0],
            newshape=(batch_size, mlp.layers[i].size, 1),
        ) @ np.reshape(
            a=mlp.variables[i - 1][2],
            newshape=(batch_size, 1, mlp.layers[i - 1].size),
        )

        # Computation of the gradients w.r.t. the a_{k}^{i-1} with the formula :
        # ((𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐀^(𝐢−𝟏) )=−(𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐁^𝐢 )∗𝑊^𝑖. Same note as above.
        partial_derivatives[i - 1][2] = -(
            partial_derivatives[i][0] @ mlp.variables[i][1]
        )

    _cost_gradients_on_examples_of_one_batch = partial_derivatives

    return _cost_gradients_on_examples_of_one_batch


def cost_gradient_one_batch(_cost_gradients_on_examples_of_one_batch: list) -> list:
    """
    This function averages the gradients of the cost on the labeled_examples of a batch to produce
    the gradient of the cost on this batch.

    Args:
        cost_gradients_on_examples_of_one_batch (list): each element of this list is is a list of
            the form [dcd𝐁^𝐢, dcd𝐖^𝐢, dcd𝐀^𝐢], where dcd𝐁^𝐢 and dcd𝐀^𝐢 are 2D ndarray of shape (
            batch_size, layer[i].size), and dcd𝐖^𝐢 a 3D ndarray of shape (batch_size, layer[i].size
            , layer[i-1].size).

    Returns:
        _cost_gradient_one_batch (list): each element of this list is is a list of the form [
            dcd𝐁^(𝐢,𝒃), dcd𝐖^(𝐢,𝒃), dcd𝐀^(𝐢,𝐛)], where dcd𝐁^(𝐢,𝒃) and dcd𝐀^(𝐢,𝐛) are 2D ndarray of
            shape (1, layer[i].size), and dcd𝐖^(𝐢,𝒃) a 2D ndarray of shape (layer[i].size,
            layer[i-1].size).
    """
    N = len(_cost_gradients_on_examples_of_one_batch)
    _cost_gradient_one_batch = []
    for i in range(N):
        _partial_derivative_wrt_layer_on_one_batch = []
        for variable in range(3):
            dcdvar = np.average(
                a=_cost_gradients_on_examples_of_one_batch[i][variable], axis=0
            )
            # dcd𝐖^𝐢 is 3D so keepdims can't be used in the average up above, but dcd𝐁^𝐢 and
            # dcd𝐀^𝐢 have to end up 2D so we add np.atleast_2d.
            _partial_derivative_wrt_layer_on_one_batch.append(np.atleast_2d(dcdvar))

        _cost_gradient_one_batch.append(_partial_derivative_wrt_layer_on_one_batch)

    return _cost_gradient_one_batch


def cost_gradient(mlp: MultilayerPerceptron, training_set: list) -> list:
    """
    Compute the gradient of the cost of the multilayer perceptron with respect to the training set.

    Args:
        mlp (MultilayerPerceptron): the neural network to be trained.
        training_set (list): the training set.

    Returns:
        _cost_gradient (list): ibid. Same size as the multilayer perceptron parameters.
    """
    _cost_gradients_one_batch = [
        cost_gradient_one_batch(
            cost_gradients_on_examples_of_one_batch(
                mlp=mlp,
                labeled_examples_batch=batch,
            )
        )
        for batch in training_set
    ]

    # The cost gradient on the training set is computed by averaging the cost gradients on the
    # batches, weighted by the number of labeled examples they include.
    N = mlp.layer_count
    batch_count = len(training_set)
    _cost_gradient = []
    batch_weights = [len(batch[0]) for batch in training_set]
    for i in range(N):
        partial_derivative_wrt_layer = []
        for variable in range(3):
            dcdvar = np.average(
                a=np.array(
                    [
                        _cost_gradients_one_batch[batch][i][variable]
                        for batch in range(batch_count)
                    ]
                ),
                weights=batch_weights,
                axis=0,
            )
            partial_derivative_wrt_layer.append(dcdvar)
        _cost_gradient.append(partial_derivative_wrt_layer)

    return _cost_gradient


def costs_one_batch(
    mlp: MultilayerPerceptron,
    labeled_examples_batch: list,
    start_layer_rank: int = None,
):
    """Make the multilayer perceptron guess for the provided examples and computes the square of
    the L^2 distance of its output to the label provided. Note that this function works in place,
    it modifies the values of the neural network (not its parameters). Alternatively, if a layer
    rank is provided, the function doesn't use the example provided and simply recomputes the
    values of the neural network starting from the starting layer rank provided, and computes the
    costs (with respect to the label provided, thus the label is still required).

    Args:
        mlp (MultilayerPerceptron)
        labeled_examples_batch (list): If a start_layer_rank is provided, of the form [label]. Else
            , of the form [labels_batch, examples_batch]. "labels_batch" and "examples_batch" are 2D
            row arrays of size respectively (nbs of examples in the batch * size of the last layer)
            and (nbs of examples in the batch *size of the first layer). Note that label is not a
            str, or an int, it has to be directly the output on the last layer that the neural
            network should have. A small function tailored to each problem should be used to turn
            a label in natural language into the corresponding array of output values.
        start_layer_rank (int). Optional. Cf function description.

    Returns:
        _costs_one_batch (np.ndarray): of dimension 1 (cost for each labeled example of the batch).
    """
    if start_layer_rank is None:
        feed(
            mlp=mlp,
            example=labeled_examples_batch[1],
        )
    else:
        feed(
            mlp=mlp,
            start_layer_rank=start_layer_rank,
        )

    N = mlp.layer_count
    # Note that precising axis=1 is optional in the case with one example, but it is added so that
    # the function can be reused in the multiple-examples case.
    _costs = (
        np.linalg.norm(
            labeled_examples_batch[0] - mlp.variables[N - 1][2],
            axis=1,
        )
        ** 2
    )
    return _costs


def cost(
    mlp: MultilayerPerceptron,
    training_set: list[list[np.ndarray, np.ndarray]],
):
    """
    Returns the average cost of the multilayer perceptron provided over the training set provided.

    Args:
        mlp (MultilayerPerceptron)
        training_set (list): list of the form [labeled_examples] where labeled_example = [label,
            example], label and example being 2D np.ndarrays. Alternatively, of the equivalent
            form [labeled_examples_batches] where labeled_examples_batch = [labels_batch,
            examples_batch], labels_batch and examples_batch being 2D np.arrays.

    Returns:
        _cost (float)
    """
    _costs = [
        np.average(
            costs_one_batch(
                mlp=mlp,
                labeled_examples_batch=labeled_examples_batch,
            )
        )
        for labeled_examples_batch in training_set
    ]
    weights = [
        len(labeled_examples_batch[0]) for labeled_examples_batch in training_set
    ]
    _cost = np.average(a=_costs, weights=weights)
    return _cost


def learning_one_step(
    mlp: MultilayerPerceptron,
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
        mlp (MultilayerPerceptron): the neural network to be trained.
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
    _cost_gradient = cost_gradient(mlp=mlp, training_set=training_set)
    N = mlp.layer_count
    layout = mlp.layout
    # Because of the inhomogenous shape of mlp.variables, we have to use for loops up until we
    # manipulate arrays.
    for i in range(N):
        for variable in range(2):
            mlp.variables[i][variable] += (
                -eta * np.average(layout) * _cost_gradient[i][variable]
            )
            if inertia:
                mlp.variables[i][variable] += (
                    -eta
                    * np.average(layout)
                    * kwargs["previous_cost_gradient"][i][variable]
                )
    mlp.cost = cost(mlp, training_set)
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
    mlp: MultilayerPerceptron,
    training_set: list[list[np.ndarray, np.ndarray]],
    eta: float,
    stop_condition: str = "fixed_step_count",
    stochastic: bool = True,
    inertia: bool = True,
    metrics_to_track: list[str] = [""],
    **kwargs,
):
    """
    Train the multilayer perceptron provided on the training set provided. Uses gradient descent
    and retropropagation. Works in-place. The stop condition used is explained below.

    Args:
        mlp (MultilayerPerceptron): the neural network to be trained.
        training_set (list): the training set to learn from.
        stop_condition (str): Either "fixed_step_count" or "stagnation". If = fixed_step_count
            , a fixed number of learning steps step_count are undergone. If = stagnation,
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
        step_count (int): If stop_condition=fixed_step_count. Number of learning steps to
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
    default_step_count = 100
    default_max_stagnation_steps = 3
    default_step_training_size = 1000
    default_inertia_strength = 0.1

    tracked_metrics = {}

    if inertia:
        previous_cost_gradient = [
            [np.zeros(shape=mlp.variables[i][variable].shape) for variable in range(3)]
            for i in range(len(mlp.layout))
        ]
    # Cette section est améliorable, clarifiable je pense. For ... in metrics_to_computes: ?
    if "training_costs" in metrics_to_track:
        costs_during_training = [
            cost(mlp, training_set)
        ]  # Starts before the first learning step (at =(after) the 0-th learning step)
    if "accuracies" in metrics_to_track:
        accuracies_during_training = [
            accuracy(mlp, kwargs["test_set"])
        ]  # Starts before the first learning step (at the 0-th learning step)
    if "gradients_norms" in metrics_to_track:
        gradients_norms_during_training = []  # Starts at the first learning step
    if "consecutive_gradients_cosines" in metrics_to_track:
        consecutive_gradients_cosines_during_training = (
            []
        )  # Starts at the first learning step
        previous_cost_gradient = [
            [np.zeros(shape=mlp.variables[i][variable].shape) for variable in range(3)]
            for i in range(len(mlp.layout))
        ]

    if stop_condition == "fixed_step_count":
        for _ in tqdm.tqdm(
            range(kwargs.get("step_count", default_step_count))
        ):  # tqdm here adds a progress bar, nothing more.
            if stochastic:
                batch_size = len(training_set[0][0])
                training_set_at_this_step = random.sample(
                    population=training_set,
                    k=int(
                        kwargs.get("step_training_size", default_step_training_size)
                        / batch_size
                    ),
                )
            else:
                training_set_at_this_step = training_set

            if inertia:
                _cost_gradient = learning_one_step(
                    mlp=mlp,
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
                    mlp=mlp,
                    training_set=training_set_at_this_step,
                    eta=eta,
                    inertia=False,
                    returns_cost_gradient=(
                        "consecutive_gradients_cosines" in metrics_to_track
                    ),
                )  # Note that if computes_consecutive_gradients.. = False, cost_gradient = None. I
            # chose this over introducing yet again another if condition.

            if "training_costs" in metrics_to_track:
                costs_during_training.append(mlp.cost)
            if "accuracies" in metrics_to_track:
                accuracies_during_training.append(accuracy(mlp, kwargs["test_set"]))
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
        costs_during_training = [cost(mlp, training_set)]
        cost_at_start = costs_during_training[0]
        counter = 0
        while counter < kwargs.get(
            "max_stagnation_steps", default_max_stagnation_steps
        ):
            learning_one_step(
                mlp=mlp,
                training_set=training_set,
                eta=eta,
                inertia=False,
            )

            costs_during_training.append(mlp.cost)

            if np.abs(cost_at_start - mlp.cost) < kwargs["stagnation_epsilon"]:
                counter += 1
            else:
                cost_at_start = mlp.cost
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
    mlp: MultilayerPerceptron,
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
        mlp (MultilayerPerceptron)
        labeled_example (list): either of the form [label, example] where label and example are 2D
            np.ndarray, or of the form [labels_batch, examples_batch] where both are also 2D
            np.ndarray.

    Returns:
        _prediction_result (bool | np.ndarray)
    """
    feed(mlp=mlp, example=labeled_example[1])
    # Note that precising axis=1 is optional in the case with one example, but it is added so that
    # the function can be reused in the multiple-examples case. In the latter case, guess and truth
    # are 1D-arrays, not floats.
    guess = np.argmax(mlp.variables[-1][2], axis=1)
    truth = np.argmax(labeled_example[0], axis=1)
    return guess == truth


def accuracy(
    mlp: MultilayerPerceptron,
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
        mlp (MultilayerPerceptron)
        test_set (list): list of the form [labeled_examples_batches] where labeled_example_batch =
        [labels_batch, examples_batch], labels_batch and examples_batch being 2D np.ndarrays with
        one or several rows.

    Returns:
        _accuracy (np.float): the proportion of sucessful guesses.
    """
    # np.average is used two times because trying to average over an unique array breaks when the
    # last batch is not full (because this array can't be easily created). I beleive this is the
    # fastest way but I might be wrong.
    batch_accuracies = [
        np.average(
            prediction_result(
                mlp=mlp,
                labeled_example=labeled_examples_batch,
            )
        )
        for labeled_examples_batch in test_set
    ]
    batch_weights = [len(batch[0]) for batch in test_set]
    _accuracy = np.average(batch_accuracies, weights=batch_weights)
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
