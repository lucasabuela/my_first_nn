import numpy as np
import script
from script import relu, sigmoid, pre_regularization_value, feed


def test_1():
    a = script.MultilayerPerceptron([3, 2, 1])
    b = a.layers[0]
    c = a.layers[1]
    d = a.layers[2]
    assert b.nodes[0].weights is None
    assert len(c.nodes[0].weights) == 3
    assert len(c.nodes[1].weights) == 3
    assert len(d.nodes[0].weights) == 2


def test_2():
    a = script.MultilayerPerceptron([784, 16, 16, 10])
    b = a.layers[0]
    c = a.layers[1]
    d = a.layers[3]
    assert b.nodes[0].weights is None
    assert len(c.nodes[0].weights) == 784
    assert b.nodes[0].activation_fct is None
    assert c.nodes[0].activation_fct == relu
    assert d.nodes[0].activation_fct == sigmoid
    e = a.variables
    assert len(e) == 4
    assert len(e[0]) == 3
    assert np.array_equal(e[0][0], np.array([None] * 784))
    assert len(e[1][1][15]) == 784


def test_multiply():
    """
    To make sure I understand well the np.multiply function.
    """
    values = [2]
    weights = [1, 2]
    assert np.array_equal(np.multiply(values, weights), np.array([2, 4]))
    values = np.array([2, -2])
    weights = np.array([[1, 2], [3, 4]])
    assert np.array_equal(np.multiply(values, weights), np.array([[2, -4], [6, -8]]))


def test_sum():
    """
    To make sure I understand well the np.sum function.
    """
    A = np.array([[1, 2], [3, 4]])
    assert np.array_equal(np.sum(A, axis=1), np.array([3, 7]))


def test_pre_regularization_value():
    # Cas scalaire (j'ai passé trois heures à chercher l'erreur suivante : je n'avais pas transformé
    # le cas scalaire en "cas vectoriel à 1D").
    biais = np.array([np.random.rand()])
    values = np.random.rand(2)
    weights = np.array([np.random.rand(2)])
    print(values)
    print(weights)
    print(np.multiply(values, weights))
    _pre_regularization_value = np.sum(np.multiply(values, weights), axis=1) - biais
    np.testing.assert_array_equal(
        pre_regularization_value(biais=biais, weights=weights, values=values),
        _pre_regularization_value,
    )
    # Cas vectoriel
    size_rank = 2
    size_previous_rank = 2
    biais = np.random.rand(size_rank)
    values = np.random.rand(size_previous_rank)
    weights = np.random.rand(size_rank, size_previous_rank)
    _pre_regularization_value = np.sum(np.multiply(values, weights), axis=1) - biais
    np.testing.assert_array_equal(
        pre_regularization_value(biais=biais, weights=weights, values=values),
        _pre_regularization_value,
    )


def test_feed():
    multilayer_perceptron = script.MultilayerPerceptron([1])
    example = np.array([1])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    np.testing.assert_array_equal(multilayer_perceptron.variables[0][2], np.array([1]))
    multilayer_perceptron = script.MultilayerPerceptron([2])
    example = np.array([1, 2])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    np.testing.assert_array_equal(
        multilayer_perceptron.variables[0][2], np.array([1, 2])
    )

    multilayer_perceptron = script.MultilayerPerceptron([1, 1])
    example = np.array([1])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2][0]
    w = multilayer_perceptron.variables[1][1][0][0]
    b = multilayer_perceptron.variables[1][0][0]
    value = sigmoid((a * w) - b)
    assert multilayer_perceptron.variables[1][2][0] == value

    multilayer_perceptron = script.MultilayerPerceptron([1, 2])
    example = np.array([1])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value = sigmoid(np.sum(a * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[1][2], value)

    multilayer_perceptron = script.MultilayerPerceptron([2, 1])
    example = np.array([1, 2])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value = sigmoid(np.sum(a * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[1][2], value)

    multilayer_perceptron = script.MultilayerPerceptron([2, 2])
    example = np.array([1, 2])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value = sigmoid(np.sum(a * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[1][2], value)

    multilayer_perceptron = script.MultilayerPerceptron([1, 1, 1])
    example = np.array([1])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = multilayer_perceptron.variables[2][1]
    b = multilayer_perceptron.variables[2][0]
    value_2 = sigmoid(np.sum(value_1 * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[2][2], value_2)

    multilayer_perceptron = script.MultilayerPerceptron([2, 2, 2])
    example = np.random.rand(2)
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = multilayer_perceptron.variables[2][1]
    b = multilayer_perceptron.variables[2][0]
    value_2 = sigmoid(np.sum(value_1 * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[2][2], value_2)

    multilayer_perceptron = script.MultilayerPerceptron([784, 16, 16])
    example = np.random.rand(784)
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = multilayer_perceptron.variables[2][1]
    b = multilayer_perceptron.variables[2][0]
    value_2 = sigmoid(np.sum(value_1 * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[2][2], value_2)

    multilayer_perceptron = script.MultilayerPerceptron([784, 16, 16, 10])
    example = np.random.rand(784)
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = multilayer_perceptron.variables[2][1]
    b = multilayer_perceptron.variables[2][0]
    value_2 = relu(np.sum(value_1 * w, axis=1) - b)
    w = multilayer_perceptron.variables[3][1]
    b = multilayer_perceptron.variables[3][0]
    value_3 = sigmoid(np.sum(value_2 * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[3][2], value_3)
